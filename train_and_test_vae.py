import torch
import os
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard
from dataset import get_data_transforms, MVTecDataset
from encoder import wide_resnet50_vae
from decoder import de_wide_resnet50_vae
from torchvision.datasets import ImageFolder
from loss_function import loss_fucntion,kl_loss,recon_loss
import tqdm
import numpy as np
from eval_func import evaluation
from utils import get_time_stamp
import os.path as osp



# train 函数
def train(class_, epochs, learning_rate, res, batch_size, print_epoch,
          data_path, save_path, score_num, print_loss,
          layerloss, rate,kl_weight, print_max, net, L2, 
          seed, ckpt_path=None, resume=False, device=0): 
    
    # 时间戳
    time_stamp = get_time_stamp()
    stamp = '_'.join([str(class_),'lr'+str(learning_rate),time_stamp])
    # 创建 log 目录
    log_dir = osp.join('./logs/{0}'.format(osp.join('train', stamp)))
    os.makedirs(log_dir, exist_ok=True)

    # 初始化 TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)

    image_size = 256
    device = 'cuda:{0}'.format(device) if torch.cuda.is_available() else 'cpu' 
    print(device)

    if not os.path.exists(save_path): 
        os.mkdir(save_path)
    
    data_transform, gt_transform = get_data_transforms(image_size, image_size)
    
    train_path = data_path + '/train'
    test_path = data_path
    val_path = data_path             
    ckp_path = save_path 
    
    train_data = ImageFolder(root=train_path, transform=data_transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                                   shuffle=True, num_workers=8)
    
    test_data = MVTecDataset(root=test_path, transform=data_transform, phase="test") 
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, 
                                                  shuffle=False, num_workers=8)
    
    # 验证集
    val_data = MVTecDataset(root=val_path, transform=data_transform, phase="val") 
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=32, 
                                                 shuffle=False, num_workers=8)
    
    # 初始化模型
    encoder = wide_resnet50_vae(pretrained=False)
    decoder = de_wide_resnet50_vae(pretrained=False)
    encoder.to(device)
    decoder.to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=learning_rate, betas=(0.5, 0.999)
    )



    # 断点续训逻辑
    start_epoch = 0
    if resume and ckpt_path:
        checkpoint = torch.load(ckpt_path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch-1}")
    

    
    max_auc = []
    max_auc_epoch = []
    

    for epoch in range(start_epoch, epochs):
        encoder.train() 
        decoder.train()
        loss_list = []
        
        for img, label in tqdm.tqdm(train_dataloader):
            img = img.to(device) 
            inputs = encoder(img)
            outputs = decoder(inputs[3], inputs[0:12], res)
            
            # 根据 layerloss 计算损失
            if layerloss == 0:
                loss = loss_fucntion(inputs[0:3], outputs, L2)[0]  
            elif layerloss == 1:
                loss = loss_fucntion(inputs[0:3], outputs, L2)[0] + \
                       rate * loss_fucntion(inputs[0:3], outputs, L2)[1]
                
            elif layerloss == 2:
                loss = rate*recon_loss(outputs) + kl_weight * kl_loss(inputs)
                # print("using the vae_loss!")
                
            optimizer.zero_grad() 
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item()) 
        avg_loss = np.mean(loss_list)#
        
        # 写入 TensorBoard
        writer.add_scalar('Loss/train', avg_loss , epoch)
        
        if print_loss == 1 and (epoch + 1) % 10 == 0:
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, avg_loss ))
        
        if (epoch + 1) % print_epoch == 0:
            # 评估模型
            auroc_sp, aupr_sp, acc, f1 = evaluation(encoder, decoder, res, val_dataloader, device, score_num)
            print('epoch:', (epoch + 1))
            print('Sample Auroc = {:.4f}'.format(auroc_sp))
            print('Sample Aupr = {:.4f}'.format(aupr_sp))
            print('Sample Acc = {:.4f}'.format(acc))
            print('Sample F1 = {:.4f}'.format(f1))

            max_auc.append(auroc_sp)
            max_auc_epoch.append(epoch + 1)
            
            if print_max == 1:
                print('max_auc = ', max(max_auc))
                print('max_epoch = ', max_auc_epoch[max_auc.index(max(max_auc))])
            print('------------------')
            
            # 写入 TensorBoard
            writer.add_scalar('AUROC/test', auroc_sp, epoch)
            writer.add_scalar('AUPR/test', aupr_sp, epoch)
            writer.add_scalar('ACC/test', acc, epoch)
            writer.add_scalar('f1/test', f1, epoch)

            # 保存模型
            os.makedirs(ckp_path, exist_ok=True)

            # 保存检查点
            save_path = os.path.join(ckp_path,  f"epoch_{epoch+1}_" + f"auc={auroc_sp}.pth")
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, save_path)
    
    # 关闭 TensorBoard writer
    writer.close()
    return auroc_sp 