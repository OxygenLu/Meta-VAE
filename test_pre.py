import os
import torch
from torch.utils.data import DataLoader
from dataset import MVTecDataset, get_data_transforms
from encoder import wide_resnet50_2,wide_resnet50_vae
from decoder import de_wide_resnet50_2,de_wide_resnet50_vae
from eval_func import evaluation
import argparse
from utils import get_time_stamp

def test_all_checkpoints(
    class_name,
    ckpt_dir,  # 修改为接收目录路径
    data_path,
    res,
    score_num,
    device_id=0,
    batch_size=32,
    image_size=256
):
    device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建保存结果的目录
    test_time = get_time_stamp()
    save_dir = f"test_results/{class_name}_{test_time}"
    os.makedirs(save_dir, exist_ok=True)

    # 数据预处理
    data_transform, _ = get_data_transforms(image_size, image_size)
    
    # 加载测试数据集
    test_dataset = MVTecDataset(
        root=data_path,
        transform=data_transform,
        phase="val",
        # class_name=class_name
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    # 初始化模型结构（仅需定义一次）
    encoder = wide_resnet50_2(pretrained=True)
    decoder = de_wide_resnet50_2(pretrained=False)
    # encoder = wide_resnet50_vae(pretrained=False)
    # decoder = de_wide_resnet50_vae(pretrained=False)
    # encoder_path='./checkpoints/vae_res6_loss2_final/auc=0.7epoch_520.pth'
    # encoder_ckpt = torch.load(encoder_path, map_location=device)
    # encoder.load_state_dict(encoder_ckpt['encoder'])
    encoder.to(device).eval()

    # 遍历目录中的所有检查点文件
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    all_results = []


    for ckpt_name in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print(f"\nTesting model: {ckpt_name}")

        # 加载模型权重
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        decoder.load_state_dict(checkpoint['decoder'])
        
        decoder.to(device).eval()

        # 禁用梯度计算
        with torch.no_grad():
            auroc_sp, aupr_sp = evaluation(
                encoder,
                decoder,
                res,
                test_loader,
                device,
                score_num
            )

        # 记录结果
        all_results.append({
            "model_name": ckpt_name,
            "auroc": auroc_sp,
            "aupr": aupr_sp
        })
        print(f"  AUROC: {auroc_sp:.4f} | AUPR: {aupr_sp:.4f}")

    # 保存所有结果
    result_path = os.path.join(save_dir, f"all_results_{class_name}.txt")
    with open(result_path, 'w') as f:
        f.write(f"Class: {class_name}\n")
        f.write(f"Total models tested: {len(all_results)}\n\n")
        for res in all_results:
            f.write(f"Model: {res['model_name']}\n")
            f.write(f"  AUROC: {res['auroc']:.4f}\n")
            f.write(f"  AUPR: {res['aupr']:.4f}\n")
            f.write("-------------------------------\n")

        # 找出最佳模型
        best_auroc = max(all_results, key=lambda x: x['auroc'])
        best_aupr = max(all_results, key=lambda x: x['aupr'])
        f.write(f"\nBest AUROC Model: {best_auroc['model_name']} | AUROC: {best_auroc['auroc']:.4f}\n")
        f.write(f"Best AUPR Model: {best_aupr['model_name']} | AUPR: {best_aupr['aupr']:.4f}\n")

    print(f"All results saved to {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test all checkpoints in a directory")
    # parser.add_argument("--class_name", type=str, required=True, help="Name of the product class")
    parser.add_argument("--class_name", default="sc2ct_raw_mix60",type=str,  help="Name of the product class")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/raw_res6_loss2_mix60", help="Path to the directory containing checkpoints")
    parser.add_argument("--data_path", default="/home/data3t/luxijun/document/Dataset/sc2ct_60" ,type=str, help="Path to the dataset")
    parser.add_argument("--res", type=int, default=3, help="Resolution parameter for the model")
    parser.add_argument("--score_num", type=int, default=1, help="Number of scores to compute")
    parser.add_argument("--device_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for testing")
    args = parser.parse_args()

    test_all_checkpoints(
        class_name=args.class_name,
        ckpt_dir=args.ckpt_dir,
        data_path=args.data_path,
        res=args.res,
        score_num=args.score_num,
        device_id=args.device_id,
        batch_size=args.batch_size
    )