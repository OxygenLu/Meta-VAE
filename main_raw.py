from train_and_test_vae_pre import train
from eval_func import setup_seed

seeds = [42, 142, 250]
# data_path = "/home/data3t/luxijun/document/Skip-TS/data/sc2ct-Mix"
# data_path = "/home/data3t/luxijun/document/SimSID/data/sc2ct/" # 
# data_path = "/home/data3t/luxijun/document/Skip-TS/data/head_ct"
# data_path = "/home/data3t/luxijun/document/Dataset/sc2ct_70/"
data_path = "/home/data3t/luxijun/document/Dataset/adluna"
# 50:res6, 60:skip-ts

if __name__ == '__main__':
    for seed in seeds:
        setup_seed(seed)# 随机种子
        max_auroc = train(class_ = 'luna_raw',   
                    epochs = 200,
                    learning_rate = 0.05 ,#0.002->res=6
                    res = 3,      
                    batch_size = 24,
                    print_epoch = 10,
                    data_path = data_path,
                    save_path = '/home/data3t/luxijun/document/Skip-TS/checkpoints/raw_luna',#loss1
                    score_num = 1, 
                    print_loss = 1,
                    layerloss = 1,
                    rate = 0.000001,
                    kl_weight=0.00015, 
                    print_max = 1,
                    net = 'res50_2',
                    L2 = 1, 
                    seed = seed,
                    # resume=True,
                    # ckpt_path='./checkpoints/lung_ct_vae_exp1new/auc=0.24epoch_460.pth',
                    device=1,
                    )
        print(f"Seed {seed} | Max AUROC: {max_auroc}")
