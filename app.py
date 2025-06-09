import torch
import cv2
import numpy as np
from torchvision import transforms
import gradio as gr
from dataset import get_data_transforms, MVTecDataset
from encoder import wide_resnet50_2,wide_resnet50_vae
from decoder import de_wide_resnet50_2,de_wide_resnet50_vae
import argparse 
from scipy.ndimage import gaussian_filter
from eval_func import evaluation, cal_anomaly_map

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

ckp_path = "./checkpoints/lung_ct/auc=0.7epoch_520.pth"
data_path = "./data/sc2ct/"
img_size = 256
res = 6

# 加载模型权重
encoder = wide_resnet50_vae(pretrained=False)
decoder = de_wide_resnet50_vae(pretrained=False)
checkpoint = torch.load(ckp_path, map_location=device)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.to(device).eval()
decoder.to(device).eval()



def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap


# 定义图像预处理函数
def preprocess_image(image):
    """
    预处理图片：调整大小、归一化、转为 Tensor，并添加 batch 维度。
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),  
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])
    img_tensor = transform(image).unsqueeze(0)  

    return img_tensor

def preprocess_image2(image):

    # 调整图片尺寸为 256x256
    resized_image = cv2.resize(image, (256, 256))
    return resized_image

# 定义推理函数
def inference(image):
    """
    接收单张图片，生成异常图并返回可视化结果。
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 预处理图片
    img_tensor = preprocess_image(image).to(device)

    # 模型推理
    with torch.no_grad():
        inputs = encoder(img_tensor)
        outputs = decoder(inputs[3], inputs[0:12], res=6)

    # 计算异常图
    anomaly_map, a_map_list = cal_anomaly_map(inputs[0:3], outputs[3:6], img_tensor.shape[-1], amap_mode='a')
    anomaly_map = gaussian_filter(anomaly_map, sigma=7)
    ano_map = min_max_norm(anomaly_map)
    ano_map = cvt2heatmap(ano_map * 255)

    # 处理原始图片
    img = cv2.cvtColor(img_tensor.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_RGB2BGR)
    img = np.uint8(min_max_norm(img) * 255)

    # 叠加异常图到原图
    ano_map = show_cam_on_image(img, ano_map)
    ano_map = cv2.cvtColor(ano_map, cv2.COLOR_BGR2RGB)

    return ano_map


# 创建 Gradio 界面
with gr.Blocks() as app:
    gr.Markdown("# 肺CT异常检测与定位")

    # 左右两半布局
    with gr.Row():
        # 左边：上传图片
        with gr.Column():
            gr.Markdown("### 上传图片")
            img_upload = gr.Image(height=600, width=600, label="上传图片")
            upload_btn = gr.Button("上传并预处理")

        # 右边：分为上下两部分
        with gr.Column():
            gr.Markdown("### 图像预处理")
            with gr.Row(): 
                preprocessed_img = gr.Image(height=300, width=300, label="预处理结果")
            gr.Markdown("### 异常定位")
            with gr.Row():  
                synthesized_img = gr.Image(height=300, width=300, label="异常检测结果")


    upload_btn.click(
        fn=preprocess_image2, 
        inputs=img_upload, 
        outputs=preprocessed_img
    )


    synthesize_btn = gr.Button("开始")
    synthesize_btn.click(
        fn = inference, 
        inputs=preprocessed_img, 
        outputs=synthesized_img
    )


app.launch()