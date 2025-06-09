import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=2):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        
        # 缩减通道数的卷积层
        self.query_conv = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=1)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # 计算query、key和value
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x HW x C'
        key = self.key_conv(x).view(batch_size, -1, height * width)                       # B x C' x HW
        value = self.value_conv(x).view(batch_size, -1, height * width)                   # B x C x HW

        # 计算注意力权重
        attention_scores = torch.bmm(query, key)                                          # B x HW x HW
        attention_probs = F.softmax(attention_scores, dim=-1)                            # B x HW x HW

        # 应用注意力权重
        out = torch.bmm(value, attention_probs.permute(0, 2, 1))                         # B x C x HW
        out = out.view(batch_size, channels, height, width)                              # B x C x H x W

        return out

class AttentionBasedDownsampling(nn.Module):
    def __init__(self, channels, target_width):
        super(AttentionBasedDownsampling, self).__init__()
        self.attention = SelfAttention(channels)
        self.target_width = target_width

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # 应用自注意力机制
        attended_x = self.attention(x)
        
        # 选择前target_width个位置的特征
        attended_x = attended_x[:, :, :, :self.target_width]
        
        return attended_x

# 使用示例
if __name__ == "__main__":
    tensor = torch.randn(1, 256, 64, 128)
    downsample_module = AttentionBasedDownsampling(channels=256, target_width=32)
    
    new_tensor = downsample_module(tensor)
    print(new_tensor.shape)  # 输出: torch.Size([1, 256, 64, 64])