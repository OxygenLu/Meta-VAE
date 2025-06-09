
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import torch.nn as nn
import torch
from attention import SelfAttention, AttentionBasedDownsampling
# import attention

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
def deconv2x2(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride,
                              groups=groups, bias=False, dilation=dilation)


class DeBottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(DeBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        if stride == 2:
            self.conv2 = deconv2x2(width, width, stride, groups, dilation)
        else:
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out

class DeResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[DeBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(DeResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 512 * block.expansion
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DeBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[DeBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x,y,res) -> Tensor:
        if (res == 1):
          feature_a = self.layer1(x)  
          feature_b = self.layer2(feature_a)  
          feature_c = self.layer3(feature_b) 
        
        if (res == 2):
          feature_a = self.layer1(x)
          feature_b = self.layer2(self.relu(feature_a+y[2]))
          feature_c = self.layer3(feature_b)
        
        if (res ==3):
          feature_a = self.layer1(x)
          feature_b = self.layer2(self.relu(feature_a+y[2]))
          feature_c = self.layer3(self.relu(feature_b+y[1]))
         
        return [feature_c, feature_b, feature_a]  
    def forward(self, x,y,res) -> Tensor:
        return self._forward_impl(x,y,res)



    def forward(self, x,y,res) -> Tensor:
        return self._forward_impl(x,y,res)

def _deresnet(
    arch: str,
    block: Type[Union[DeBottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> DeResNet:
    model = DeResNet(block, layers, **kwargs)
    return model


def de_wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DeResNet:
    kwargs['width_per_group'] = 64 * 2
    return _deresnet('wide_resnet50_2', DeBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


# decoder_vae
class DeResNet_decoder(nn.Module):

    def __init__(
        self,
        block: Type[Union[DeBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
         
    ) -> None:
        super(DeResNet_decoder, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.channels = 16
        self.inplanes = 512 * block.expansion
        self.dilation = 1
        self.latent_dim = 128
        # self.target_width = target_width

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        self.relu = nn.ReLU(inplace=True)


        # self.fc_decode = nn.Linear(self.latent_dim, self.target_width)
    

        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, DeBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[DeBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                deconv2x2(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, upsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧，用于采样潜在变量"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    

    def decode_latent(self, z: torch.Tensor, channels:int, target_width:int) -> torch.Tensor:
        """从潜在变量重建特征图"""
        fc_decode = nn.Linear(128, target_width).to(device)
        x = fc_decode(z)
        downsample_module = AttentionBasedDownsampling(channels=channels, target_width=target_width).to(device)
        x = downsample_module(x)
        return x

    def forward(self, x,y,res) -> Tensor:
        if (res == 1):
            feature_a = self.layer1(x)  
            feature_b = self.layer2(feature_a)  
            feature_c = self.layer3(feature_b) 
        
        elif (res == 2):
            feature_a = self.layer1(x)
            feature_b = self.layer2(self.relu(feature_a+y[2]))
            feature_c = self.layer3(feature_b)
        
        elif(res == 3):
            # print(x.shape)
            feature_a = self.layer1(x)
            feature_b = self.layer2(self.relu(feature_a+y[2]))
            feature_c = self.layer3(self.relu(feature_b+y[1]))

        # vae
        elif(res == 4):
           # print(x.shape)
            # 计算均值和对数方差
            """
            feature_a,  feature_b,  feature_c,  feature_d, # feature
            fc_mua,     fc_mub,     fc_muc,     fc_mud,    # mu
            fc_logvara, fc_logvarb, fc_logvarc, fc_logvard
            """

            # 重参数化
            # z1 = self.reparameterize(y[4], y[8])
            z2 = self.reparameterize(y[5], y[9])
            z3 = self.reparameterize(y[6], y[10])
            z4 = self.reparameterize(y[7], y[11])
            # print(f"x:{x.shape},z4:{z4.shape},z3:{z3.shape}")
            # 从潜在变量重建特征图
            recon4 = self.decode_latent(z4, 1024, 16)#1, 1024, 16, 16
            recon3 = self.decode_latent(z3, 512, 32)#1, 512, 32, 32
            recon2 = self.decode_latent(z2, 256, 64)#1, 512, 32, 32
            # recon4 = self.decode_latent(z4)
            # print(f"re4:{recon4.shape},re3:{recon3.shape},re2:{recon2.shape}")

            feature_a = self.layer1(x)# ->1, 1024, 16, 16
            # feature_a = feature_a+recon4# ->1, 1024, 16, 16

            feature_b = self.layer2(self.relu(feature_a+recon4+y[2]))# ->1, 512, 32, 32
            # feature_b = feature_b+recon3# ->1, 512, 32, 32

            feature_c = self.layer3(self.relu(feature_b+recon3+y[1]))#->1, 256, 64, 64
            # feature_c = feature_c+recon2

        elif(res == 5):
            # print(x.shape)
            # 计算均值和对数方差
            """
            feature_a,  feature_b,  feature_c,  feature_d, # feature
            fc_mua,     fc_mub,     fc_muc,     fc_mud,    # mu
            fc_logvara, fc_logvarb, fc_logvarc, fc_logvard
            """

            # 重参数化
            # z1 = self.reparameterize(y[4], y[8])
            z2 = self.reparameterize(y[5], y[9])
            z3 = self.reparameterize(y[6], y[10])
            z4 = self.reparameterize(y[7], y[11])
            # print(f"x:{x.shape},z4:{z4.shape},z3:{z3.shape}")
            # 从潜在变量重建特征图
            recon4 = self.decode_latent(z4, 1024, 16)#1, 1024, 16, 16
            recon3 = self.decode_latent(z3, 512, 32)#1, 512, 32, 32
            recon2 = self.decode_latent(z2, 256, 64)#1, 512, 32, 32
            # recon4 = self.decode_latent(z4)
            # print(f"re4:{recon4.shape},re3:{recon3.shape},re2:{recon2.shape}")

            feature_a = self.layer1(x)# ->1, 1024, 16, 16
            # feature_a = feature_a+recon4# ->1, 1024, 16, 16

            feature_b = self.layer2(self.relu(feature_a+recon4))# ->1, 512, 32, 32
            # feature_b = feature_b+recon3# ->1, 512, 32, 32

            feature_c = self.layer3(self.relu(feature_b+recon3))#->1, 256, 64, 64
            # feature_c = feature_c+recon2

        elif(res == 6):
            # print(x.shape)
            # 计算均值和对数方差
            """
            feature_a,  feature_b,  feature_c,  feature_d, # feature
            fc_mua,     fc_mub,     fc_muc,     fc_mud,    # mu
            fc_logvara, fc_logvarb, fc_logvarc, fc_logvard
            """

            # 重参数化
            # z1 = self.reparameterize(y[4], y[8])
            z2 = self.reparameterize(y[5], y[9])
            z3 = self.reparameterize(y[6], y[10])
            z4 = self.reparameterize(y[7], y[11])
            # print(f"x:{x.shape},z4:{z4.shape},z3:{z3.shape}")

            # 从潜在变量重建特征图
            recon4 = self.decode_latent(z4, 1024, 16)#1, 1024, 16, 16
            recon3 = self.decode_latent(z3, 512, 32)#1, 512, 32, 32
            recon2 = self.decode_latent(z2, 256, 64)#1, 256, 64, 64
            # recon4 = self.decode_latent(z4)
            # print(f"re4:{recon4.shape},re3:{recon3.shape},re2:{recon2.shape}")

            feature_a = self.layer1(x)# ->1, 1024, 16, 16
            # feature_a = feature_a+recon4# ->1, 1024, 16, 16

            feature_b = self.layer2(self.relu(feature_a+y[2]))# ->1, 512, 32, 32
            # feature_b = feature_b+recon3# ->1, 512, 32, 32

            feature_c = self.layer3(self.relu(feature_b+y[1]))#->1, 256, 64, 64
            # feature_c = feature_c+recon2

        return [feature_c, feature_b, feature_a, 
                recon2, recon3, recon4]


def vae_decoder(
    arch: str,
    block: Type[Union[DeBottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> DeResNet_decoder:
    model = DeResNet_decoder(block, layers, **kwargs)
    return model


def de_wide_resnet50_vae(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DeResNet_decoder:
    kwargs['width_per_group'] = 64 * 2
    return vae_decoder('wide_resnet50_2', DeBottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)