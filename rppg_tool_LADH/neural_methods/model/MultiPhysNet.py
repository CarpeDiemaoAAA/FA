import math
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple

class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),  # 压缩通道
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),  # 恢复通道
            nn.Sigmoid()  # 激活函数，生成通道权重
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 压缩空间维度
        y = self.fc(y).view(b, c, 1, 1, 1)  # 恢复形状
        return x * y  # 通道注意力加权

class IR_SE_CNN(nn.Module):
    def __init__(self, input_channels=3):
        super(IR_SE_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(16),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(32),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            SEBlock(64),
        )

    def forward(self, x):
        return self.features(x)

# class FusionNet(nn.Module):
#     def __init__(self):
#         super(FusionNet, self).__init__()
#         self.conv1 = nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm3d(64)
#         self.conv3 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)  

#     def forward(self, x1, x2):
#         x = torch.cat((x1, x2), dim=1)
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.conv3(x)  
#         return x

class FusionNet(nn.Module):
    def __init__(self, feature_channels=64):
        super(FusionNet, self).__init__()
        self.rgb_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(feature_channels, feature_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.ir_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(feature_channels, feature_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels)
        )
        
        # 通道注意力Refinement，提升融合后特征质量
        self.channel_refine = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(feature_channels, feature_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels // 4, feature_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_rgb, x_ir):
        # Gating: 输出每个模态的权重 [B, 1, 1, 1, 1]
        gate_rgb = self.rgb_gate(x_rgb)
        gate_ir = self.ir_gate(x_ir)
        # 归一化门控权重，确保两个模态的贡献之和为1
        gate_sum = gate_rgb + gate_ir + 1e-8
        gate_rgb = gate_rgb / gate_sum
        gate_ir = gate_ir / gate_sum

        fused = gate_rgb * x_rgb + gate_ir * x_ir

        # 残差连接 + 通道注意力精炼
        fused_out = self.fuse_conv(fused)
        channel_att = self.channel_refine(fused_out)
        fused_out = fused_out * channel_att
        fused_out = fused_out + fused  # 残差连接
        return fused_out
    
    
class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            SEBlock(16)
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            SEBlock(32)
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            SEBlock(64)
        )


        self.rppg_branch = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),            
            nn.ELU(),
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
            nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0),
        )

        self.rr_branch = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.AdaptiveAvgPool3d((frames, 1, 1)),
            nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0) 
        )
        
        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        # SpO2路径1：从rPPG波形提取特征（保留原始路径）
        self.spo2_from_rppg = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
        )
        
        # SpO2路径2：直接从编码器视觉特征提取
        self.spo2_from_visual = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # SpO2路径3：从IR编码器特征直接提取（红外光吸收率与血氧浓度直接相关）
        self.spo2_from_ir = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # SpO2融合预测头：128(rppg) + 32(visual) + 32(ir) = 192
        self.spo2_head = nn.Sequential(
            nn.Linear(128 + 32 + 32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.ir_encoder = IR_SE_CNN()
        self.fusion_net = FusionNet()
        
        # SpO2相关层的Kaiming初始化
        self._init_spo2_weights()

    def _init_spo2_weights(self):
        """对SpO2相关模块进行Kaiming初始化，提升训练起点质量"""
        for module_list in [self.spo2_from_rppg, self.spo2_from_visual, self.spo2_from_ir, self.spo2_head]:
            for m in module_list.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2=None):  # Batch_size*[3, T, 128,128]
        [batch, channel, length, width, height] = x1.shape   
        x2_ir_feat = None  # 保存IR编码器原始特征，供SpO2专用路径使用
        if x2 is not None:
            x1_visual = self.share_m(x1)
            x2_ir_feat = self.ir_encoder(x2)
            
            x = self.fusion_net(x1_visual, x2_ir_feat)
            # x = self.encode_video_x(x)

            # x = self.fusion_net(x1, x2)
            # x = self.encode_video_x(x)
            # x = self.encode_video(x)
        else:
            # x = self.share_m(x1)
            # rPPG = self.encode_video_rppg(x)
            # rr = self.encode_video_rr(x)
            x = self.encode_video(x1)
        # print(f"x.shape: {x.shape}")
        

        rPPG = self.rppg_branch(x)
        rPPG = rPPG.view(batch, length)
        rr = self.rr_branch(x)
        rr = rr.view(batch, length)

        # SpO2三路径预测：
        # 路径1-从rPPG波形（不再detach，允许SpO2损失反传优化编码器）
        spo2_feat_rppg = self.spo2_from_rppg(rPPG.unsqueeze(1))  # [B, 128]
        # 路径2-从融合视觉特征
        spo2_feat_visual = self.spo2_from_visual(x)  # [B, 32]
        # 路径3-从IR编码器特征（IR对SpO2最敏感，双模态时直接利用）
        if x2_ir_feat is not None:
            spo2_feat_ir = self.spo2_from_ir(x2_ir_feat)  # [B, 32]
        else:
            spo2_feat_ir = torch.zeros(batch, 32, device=x.device)
        
        spo2_feat = torch.cat([spo2_feat_rppg, spo2_feat_visual, spo2_feat_ir], dim=1)  # [B, 192]
        spo2 = self.spo2_head(spo2_feat)
        spo2 = spo2.view(batch, 1)
        spo2 = spo2 * 15 + 85

        return rPPG, spo2, rr

    def encode_video(self, x):
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock8(x)
        return x
    
    def share_m(self, x):
        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock2(x)
        x = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock4(x)
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = nn.AdaptiveAvgPool3d((64, 18, 18))(x)   # [8, 64, 64, 18, 18]
        return x

    def encode_video_x(self, x):
        x = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x)
        x = self.ConvBlock6(x)
        x = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x)
        x = self.ConvBlock8(x)
        return x