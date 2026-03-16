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
            nn.Conv3d(feature_channels, feature_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(feature_channels, feature_channels,
                      kernel_size=3, padding=1),
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

        # SpO2双通道：基于Beer-Lambert法则，保留时间维度捕获AC/DC脉搏变化
        # 路径1：从RGB编码器提取时空特征（保留4个时间步以捕获AC分量信息）
        self.spo2_rgb_temporal = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 1, 1)),
            nn.Flatten(),
            nn.Linear(64 * 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
        )

        # 路径2：从IR编码器提取时空特征（IR通道的AC/DC比与血氧浓度直接相关）
        self.spo2_ir_temporal = nn.Sequential(
            nn.AdaptiveAvgPool3d((4, 1, 1)),
            nn.Flatten(),
            nn.Linear(64 * 4, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
        )

        # 路径3：RGB-IR交互特征，模拟R = (AC_red/DC_red) / (AC_ir/DC_ir) 比率关系
        self.spo2_cross = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )

        # SpO2预测头：32(rgb) + 32(ir) + 16(cross) = 80
        # 不使用Sigmoid：避免输出压缩到窄范围，直接回归SpO2偏移量
        self.spo2_head = nn.Sequential(
            nn.Linear(80, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

        self.ir_encoder = IR_SE_CNN()
        self.fusion_net = FusionNet()

        # SpO2相关层的Kaiming初始化
        self._init_spo2_weights()

    def _init_spo2_weights(self):
        """对SpO2相关模块进行Kaiming初始化，提升训练起点质量"""
        for module_list in [self.spo2_rgb_temporal, self.spo2_ir_temporal, self.spo2_cross, self.spo2_head]:
            for m in module_list.modules():
                if isinstance(m, (nn.Linear, nn.Conv1d)):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2=None):  # Batch_size*[3, T, 128,128]
        [batch, channel, length, width, height] = x1.shape
        x2_ir_feat = None  # 保存IR编码器原始特征，供SpO2专用路径使用
        x1_visual = None   # 保存RGB编码器预融合特征，供SpO2光学比率路径使用
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

        # SpO2双通道预测：基于时空特征+交叉比率的直接回归
        if x1_visual is not None and x2_ir_feat is not None:
            # 双模态：分别提取RGB和IR的时空特征（保留时间维度捕获AC/DC变化）
            spo2_feat_rgb = self.spo2_rgb_temporal(x1_visual)   # [B, 32]
            spo2_feat_ir = self.spo2_ir_temporal(x2_ir_feat)    # [B, 32]
            # 交叉特征：乘积捕获比率关系 + 差异捕获吸收差异
            cross_input = torch.cat([spo2_feat_rgb * spo2_feat_ir,
                                     # [B, 64]
                                     spo2_feat_rgb - spo2_feat_ir], dim=1)
            spo2_cross_feat = self.spo2_cross(cross_input)  # [B, 16]
        else:
            # 单模态fallback
            spo2_feat_rgb = self.spo2_rgb_temporal(x)  # [B, 32]
            spo2_feat_ir = torch.zeros(batch, 32, device=x.device)
            spo2_cross_feat = torch.zeros(batch, 16, device=x.device)

        spo2_feat = torch.cat(
            [spo2_feat_rgb, spo2_feat_ir, spo2_cross_feat], dim=1)  # [B, 80]
        spo2 = self.spo2_head(spo2_feat)
        spo2 = spo2.view(batch, 1)
        # 直接回归：以典型SpO2均值为中心，无Sigmoid压缩
        spo2 = 94.0 + spo2
        spo2 = torch.clamp(spo2, 70.0, 100.0)  # 安全范围约束

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
