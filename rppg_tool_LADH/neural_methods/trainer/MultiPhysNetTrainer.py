"""MultiPhysNet Trainer."""
import os
from collections import OrderedDict
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from evaluation.metrics import calculate_metrics_RR
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.MultiPhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
import pdb
import csv
import matplotlib.pyplot as plt


class MultiPhysNetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.model_name = config.MODEL.NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config

        self.min_valid_loss = None
        self.best_epoch = 0
        self.task = config.TASK
        self.dataset_type = config.DATASET_TYPE  # face face_IR not task
        self.train_state = config.TRAIN.DATA.INFO.STATE
        self.valid_state = config.VALID.DATA.INFO.STATE
        self.test_state = config.TEST.DATA.INFO.STATE
        self.lr = config.TRAIN.LR
        self.log_dir = config.LOG.PATH
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.MultiPhysNet.FRAME_NUM).to(self.device)

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=config.TRAIN.LR, weight_decay=1e-4)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
            self._prepare_spo2_label_histogram(data_loader)
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError(
                "MultiPhysNet trainer initialized in incorrect toolbox mode!")

    def _prepare_spo2_label_histogram(self, data_loader):
        """预计算训练集SpO2标签分布，用于损失中的样本重加权。"""
        self.spo2_bin_edges = None
        self.spo2_bin_weights = None

        if self.task not in ["spo2", "both"]:
            return

        train_loader = data_loader.get("train", None)
        if train_loader is None:
            return
        dataset = getattr(train_loader, "dataset", None)
        if dataset is None or not hasattr(dataset, "labels_spo2"):
            return

        label_paths = getattr(dataset, "labels_spo2", [])
        if len(label_paths) == 0:
            return

        means = []
        for path in label_paths:
            try:
                arr = np.load(path).astype(np.float32).reshape(-1)
                means.append(float(np.mean(arr)))
            except Exception:
                continue

        if len(means) < 16:
            return

        means = np.asarray(means, dtype=np.float32)
        bins = 24
        lo, hi = float(np.min(means)), float(np.max(means))
        if hi - lo < 1e-6:
            return

        edges = np.linspace(lo, hi + 1e-6, bins + 1, dtype=np.float32)
        ids = np.clip(np.digitize(
            means, edges[1:-1], right=False), 0, bins - 1)
        counts = np.bincount(ids, minlength=bins).astype(np.float32)
        median_count = np.median(
            counts[counts > 0]) if np.any(counts > 0) else 1.0

        # 稀有区间更高权重，避免模型塌缩到94附近
        rarity = np.ones_like(counts, dtype=np.float32)
        nz = counts > 0
        rarity[nz] = (median_count / counts[nz]) ** 0.75
        rarity = np.clip(rarity, 0.6, 3.5)

        self.spo2_bin_edges = edges
        self.spo2_bin_weights = rarity

        print("[SpO2 Loss Reweighting]")
        print(f"  Label mean±std: {means.mean():.2f}±{means.std():.2f}")
        print(
            f"  Bin range: {lo:.2f} ~ {hi:.2f} | non-empty bins: {int(np.sum(counts > 0))}/{bins}")

    def _get_spo2_sample_weights(self, spo2_label):
        """根据全局分布给每个样本赋权（稀有标签更高）。"""
        if self.spo2_bin_edges is None or self.spo2_bin_weights is None:
            return torch.ones_like(spo2_label)

        label_np = spo2_label.detach().view(-1).cpu().numpy().astype(np.float32)
        ids = np.clip(
            np.digitize(label_np, self.spo2_bin_edges[1:-1], right=False),
            0,
            len(self.spo2_bin_weights) - 1,
        )
        w = self.spo2_bin_weights[ids]
        w = w / (np.mean(w) + 1e-8)
        w = np.clip(w, 0.6, 3.5)
        return torch.as_tensor(w, dtype=spo2_label.dtype, device=spo2_label.device).view_as(spo2_label)

    def _compute_spo2_loss(self, spo2_pred, spo2_label, multitask=False):
        """SpO2损失：分布重加权回归 + 排序一致性 + 线性差分 + 统计匹配。"""
        sample_w = self._get_spo2_sample_weights(spo2_label)
        sq_err = (spo2_pred - spo2_label) ** 2
        huber = F.smooth_l1_loss(spo2_pred, spo2_label, reduction='none')
        loss_mse = (sample_w * sq_err).mean()
        loss_huber = (sample_w * huber).mean()

        pred_flat = spo2_pred.flatten()
        label_flat = spo2_label.flatten()
        n = pred_flat.shape[0]
        loss_rank = torch.tensor(0.0, device=spo2_pred.device)
        loss_pair = torch.tensor(0.0, device=spo2_pred.device)
        loss_var = torch.tensor(0.0, device=spo2_pred.device)
        loss_mean = torch.tensor(0.0, device=spo2_pred.device)

        if n > 1:
            pred_diff = pred_flat.unsqueeze(
                1) - pred_flat.unsqueeze(0)   # [N, N]
            label_diff = label_flat.unsqueeze(
                1) - label_flat.unsqueeze(0)  # [N, N]

            # 仅在有区分度的标签对上约束
            valid_mask = (label_diff.abs() > 0.1)
            if valid_mask.any():
                target = torch.sign(label_diff[valid_mask])
                # 标签差异越大，margin越大，提升线性区分能力
                adaptive_margin = 0.12 + 0.35 * label_diff[valid_mask].abs()
                rank_violations = torch.clamp(
                    -target * pred_diff[valid_mask] + adaptive_margin, min=0)
                loss_rank = rank_violations.mean()

                # 差分线性损失：不仅顺序正确，还要差值接近
                pair_w = torch.clamp(
                    label_diff[valid_mask].abs(), min=0.1, max=6.0)
                loss_pair = (
                    pair_w * (pred_diff[valid_mask] - label_diff[valid_mask]).abs()).mean() / pair_w.mean()

            pred_var = pred_flat.var()
            label_var = label_flat.var().detach()
            loss_var = (torch.sqrt(pred_var + 1e-6) -
                        torch.sqrt(label_var + 1e-6)) ** 2
            # 均值匹配：抑制整体偏移
            loss_mean = (pred_flat.mean() - label_flat.mean().detach()) ** 2

        base_loss = (
            1.20 * loss_mse +
            0.70 * loss_huber +
            1.60 * loss_rank +
            0.85 * loss_pair +
            0.45 * loss_var +
            0.30 * loss_mean
        )

        if multitask:
            # 提升SpO2任务权重（由数据分析可知标签分布窄，需更强监督）
            return 0.45 * base_loss
        else:
            return base_loss

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []

        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            running_loss_bvp = 0.0
            running_loss_spo2 = 0.0
            running_loss_rr = 0.0
            train_loss = []
            train_loss_bvp = []
            train_loss_spo2 = []
            train_loss_rr = []

            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=120)

            for idx, batch in enumerate(tbar):

                # print(f"batch.shape: {batch[0].shape}")
                # print(f"batch.shape: {batch[1].shape}")
                tbar.set_description("Train epoch %s" % epoch)
                # print(f"batch  : {batch[0].to(torch.float32).to(self.device).shape}")

                loss_bvp = torch.tensor(0.0)
                loss_spo2 = torch.tensor(0.0)
                loss_rr = torch.tensor(0.0)

                if self.dataset_type != "both":
                    if self.task == "bvp":
                        # batch[0] = batch[0].permute(0, 2, 1, 3, 4)
                        rPPG, _, _ = self.model(batch[0].to(
                            torch.float32).to(self.device))
                        BVP_label = batch[1].to(torch.float32).to(self.device)
                        rPPG = (rPPG - torch.mean(rPPG)) / \
                            (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)
                                     ) / (torch.std(BVP_label) + 1e-8)
                        loss = self.loss_model(rPPG, BVP_label)
                        # print(f"\nrPPG Label Info:")
                        # print(f"rPPG label shape: {BVP_label.shape}")
                        # print(f"rPPG label values: {BVP_label[:10]}")  # 打印前10个值
                        # print(f"rPPG label mean: {BVP_label.mean():.4f}")
                        # print(f"rPPG label std: {BVP_label.std():.4f}")
                        # print(f"BVP  loss: {loss}")
                        running_loss_bvp += loss.item()
                    elif self.task == "spo2":
                        _, spo2_pred, _ = self.model(
                            batch[0].to(torch.float32).to(self.device))
                        spo2_label = batch[2].to(torch.float32).to(
                            self.device).squeeze(-1)
                        loss = self._compute_spo2_loss(
                            spo2_pred, spo2_label, multitask=False)
                        running_loss_spo2 += loss.item()
                    elif self.task == "rr":
                        # print(f"RR label values: {batch[3].to(torch.float32).to(self.device)[:10]}")  # 打印前10个值
                        # print(f"batch: {(batch[0].to(torch.float32).to(self.device)).shape}")
                        _, _, rr_pred = self.model(
                            batch[0].to(torch.float32).to(self.device))
                        rr_label = batch[3].to(torch.float32).to(self.device)
                        # print(f"rr_label: {rr_label}")
                        # print(f"rr_pred: {rr_pred}")
                        rr_pred = (rr_pred - torch.mean(rr_pred)) / \
                            (torch.std(rr_pred) + 1e-8)
                        rr_label = (rr_label - torch.mean(rr_label)
                                    ) / (torch.std(rr_label) + 1e-8)

                        # # 添加打印信息
                        # print(f"\nRR Training Info:")
                        # print(f"RR label shape: {rr_label.shape}")
                        # print(f"RR label values: {rr_label[:10]}")  # 打印前10个值
                        # print(f"RR label mean: {rr_label.mean():.4f}")
                        # print(f"RR label std: {rr_label.std():.4f}")

                        loss = self.loss_model(rr_pred, rr_label)
                        # print(f"RR  loss: {loss}")
                        running_loss_rr += loss.item()
                    elif self.task == "both":
                        rPPG, spo2_pred, rr_pred = self.model(
                            batch[0].to(torch.float32).to(self.device))
                        # print(f"batch[0] shape: {batch[0].shape}")
                        # print(f"batch[1] shape: {batch[1].shape}")
                        # print(f"batch[2] shape: {batch[2].shape}")
                        BVP_label = batch[1].to(torch.float32).to(self.device)
                        spo2_label = batch[2].to(torch.float32).to(
                            self.device).squeeze(-1)
                        rr_label = batch[3].to(torch.float32).to(self.device)
                        rPPG = (rPPG - torch.mean(rPPG)) / \
                            (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)
                                     ) / (torch.std(BVP_label) + 1e-8)
                        rr_pred = (rr_pred - torch.mean(rr_pred)) / \
                            (torch.std(rr_pred) + 1e-8)
                        rr_label = (rr_label - torch.mean(rr_label)
                                    ) / (torch.std(rr_label) + 1e-8)
                        loss_bvp = self.loss_model(rPPG, BVP_label)
                        loss_spo2 = self._compute_spo2_loss(
                            spo2_pred, spo2_label, multitask=True)
                        loss_rr = self.loss_model(rr_pred, rr_label)

                        loss = loss_bvp + loss_spo2 + loss_rr
                        running_loss_bvp += loss_bvp.item()
                        running_loss_spo2 += loss_spo2.item()
                        running_loss_rr += loss_rr.item()

                    else:
                        raise ValueError(f"Unknown task: {self.task}")

                else:  # both face and face_IR
                    face_data = batch[0].to(torch.float32).to(self.device)
                    face_IR_data = batch[1].to(torch.float32).to(self.device)

                    if self.task == "bvp":
                        rPPG, _, _ = self.model(face_data, face_IR_data)
                        BVP_label = batch[2].to(torch.float32).to(self.device)
                        rPPG = (rPPG - torch.mean(rPPG)) / \
                            (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)
                                     ) / (torch.std(BVP_label) + 1e-8)
                        loss = self.loss_model(rPPG, BVP_label)
                        running_loss_bvp += loss.item()
                    elif self.task == "spo2":
                        _, spo2_pred, _ = self.model(face_data, face_IR_data)
                        spo2_label = batch[3].to(torch.float32).to(
                            self.device).squeeze(-1)
                        loss = self._compute_spo2_loss(
                            spo2_pred, spo2_label, multitask=False)
                        running_loss_spo2 += loss.item()
                    elif self.task == "rr":
                        _, _, rr_pred = self.model(face_data, face_IR_data)
                        rr_label = batch[4].to(torch.float32).to(
                            self.device).squeeze(-1)

                        # 标准化RR标签和预测值
                        rr_label = (rr_label - rr_label.mean()) / \
                            (rr_label.std() + 1e-8)
                        rr_pred = (rr_pred - rr_pred.mean()) / \
                            (rr_pred.std() + 1e-8)

                        loss = self.loss_model(rr_pred, rr_label)
                        running_loss_rr += loss.item()
                    elif self.task == "both":
                        rPPG, spo2_pred, rr_pred = self.model(
                            face_data, face_IR_data)
                        BVP_label = batch[2].to(torch.float32).to(self.device)
                        spo2_label = batch[3].to(torch.float32).to(
                            self.device).squeeze(-1)
                        rr_label = batch[4].to(torch.float32).to(
                            self.device).squeeze(-1)

                        rPPG = (rPPG - torch.mean(rPPG)) / \
                            (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)
                                     ) / (torch.std(BVP_label) + 1e-8)
                        rr_pred = (rr_pred - torch.mean(rr_pred)) / \
                            (torch.std(rr_pred) + 1e-8)
                        rr_label = (rr_label - torch.mean(rr_label)
                                    ) / (torch.std(rr_label) + 1e-8)

                        loss_bvp = self.loss_model(rPPG, BVP_label)
                        loss_spo2 = self._compute_spo2_loss(
                            spo2_pred, spo2_label, multitask=True)
                        loss_rr = self.loss_model(rr_pred, rr_label)

                        loss = loss_bvp + loss_spo2 + loss_rr

                        running_loss_bvp += loss_bvp.item()
                        running_loss_spo2 += loss_spo2.item()
                        running_loss_rr += loss_rr.item()
                    else:
                        raise ValueError(f"Unknown task: {self.task}")

                self.optimizer.zero_grad()
                loss.backward()
                running_loss += loss.item()
                train_loss.append(loss.item())
                if self.task in ["bvp", "both"]:
                    train_loss_bvp.append(running_loss_bvp)
                if self.task in ["spo2", "both"]:
                    train_loss_spo2.append(running_loss_spo2)
                if self.task in ["rr", "both"]:
                    train_loss_rr.append(running_loss_rr)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()
                lrs.append(self.scheduler.get_last_lr()[0])
            tbar.set_postfix(loss=loss.item(), loss_bvp=running_loss_bvp,
                             loss_spo2=running_loss_spo2, loss_rr=running_loss_rr)
            print(f"train loss: {np.mean(train_loss)}")
            if self.task == "bvp":
                # print(f"train_loss_bvp: {np.mean(train_loss_bvp)}")
                mean_training_losses.append(np.mean(train_loss_bvp))
            if self.task == "spo2":
                mean_training_losses.append(np.mean(train_loss_spo2))
            if self.task == "rr":
                mean_training_losses.append(np.mean(train_loss_rr))
            if self.task == "both":
                mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                loss_csv_path = os.path.join(self.log_dir, 'loss.csv')
                with open(loss_csv_path, 'a', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    data_to_add = [
                        epoch+1, np.mean(train_loss), valid_loss
                    ]
                    csv_writer.writerow(data_to_add)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(
                        self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(
                        self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(
                mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        valid_loss_bvp = 0.0
        valid_loss_spo2 = 0.0
        valid_loss_rr = 0.0
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                if self.dataset_type != "both":  # face or face_IR

                    if self.task == "bvp":
                        # valid_batch[0] = valid_batch[0].permute(0, 2, 1, 3, 4)
                        BVP_label = valid_batch[1].to(
                            torch.float32).to(self.device)
                        rPPG, _, _ = self.model(valid_batch[0].to(
                            torch.float32).to(self.device))
                        rPPG = (rPPG - torch.mean(rPPG)) / \
                            (torch.std(rPPG) + 1e-8)  # normalize
                        BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            (torch.std(BVP_label) + 1e-8)  # normalize
                        loss = self.loss_model(rPPG, BVP_label)
                        valid_loss_bvp += loss.item()
                    elif self.task == "spo2":
                        spo2_label = valid_batch[2].to(
                            torch.float32).to(self.device).squeeze(-1)
                        _, spo2_pred, _ = self.model(
                            valid_batch[0].to(torch.float32).to(self.device))
                        loss = self._compute_spo2_loss(
                            spo2_pred, spo2_label, multitask=False)
                        valid_loss_spo2 += loss.item()
                    elif self.task == "rr":
                        # print(f"RR Validing values: {valid_batch[3].to(torch.float32).to(self.device)[:10]}")  # 打印前10个值
                        rr_label = valid_batch[3].to(
                            torch.float32).to(self.device)
                        # print(f"rr_label: {rr_label}")
                        _, _, rr_pred = self.model(
                            valid_batch[0].to(torch.float32).to(self.device))
                        # print(f"rr_pred: {rr_pred}")
                        # 标准化RR标签和预测值
                        rr_pred = (rr_pred - torch.mean(rr_pred)) / \
                            (torch.std(rr_pred) + 1e-8)
                        rr_label = (rr_label - torch.mean(rr_label)
                                    ) / (torch.std(rr_label) + 1e-8)
                        loss = self.loss_model(rr_pred, rr_label)
                        valid_loss_rr += loss.item()
                        # print(f"\nRR Validing Info:")
                        # print(f"RR label shape: {rr_label.shape}")
                        # print(f"RR label values: {rr_label[:10]}")  # 打印前10个值
                        # print(f"RR label mean: {rr_label.mean():.4f}")
                        # print(f"RR label std: {rr_label.std():.4f}")
                        # print(f"RR  loss: {loss}")
                    else:  # both task
                        # valid_batch[0] = valid_batch[0].permute(0, 2, 1, 3, 4)
                        data = valid_batch[0].to(torch.float32).to(self.device)
                        rPPG, spo2_pred, rr_pred = self.model(data)
                        BVP_label = valid_batch[1].to(
                            torch.float32).to(self.device)
                        spo2_label = valid_batch[2].to(
                            torch.float32).to(self.device).squeeze(-1)
                        rr_label = valid_batch[3].to(
                            torch.float32).to(self.device)

                        rPPG = (rPPG - torch.mean(rPPG)) / \
                            (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)
                                     ) / (torch.std(BVP_label) + 1e-8)
                        rr_pred = (rr_pred - torch.mean(rr_pred)) / \
                            (torch.std(rr_pred) + 1e-8)
                        rr_label = (rr_label - torch.mean(rr_label)
                                    ) / (torch.std(rr_label) + 1e-8)

                        loss_bvp = self.loss_model(rPPG, BVP_label)
                        loss_spo2 = self._compute_spo2_loss(
                            spo2_pred, spo2_label, multitask=True)
                        loss_rr = self.loss_model(rr_pred, rr_label)

                        loss = loss_bvp + loss_spo2 + loss_rr
                        # loss = 600 * loss_bvp + loss_spo2 +  600 * loss_rr
                        # loss = 100 * loss_bvp + loss_spo2 + 100 * loss_rr
                        # print(f"loss_bvp: {loss_bvp}"
                        # print(f"loss_spo2: {loss_spo2}")
                        # print(f"loss_rr: {loss_rr}")
                        valid_loss_bvp += loss_bvp.item()
                        valid_loss_spo2 += loss_spo2.item()
                        valid_loss_rr += loss_rr.item()
                else:  # both
                    face_data = valid_batch[0].to(
                        torch.float32).to(self.device)
                    face_IR_data = valid_batch[1].to(
                        torch.float32).to(self.device)
                    if self.task == "both":
                        rPPG, spo2_pred, rr_pred = self.model(
                            face_data, face_IR_data)
                        BVP_label = valid_batch[2].to(
                            torch.float32).to(self.device)
                        spo2_label = valid_batch[3].to(
                            torch.float32).to(self.device).squeeze(-1)
                        rr_label = valid_batch[4].to(
                            torch.float32).to(self.device)

                        rPPG = (rPPG - torch.mean(rPPG)) / \
                            (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)
                                     ) / (torch.std(BVP_label) + 1e-8)

                        loss_bvp = self.loss_model(rPPG, BVP_label)
                        loss_spo2 = self._compute_spo2_loss(
                            spo2_pred, spo2_label, multitask=True)
                        loss_rr = self.loss_model(rr_pred, rr_label)

                        loss = loss_bvp + loss_spo2 + loss_rr
                        valid_loss_bvp += loss_bvp.item()
                        valid_loss_spo2 += loss_spo2.item()
                        valid_loss_rr += loss_rr.item()
                    elif self.task == "bvp":
                        rPPG, _, _ = self.model(face_data, face_IR_data)
                        BVP_label = valid_batch[2].to(
                            torch.float32).to(self.device)
                        rPPG = (rPPG - torch.mean(rPPG)) / \
                            (torch.std(rPPG) + 1e-8)
                        BVP_label = (BVP_label - torch.mean(BVP_label)
                                     ) / (torch.std(BVP_label) + 1e-8)
                        loss = self.loss_model(rPPG, BVP_label)
                        valid_loss_bvp += loss.item()
                    elif self.task == "spo2":
                        _, spo2_pred, _ = self.model(face_data, face_IR_data)
                        spo2_label = valid_batch[3].to(
                            torch.float32).to(self.device).squeeze(-1)
                        loss = self._compute_spo2_loss(
                            spo2_pred, spo2_label, multitask=False)
                        valid_loss_spo2 += loss.item()
                    elif self.task == "rr":
                        _, _, rr_pred = self.model(face_data, face_IR_data)
                        rr_label = valid_batch[4].to(
                            torch.float32).to(self.device)
                        rr_pred = (rr_pred - torch.mean(rr_pred)) / \
                            (torch.std(rr_pred) + 1e-8)
                        rr_label = (rr_label - torch.mean(rr_label)
                                    ) / (torch.std(rr_label) + 1e-8)
                        loss = self.loss_model(rr_pred, rr_label)
                        valid_loss_rr += loss.item()

                valid_loss.append(loss.item())
                valid_step += 1
                vbar.set_postfix(loss=loss.item(), loss_bvp=valid_loss_bvp,
                                 loss_spo2=valid_loss_spo2, loss_rr=valid_loss_rr)

            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print("\n===Testing===")
        rppg_predictions = dict()
        spo2_predictions = dict()
        rr_predictions = dict()
        rppg_labels = dict()
        spo2_labels = dict()
        rr_labels = dict()
        print(f"dataset_type: {self.dataset_type}")
        # define column names
        header = [
            'V_TYPE', 'TASK', 'LR', 'Epoch Number', 'HR_MAE', 'HR_MAE_STD', 'HR_RMSE', 'HR_RMSE_STD',
            'HR_MAPE', 'HR_MAPE_STD', 'HR_Pearson', 'HR_Pearson_STD', 'HR_SNR', 'HR_SNR_STD',
            'SPO2_MAE', 'SPO2_MAE_STD', 'SPO2_RMSE', 'SPO2_RMSE_STD', 'SPO2_MAPE',
            'SPO2_MAPE_STD', 'SPO2_Pearson', 'SPO2_Pearson_STD', 'SPO2_SNR', 'SPO2_SNR_STD',
            'RR_MAE', 'RR_MAE_STD', 'RR_RMSE', 'RR_RMSE_STD',
            'RR_MAPE', 'RR_MAPE_STD', 'RR_Pearson', 'RR_Pearson_STD', 'RR_SNR', 'RR_SNR_STD',
            'Model', 'train_state', 'valid_state', 'test_state'
        ]

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError(
                    "Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(
                self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print(
                    "Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]

                # print(f"test_batch[3].shape: {test_batch[3].shape}")
                # print(f"test_batch[1]: {test_batch[1]}")
                # print(f"test_batch[2]: {test_batch[2]}")
                # print(f"test_batch[3]: {test_batch[3]}")
                # print(f"test_batch[4]: {test_batch[4]}")
                # print(f"test_batch[5]: {test_batch[5]}")

                if self.dataset_type == "both":
                    face_data = test_batch[0].to(self.config.DEVICE)
                    face_IR_data = test_batch[1].to(self.config.DEVICE)

                    rppg_label = test_batch[2].to(self.config.DEVICE)
                    spo2_label = test_batch[3].to(self.config.DEVICE)
                    rr_label = test_batch[4].to(self.config.DEVICE)

                    pred_ppg_test, pred_spo2_test, pred_rr_test = self.model(
                        face_data, face_IR_data)
                else:
                    # test_batch[0] = test_batch[0].permute(0, 2, 1, 3, 4)
                    data = test_batch[0].to(self.config.DEVICE)
                    rppg_label = test_batch[1].to(self.config.DEVICE)
                    spo2_label = test_batch[2].to(self.config.DEVICE)
                    rr_label = test_batch[3].to(self.config.DEVICE)
                    pred_ppg_test, pred_spo2_test, pred_rr_test = self.model(
                        data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    rppg_label = rppg_label.cpu()
                    spo2_label = spo2_label.cpu()
                    rr_label = rr_label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()
                    pred_spo2_test = pred_spo2_test.cpu()
                    pred_rr_test = pred_rr_test.cpu()

                if self.dataset_type == "both":
                    for idx in range(batch_size):
                        subj_index = test_batch[5][idx]
                        sort_index = (test_batch[6][idx])
                        if subj_index not in rppg_predictions:
                            rppg_predictions[subj_index] = dict()
                            rppg_labels[subj_index] = dict()
                        rppg_predictions[subj_index][sort_index] = pred_ppg_test[idx]
                        rppg_labels[subj_index][sort_index] = rppg_label[idx]

                    for idx in range(batch_size):
                        subj_index = test_batch[5][idx]
                        sort_index = (test_batch[6][idx])
                        if subj_index not in spo2_predictions:
                            spo2_predictions[subj_index] = dict()
                            spo2_labels[subj_index] = dict()
                        spo2_predictions[subj_index][sort_index] = pred_spo2_test[idx]
                        spo2_labels[subj_index][sort_index] = spo2_label[idx]

                    for idx in range(batch_size):
                        subj_index = test_batch[5][idx]
                        sort_index = (test_batch[6][idx])
                        if subj_index not in rr_predictions:
                            rr_predictions[subj_index] = dict()
                            rr_labels[subj_index] = dict()
                        rr_predictions[subj_index][sort_index] = pred_rr_test[idx]
                        rr_labels[subj_index][sort_index] = rr_label[idx]
                else:
                    for idx in range(batch_size):
                        subj_index = test_batch[4][idx]
                        sort_index = int(test_batch[5][idx])
                        if subj_index not in rppg_predictions:
                            rppg_predictions[subj_index] = dict()
                            rppg_labels[subj_index] = dict()
                        rppg_predictions[subj_index][sort_index] = pred_ppg_test[idx]
                        rppg_labels[subj_index][sort_index] = rppg_label[idx]

                    for idx in range(batch_size):
                        subj_index = test_batch[4][idx]
                        sort_index = int(test_batch[5][idx])
                        if subj_index not in spo2_predictions:
                            spo2_predictions[subj_index] = dict()
                            spo2_labels[subj_index] = dict()
                        spo2_predictions[subj_index][sort_index] = pred_spo2_test[idx]
                        spo2_labels[subj_index][sort_index] = spo2_label[idx]

                    for idx in range(batch_size):
                        subj_index = test_batch[4][idx]
                        # print(f"subj_index: {subj_index}")
                        sort_index = int(test_batch[5][idx])
                        # print(f"sort_index: {sort_index}")
                        if subj_index not in rr_predictions:
                            rr_predictions[subj_index] = dict()
                            rr_labels[subj_index] = dict()
                        rr_predictions[subj_index][sort_index] = pred_rr_test[idx]
                        rr_labels[subj_index][sort_index] = rr_label[idx]

        print('')
        result_csv_path = os.path.join(self.log_dir, 'result.csv')
        file_exists = os.path.isfile(result_csv_path)
        with open(result_csv_path, 'a', newline='') as csvfile:
            # inference => How to be more Lupin
            # epoch_num = int(self.config.INFERENCE.MODEL_PATH.split('/')[-1].split('.')[0].split('_')[-1][5:]) + 1
            epoch_num = self.max_epoch_num  # train
            csv_writer = csv.writer(csvfile)

            if not file_exists:
                csv_writer.writerow(header)
            if self.task == "bvp":
                result = calculate_metrics(
                    rppg_predictions, rppg_labels, self.config, "rppg")
                metrics = result["metrics"]
                # MAE RMSE MAPE Pearson SNR
                HR_MAE, HR_MAE_STD = metrics.get("FFT_MAE", (None, None))
                HR_RMSE, HR_RMSE_STD = metrics.get("FFT_RMSE", (None, None))
                HR_MAPE, HR_MAPE_STD = metrics.get("FFT_MAPE", (None, None))
                HR_Pearson, HR_Pearson_STD = metrics.get(
                    "FFT_Pearson", (None, None))
                HR_SNR, HR_SNR_STD = metrics.get(
                    "FFT_SNR", (None, None)) if "FFT_SNR" in metrics else (None, None)

                data_to_add = [
                    self.dataset_type, self.task, self.lr, epoch_num, HR_MAE, HR_MAE_STD, HR_RMSE, HR_RMSE_STD,
                    HR_MAPE, HR_MAPE_STD, HR_Pearson, HR_Pearson_STD, HR_SNR, HR_SNR_STD,
                    "/", "/", "/", "/", "/", "/",
                    "/", "/", "/", "/", "/", "/", "/", "/", "/", "/", "/", "/", "/", "/",
                    self.model_name, self.train_state, self.valid_state, self.test_state
                ]
            elif self.task == "spo2":
                # print("spo2_predictions: ")
                # print(spo2_predictions)
                # print("spo2_labels: ")
                # print(spo2_labels)
                result = calculate_metrics(
                    spo2_predictions, spo2_labels, self.config, "spo2")
                metrics = result["metrics"]
                # MAE RMSE MAPE Pearson
                SPO2_MAE, SPO2_MAE_STD = metrics.get("FFT_MAE", (None, None))
                SPO2_RMSE, SPO2_RMSE_STD = metrics.get(
                    "FFT_RMSE", (None, None))
                SPO2_MAPE, SPO2_MAPE_STD = metrics.get(
                    "FFT_MAPE", (None, None))
                SPO2_Pearson, SPO2_Pearson_STD = metrics.get(
                    "FFT_Pearson", (None, None))
                data_to_add = [
                    self.dataset_type, self.task, self.lr, epoch_num, "/", "/", "/", "/",
                    "/", "/", "/", "/", "/", "/",
                    SPO2_MAE, SPO2_MAE_STD, SPO2_RMSE, SPO2_RMSE_STD, SPO2_MAPE, SPO2_MAPE_STD,
                    SPO2_Pearson, SPO2_Pearson_STD, "/", "/", "/", "/", "/", "/",
                    "/", "/", "/", "/", "/", "/",
                    self.model_name, self.train_state, self.valid_state, self.test_state
                ]

            elif self.task == "rr":
                # print("rr_labels: ")
                # print(rr_labels)
                # print("rr_predictions: ")
                # print(rr_predictions)
                result = calculate_metrics_RR(
                    rr_predictions, rr_labels, self.config, "rr")
                metrics = result["metrics"]
                RR_MAE, RR_MAE_STD = metrics.get("FFT_MAE", (0, 0))
                RR_RMSE, RR_RMSE_STD = metrics.get("FFT_RMSE", (0, 0))
                RR_MAPE, RR_MAPE_STD = metrics.get("FFT_MAPE", (0, 0))
                RR_Pearson, RR_Pearson_STD = metrics.get("FFT_Pearson", (0, 0))
                RR_SNR, RR_SNR_STD = metrics.get(
                    "FFT_SNR", (None, None)) if "FFT_SNR" in metrics else (None, None)

                data_to_add = [
                    self.dataset_type, self.task, self.lr, epoch_num, "/", "/", "/", "/",
                    "/", "/", "/", "/", "/", "/",
                    "/", "/", "/", "/", "/", "/",
                    "/", "/", "/", "/",
                    RR_MAE, RR_MAE_STD, RR_RMSE, RR_RMSE_STD,
                    RR_MAPE, RR_MAPE_STD, RR_Pearson, RR_Pearson_STD, RR_SNR, RR_SNR_STD,
                    self.model_name, self.train_state, self.valid_state, self.test_state
                ]

            elif self.task == "both":
                # 计算BVP指标
                result_rppg = calculate_metrics(
                    rppg_predictions, rppg_labels, self.config, "rppg")
                metrics_rppg = result_rppg["metrics"]
                HR_MAE, HR_MAE_STD = metrics_rppg.get("FFT_MAE", (None, None))
                HR_RMSE, HR_RMSE_STD = metrics_rppg.get(
                    "FFT_RMSE", (None, None))
                HR_MAPE, HR_MAPE_STD = metrics_rppg.get(
                    "FFT_MAPE", (None, None))
                HR_Pearson, HR_Pearson_STD = metrics_rppg.get(
                    "FFT_Pearson", (None, None))
                HR_SNR, HR_SNR_STD = metrics_rppg.get("FFT_SNR", (None, None))

                # 计算SpO2指标
                result_spo2 = calculate_metrics(
                    spo2_predictions, spo2_labels, self.config, "spo2")
                metrics_spo2 = result_spo2["metrics"]
                SPO2_MAE, SPO2_MAE_STD = metrics_spo2.get(
                    "FFT_MAE", (None, None))
                SPO2_RMSE, SPO2_RMSE_STD = metrics_spo2.get(
                    "FFT_RMSE", (None, None))
                SPO2_MAPE, SPO2_MAPE_STD = metrics_spo2.get(
                    "FFT_MAPE", (None, None))
                SPO2_Pearson, SPO2_Pearson_STD = metrics_spo2.get(
                    "FFT_Pearson", (None, None))

                # 计算RR指标
                result_rr = calculate_metrics_RR(
                    rr_predictions, rr_labels, self.config, "rr")
                metrics_rr = result_rr["metrics"]
                RR_MAE, RR_MAE_STD = metrics_rr.get("FFT_MAE", (0, 0))
                RR_RMSE, RR_RMSE_STD = metrics_rr.get("FFT_RMSE", (0, 0))
                RR_MAPE, RR_MAPE_STD = metrics_rr.get("FFT_MAPE", (0, 0))
                RR_Pearson, RR_Pearson_STD = metrics_rr.get(
                    "FFT_Pearson", (0, 0))
                RR_SNR, RR_SNR_STD = metrics_rr.get("FFT_SNR", (None, None))

                data_to_add = [
                    self.dataset_type, self.task, self.lr, epoch_num,
                    HR_MAE, HR_MAE_STD, HR_RMSE, HR_RMSE_STD,
                    HR_MAPE, HR_MAPE_STD, HR_Pearson, HR_Pearson_STD, HR_SNR, HR_SNR_STD,
                    SPO2_MAE, SPO2_MAE_STD, SPO2_RMSE, SPO2_RMSE_STD, SPO2_MAPE, SPO2_MAPE_STD,
                    SPO2_Pearson, SPO2_Pearson_STD, "/", "/",
                    RR_MAE, RR_MAE_STD, RR_RMSE, RR_RMSE_STD,
                    RR_MAPE, RR_MAPE_STD, RR_Pearson, RR_Pearson_STD, RR_SNR, RR_SNR_STD,
                    self.model_name, self.train_state, self.valid_state, self.test_state,
                ]

                # 添加inference模式下的MAE记录
                if self.config.INFERENCE.MODEL_PATH:
                    epoch_number = self.extract_epoch_from_path(
                        self.config.INFERENCE.MODEL_PATH)
                    data_to_add_hr_spo2_rr_MAE = [
                        epoch_number, HR_MAE, SPO2_MAE, RR_MAE

                    ]

        # write data
            if self.config.TOOLBOX_MODE != "only_test":
                csv_writer.writerow(data_to_add)
            else:
                # only_test  hr_spo2_MAE
                mae_csv_path = os.path.join(self.log_dir, 'MAE.csv')
                with open(mae_csv_path, 'a', newline='') as csvf:
                    writer = csv.writer(csvf)
                    writer.writerow(data_to_add_hr_spo2_rr_MAE)

        if self.config.TEST.OUTPUT_SAVE_DIR:  # saving test outputs
            self.save_test_outputs(rppg_predictions, rppg_labels, self.config)
            self.save_test_outputs(spo2_predictions, spo2_labels, self.config)
            self.save_test_outputs(rr_predictions, rr_labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)

    def save_test_outputs(self, predictions, labels, config):
        if not os.path.exists(config.TEST.OUTPUT_SAVE_DIR):
            os.makedirs(config.TEST.OUTPUT_SAVE_DIR)
        output_file = os.path.join(
            config.TEST.OUTPUT_SAVE_DIR, f"{self.model_file_name}_test_outputs.npz")
        np.savez(output_file, predictions=predictions, labels=labels)
        print(f"Saved test outputs to: {output_file}")
    # when inference

    def extract_epoch_from_path(self, model_path):
        print(model_path)
        parts = model_path.split('/')
        for part in parts:
            if 'Epoch' in part:

                a = part.find("Epoch")
                b = part.find(".pth")
                epoch_str = part[a+5:b]
                return int(epoch_str)+1
        raise ValueError("The model path does not contain an epoch number.")


def plot_rr_wave_2(rr_pred, rr_label):

    rr_pred = np.array(rr_pred)
    rr_label = np.array(rr_label)
    # Plotting
    plt.figure(figsize=(20, 6))
    plt.plot(rr_pred, label="Predicted RR", color='blue', linewidth=1.5)
    plt.plot(rr_label, label="True RR", color='red', linewidth=1.5)

    plt.title("RR ")
    plt.xlabel("number")
    plt.ylabel("RR Rate")
    plt.legend()
    plt.savefig('./2222.png')
    plt.show()


def plot_rr_wave_1(rr_pred, rr_lable):

    rr_pred = np.array(rr_pred)
    rr_lable = np.array(rr_lable)

    fig, axes = plt.subplots(2, 1, figsize=(20, 15))
    axes[0].plot(rr_pred, label='RR Pred')
    axes[0].set_title('RR Data Plot')
    axes[0].set_xlabel('Number')
    axes[0].set_ylabel('Data Value')
    axes[0].legend()

    # # plot the filtered RR data and the power spectral density
    axes[1].plot(rr_lable, label='RR Lable')
    axes[1].set_title('RR Data Plot')
    axes[1].set_xlabel('Number')
    axes[1].set_ylabel('Data Value')
    axes[1].legend()

    plt.savefig('./1111.png')
    plt.show()
