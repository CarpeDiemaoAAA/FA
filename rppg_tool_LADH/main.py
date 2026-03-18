""" The main function of rPPG deep learning pipeline."""

import argparse
import random
import time

import numpy as np
import torch
from config import get_config
from dataset import data_loader
from neural_methods import trainer
from unsupervised_methods.unsupervised_predictor import unsupervised_predict
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Create a general generator for use with the validation dataloader,
# the test dataloader, and the unsupervised dataloader
general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
# Create a training generator to isolate the train dataloader from
# other dataloaders and better control non-deterministic behavior
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)


def seed_worker(worker_seed=RANDOM_SEED):
    # worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_spo2_balanced_sampler(dataset, bins=24, rarity_pow=0.7, std_boost=0.4):
    """根据SpO2标签分布构建加权采样器。

    - 稀有SpO2区间：更高采样权重
    - chunk内标签方差更高：轻度增权（更有学习价值）
    """
    if not hasattr(dataset, 'labels_spo2') or len(dataset.labels_spo2) == 0:
        return None

    means = []
    stds = []
    for path in dataset.labels_spo2:
        try:
            arr = np.load(path).astype(np.float32).reshape(-1)
            means.append(float(np.mean(arr)))
            stds.append(float(np.std(arr)))
        except Exception:
            means.append(94.0)
            stds.append(0.0)

    means = np.asarray(means, dtype=np.float32)
    stds = np.asarray(stds, dtype=np.float32)

    if len(means) < 4:
        return None

    lo, hi = float(np.min(means)), float(np.max(means))
    if hi - lo < 1e-6:
        return None

    hist_edges = np.linspace(lo, hi + 1e-6, bins + 1, dtype=np.float32)
    bin_ids = np.clip(np.digitize(
        means, hist_edges[1:-1], right=False), 0, bins - 1)
    bin_counts = np.bincount(bin_ids, minlength=bins).astype(np.float32)
    median_count = np.median(bin_counts[bin_counts > 0]) if np.any(
        bin_counts > 0) else 1.0

    rarity_weight = np.ones_like(means, dtype=np.float32)
    for i in range(len(means)):
        count = max(bin_counts[bin_ids[i]], 1.0)
        rarity_weight[i] = (median_count / count) ** float(rarity_pow)

    # chunk内SpO2方差增益：低方差不抛弃，只是降低优先级
    std_norm = stds / (np.percentile(stds, 95) + 1e-6)
    std_factor = 1.0 + float(std_boost) * np.clip(std_norm, 0.0, 1.0)

    weights = rarity_weight * std_factor
    weights = weights / (np.mean(weights) + 1e-8)
    weights = np.clip(weights, 0.35, 4.0)

    print("\n[SpO2 Balanced Sampling]")
    print(
        f"  Samples: {len(weights)} | SpO2 mean±std: {means.mean():.2f}±{means.std():.2f}")
    print(f"  Weight range: {weights.min():.3f} ~ {weights.max():.3f}")

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def add_args(parser):
    """Adds arguments for parser."""
    parser.add_argument('--config_file', required=False,
                        default="configs/train_configs/PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml", type=str, help="The name of the model.")
    '''Neural Method Sample YAML LIST:
      SCAMPS_SCAMPS_UBFC-rPPG_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_UBFC-rPPG_PHYSNET_BASIC.yaml
      SCAMPS_SCAMPS_PURE_DEEPPHYS_BASIC.yaml
      SCAMPS_SCAMPS_PURE_TSCAN_BASIC.yaml
      SCAMPS_SCAMPS_PURE_PHYSNET_BASIC.yaml
      PURE_PURE_UBFC-rPPG_TSCAN_BASIC.yaml
      PURE_PURE_UBFC-rPPG_DEEPPHYS_BASIC.yaml
      PURE_PURE_UBFC-rPPG_PHYSNET_BASIC.yaml
      PURE_PURE_MMPD_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_TSCAN_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_DEEPPHYS_BASIC.yaml
      UBFC-rPPG_UBFC-rPPG_PURE_PHYSNET_BASIC.yaml
      MMPD_MMPD_UBFC-rPPG_TSCAN_BASIC.yaml
    Unsupervised Method Sample YAML LIST:
      PURE_UNSUPERVISED.yaml
      UBFC-rPPG_UNSUPERVISED.yaml
    '''
    parser.add_argument('--model_path', required=False,
                        type=str, help="Path to the model for inference.")
    parser.add_argument('--epochs', required=False,
                        type=str, help="control epochs")
    parser.add_argument('--r_lr', required=False,
                        type=float, help="lr.")
    parser.add_argument('--path', required=False,
                        type=str, help="save path.")
    return parser


def train_and_test(config, data_loader_dict):
    """Trains the model."""
    if config.MODEL.NAME == "MultiPhysNet":
        model_trainer = trainer.MultiPhysNetTrainer.MultiPhysNetTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(
            config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    """Tests the model."""
    if config.MODEL.NAME == "MultiPhysNet":
        model_trainer = trainer.MultiPhysNetTrainer.MultiPhysNetTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == "iBVPNet":
        model_trainer = trainer.iBVPNetTrainer.iBVPNetTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == "Tscan":
        model_trainer = trainer.TscanTrainer.TscanTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == "EfficientPhys":
        model_trainer = trainer.EfficientPhysTrainer.EfficientPhysTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == 'DeepPhys':
        model_trainer = trainer.DeepPhysTrainer.DeepPhysTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == 'BigSmall':
        model_trainer = trainer.BigSmallTrainer.BigSmallTrainer(
            config, data_loader_dict)
    elif config.MODEL.NAME == 'PhysFormer':
        model_trainer = trainer.PhysFormerTrainer.PhysFormerTrainer(
            config, data_loader_dict)
    else:
        raise ValueError('Your Model is Not Supported  Yet!')
    model_trainer.test(data_loader_dict)


def unsupervised_method_inference(config, data_loader):
    if not config.UNSUPERVISED.METHOD:
        raise ValueError("Please set unsupervised method in yaml!")
    for unsupervised_method in config.UNSUPERVISED.METHOD:
        if unsupervised_method == "POS":
            unsupervised_predict(config, data_loader, "POS")
        elif unsupervised_method == "CHROM":
            unsupervised_predict(config, data_loader, "CHROM")
        elif unsupervised_method == "ICA":
            unsupervised_predict(config, data_loader, "ICA")
        elif unsupervised_method == "GREEN":
            unsupervised_predict(config, data_loader, "GREEN")
        elif unsupervised_method == "LGI":
            unsupervised_predict(config, data_loader, "LGI")
        elif unsupervised_method == "PBV":
            unsupervised_predict(config, data_loader, "PBV")
        elif unsupervised_method == "OMIT":
            unsupervised_predict(config, data_loader, "OMIT")
        else:
            raise ValueError("Not supported unsupervised method!")


if __name__ == "__main__":
    # parse arguments.
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()
    # If the model_path parameter is provided, it overrides the value in the configuration file.
    # configurations.
    config = get_config(args)
    # print('Configuration:')
    # print(config, end='\n\n')
    if args.epochs:
        config['TRAIN']['EPOCHS'] = int(args.epochs)
    if args.r_lr:
        config['TRAIN']['LR'] = float(args.r_lr)
    if args.path:
        config['LOG']['PATH'] = args.path

    data_loader_dict = dict()  # dictionary of data loaders
    if config.TOOLBOX_MODE == "train_and_test":
        # train_loader
        if config.TRAIN.DATA.DATASET == "UBFC-rPPG":
            train_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.TRAIN.DATA.DATASET == "PURE":
            train_loader = data_loader.PURELoader.PURELoader
        elif config.TRAIN.DATA.DATASET == "SCAMPS":
            train_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TRAIN.DATA.DATASET == "MMPD":
            train_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TRAIN.DATA.DATASET == "BP4DPlus":
            train_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.TRAIN.DATA.DATASET == "BP4DPlusBigSmall":
            train_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.TRAIN.DATA.DATASET == "UBFC-PHYS":
            train_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.TRAIN.DATA.DATASET == "iBVP":
            train_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.TRAIN.DATA.DATASET == "SUMS":
            train_loader = data_loader.SUMSLoader.SUMSLoader
        elif config.TRAIN.DATA.DATASET == "LADH":
            train_loader = data_loader.LADHLoader.LADHLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")

        # Create and initialize the train dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset paths
        if (config.TRAIN.DATA.DATASET and config.TRAIN.DATA.DATA_PATH):
            print("train_loader:\n")
            # print(train_loader)
            print("\n")
            train_data_loader = train_loader(
                name="train",
                data_path=config.TRAIN.DATA.DATA_PATH,
                config_data=config.TRAIN.DATA)

            use_spo2_balanced_sampler = bool(
                config.TASK in ["spo2", "both"] and
                config.TRAIN.SPO2_BALANCED_SAMPLING
            )
            train_sampler = None
            if use_spo2_balanced_sampler:
                train_sampler = build_spo2_balanced_sampler(
                    dataset=train_data_loader,
                    bins=config.TRAIN.SPO2_SAMPLING_BINS,
                    rarity_pow=config.TRAIN.SPO2_SAMPLING_POW,
                    std_boost=config.TRAIN.SPO2_SAMPLING_STD_BOOST,
                )

            data_loader_dict['train'] = DataLoader(
                dataset=train_data_loader,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                worker_init_fn=seed_worker,
                generator=train_generator
            )
        else:
            data_loader_dict['train'] = None

        # valid_loader
        if config.VALID.DATA.DATASET == "UBFC-rPPG":
            valid_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.VALID.DATA.DATASET == "PURE":
            valid_loader = data_loader.PURELoader.PURELoader
        elif config.VALID.DATA.DATASET == "SCAMPS":
            valid_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.VALID.DATA.DATASET == "MMPD":
            valid_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.VALID.DATA.DATASET == "BP4DPlus":
            valid_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.VALID.DATA.DATASET == "BP4DPlusBigSmall":
            valid_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.VALID.DATA.DATASET == "UBFC-PHYS":
            valid_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.VALID.DATA.DATASET == "iBVP":
            valid_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.VALID.DATA.DATASET == "SUMS":
            valid_loader = data_loader.SUMSLoader.SUMSLoader
        elif config.VALID.DATA.DATASET == "LADH":
            valid_loader = data_loader.LADHLoader.LADHLoader
        elif config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
            raise ValueError(
                "Validation dataset not specified despite USE_LAST_EPOCH set to False!")
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP")

        # Create and initialize the valid dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if (config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH):
            valid_data = valid_loader(
                name="valid",
                data_path=config.VALID.DATA.DATA_PATH,
                config_data=config.VALID.DATA)
            data_loader_dict["valid"] = DataLoader(
                dataset=valid_data,
                num_workers=16,
                batch_size=config.TRAIN.BATCH_SIZE,  # batch size for val is the same as train
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['valid'] = None

    if config.TOOLBOX_MODE == "train_and_test" or config.TOOLBOX_MODE == "only_test":
        if args.model_path:
            config['INFERENCE']['MODEL_PATH'] = args.model_path
            print(f"Using model path: {config['INFERENCE']['MODEL_PATH']}")

        # test_loader
        if config.TEST.DATA.DATASET == "UBFC-rPPG":
            test_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.TEST.DATA.DATASET == "PURE":
            test_loader = data_loader.PURELoader.PURELoader
        elif config.TEST.DATA.DATASET == "SCAMPS":
            test_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.TEST.DATA.DATASET == "MMPD":
            test_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.TEST.DATA.DATASET == "BP4DPlus":
            test_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.TEST.DATA.DATASET == "BP4DPlusBigSmall":
            test_loader = data_loader.BP4DPlusBigSmallLoader.BP4DPlusBigSmallLoader
        elif config.TEST.DATA.DATASET == "UBFC-PHYS":
            test_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.TEST.DATA.DATASET == "iBVP":
            test_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.TEST.DATA.DATASET == "SUMS":
            test_loader = data_loader.SUMSLoader.SUMSLoader
        elif config.TEST.DATA.DATASET == "LADH":
            test_loader = data_loader.LADHLoader.LADHLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+ (Normal and BigSmall preprocessing), UBFC-PHYS and iBVP.")

        if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
            print(
                "Testing uses last epoch, validation dataset is not required.", end='\n\n')

        # Create and initialize the test dataloader given the correct toolbox mode,
        # a supported dataset name, and a valid dataset path
        if config.TEST.DATA.DATASET and config.TEST.DATA.DATA_PATH:
            test_data = test_loader(
                name="test",
                data_path=config.TEST.DATA.DATA_PATH,
                config_data=config.TEST.DATA)
            data_loader_dict["test"] = DataLoader(
                dataset=test_data,
                num_workers=16,
                batch_size=config.INFERENCE.BATCH_SIZE,
                shuffle=False,
                worker_init_fn=seed_worker,
                generator=general_generator
            )
        else:
            data_loader_dict['test'] = None

    elif config.TOOLBOX_MODE == "unsupervised_method":
        # unsupervised method dataloader
        if config.UNSUPERVISED.DATA.DATASET == "UBFC-rPPG":
            unsupervised_loader = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
        elif config.UNSUPERVISED.DATA.DATASET == "PURE":
            unsupervised_loader = data_loader.PURELoader.PURELoader
        elif config.UNSUPERVISED.DATA.DATASET == "SCAMPS":
            unsupervised_loader = data_loader.SCAMPSLoader.SCAMPSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "MMPD":
            unsupervised_loader = data_loader.MMPDLoader.MMPDLoader
        elif config.UNSUPERVISED.DATA.DATASET == "BP4DPlus":
            unsupervised_loader = data_loader.BP4DPlusLoader.BP4DPlusLoader
        elif config.UNSUPERVISED.DATA.DATASET == "UBFC-PHYS":
            unsupervised_loader = data_loader.UBFCPHYSLoader.UBFCPHYSLoader
        elif config.UNSUPERVISED.DATA.DATASET == "iBVP":
            unsupervised_loader = data_loader.iBVPLoader.iBVPLoader
        elif config.UNSUPERVISED.DATA.DATASET == "SUMS":
            unsupervised_loader = data_loader.SUMSLoader.SUMSLoader
        elif config.UNSUPERVISED.TRAIN.DATA.DATASET == "LADH":
            unsupervised_loader = data_loader.LADHLoader.LADHLoader
        else:
            raise ValueError("Unsupported dataset! Currently supporting UBFC-rPPG, PURE, MMPD, \
                             SCAMPS, BP4D+, UBFC-PHYS and iBVP.")

        unsupervised_data = unsupervised_loader(
            name="unsupervised",
            data_path=config.UNSUPERVISED.DATA.DATA_PATH,
            config_data=config.UNSUPERVISED.DATA)
        data_loader_dict["unsupervised"] = DataLoader(
            dataset=unsupervised_data,
            num_workers=16,
            batch_size=1,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=general_generator
        )

    else:
        raise ValueError(
            "Unsupported toolbox_mode! Currently support train_and_test or only_test or unsupervised_method.")

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "unsupervised_method":
        unsupervised_method_inference(config, data_loader_dict)
    else:
        print("TOOLBOX_MODE only support train_and_test or only_test !", end='\n\n')
