from lib.dataset.mapfree import MapFreeDataset, MapFreeScene
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from yacs.config import CfgNode as CN


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: CN, drop_last_val: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.drop_last_val = drop_last_val
        self.dataset_type = MapFreeDataset

    def val_dataloader(self):
        dataset = self.dataset_type(self.cfg, "val")
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.TRAINING.BATCH_SIZE,
            num_workers=self.cfg.TRAINING.NUM_WORKERS,
            sampler=None,
            drop_last=self.drop_last_val,
        )
        return dataloader
