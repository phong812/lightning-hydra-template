from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from omegaconf import DictConfig
import torch



class GAN_DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        train_val_test_split: Tuple[int, int, int] = (50000, 10000, 10000),
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        self.batch_size_per_device = batch_size
        
        self.save_hyperparameters(logger=False)
        self.transforms = transforms.Compose([transforms.ToTensor()])
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
    
    def num_classes(self) -> int:
        return 10
    
    def prepare_data(self) -> None:
        MNIST(self.hparams.data_dir, train=True, download=True)
        MNIST(self.hparams.data_dir, train=False, download=True)
        
    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = MNIST(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = MNIST(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            print(dataset)
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )
        print(self.data_train)
            
    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        
    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        
    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
    
if __name__ == "__main__":
    import pyrootutils
    from omegaconf import DictConfig
    import hydra
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    
    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")
    config_path = str(path / "configs" / "data")
    output_path = str(path / "outputs")
    print("root", path, config_path)
        
    def im_show(img):
            
        batch_size = img.shape[0]
        plt.figure(figsize=(8, 8))
        for i in range(batch_size):
            plt.subplot(4, 8, i + 1)
            plt.imshow(img[i, 0], cmap="gray")
            plt.axis("off")
            
    plt.show()
        
    def test_datamodule(cfg: DictConfig):
        datamodule: GAN_DataModule = hydra.utils.instantiate(cfg)
        datamodule.prepare_data()
        datamodule.set_up()
        loader = datamodule.test_dataloader()
            
        bx = next(iter(loader))
        im_show(bx)
        
        for bx in tqdm(datamodule.train_dataloader()):
            pass
        print("training data passed")
        
        for bx in tqdm(datamodule.val_dataloader()):
            pass
        print("validation data passed")
        
    @hydra.main(version_base="1.3", config_path=config_path, config_name="gan_dataset.yaml")
    def main(cfg: DictConfig) -> None:
        print(cfg)
        test_datamodule(cfg)
    
    main()