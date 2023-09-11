from dataclasses import dataclass


@dataclass
class Config:
    net: str
    pretrained: bool
    lr: float
    momentum: float
    num_epochs: int
    batch_size: int
    nvidia: int
    num_val: int
    which_data: str
    train_data_1: str
    train_data_2: str
    test_data: str
