from dataclasses import dataclass


@dataclass
class Config:
    nvidia: int
    num_val: int
    net: str
    weights: bool
    transfer: bool
    lr: float
    momentum: float
    num_epochs: int
    batch_size: int
    which_data: str
    train_data: str
    test_data: str
