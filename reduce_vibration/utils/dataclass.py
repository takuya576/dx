from dataclasses import dataclass


@dataclass
class Config:
    net: str
    weights: bool
    transfer: bool
    lr: float
    momentum: float
    num_epochs: int
    batch_size: int
    nvidia: int
    num_val: int
    which_data: str
    train_data: str
    test_data: str
    generated: bool
    alpha: float
    beta: float
    a: float
    sigmoid: bool
