import json
import pathlib

import toml

from utils.dataclass import Config


def load_config(config_path: pathlib.Path) -> Config:
    """configのロード

    Args:
        config_path (pathlib.Path): config.tomlのパス

    Returns:
        Config: config.tomlの内容を登録したConfigクラス
    """
    with config_path.open("r") as f:
        d = toml.load(f)
    return Config(**d)
