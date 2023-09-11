import json
import pathlib

from utils.dataclass import Config


def load_config(config_path: pathlib.Path) -> Config:
    """configのロード

    Args:
        config_path (pathlib.Path): config.jsonのパス

    Returns:
        Config: config.jsonの内容を登録したConfigクラス
    """
    with config_path.open("r") as f:
        d = json.load(f)
    return Config(**d)
