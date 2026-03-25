# core/config_loader.py

import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class IDUQConfig:
    io: Dict[str, Any]
    kinematics: Dict[str, Any]
    perception: Dict[str, Any]
    alignment: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str) -> "IDUQConfig":
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)