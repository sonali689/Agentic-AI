import yaml
import os
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str = "config/default_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get_agent_config(self, agent_name: str) -> Dict[str, Any]:
        return self.config['agents'].get(agent_name, {})
    
    def get_path(self, path_name: str) -> str:
        return self.config['paths'].get(path_name, "")