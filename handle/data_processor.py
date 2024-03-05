import json
from typing import Dict, Optional, List, Any

from models.base import ConfigurationModel, ActionItemModel, TrimActionItemModel


class DataProcessor:
    def __init__(self, configurations: Dict[str, Dict]):
        self.configurations = configurations
        self.processed_configurations = {}  # 用于存储处理后的配置

    def process(self) -> Dict[str, ConfigurationModel]:
        for name, config in self.configurations.items():
            actions = [self.convert_action_item(action) for action in config.get('actions', [])]
            sub_functions = {k: [self.convert_action_item(action) for action in v] for k, v in
                             config.get('sub_function', {}).items()}

            configuration_model = ConfigurationModel(
                params=config.get('params', []),
                actions=actions,
                sub_function=sub_functions,
                values=config.get('values', []),
                address=config.get('address', [])
            )
            self.processed_configurations[name] = configuration_model
        return self.processed_configurations

    @staticmethod
    def convert_action_item(action: Dict) -> ActionItemModel:
        if 'params' in action and isinstance(action['params'], str):
            try:
                action['params'] = json.loads(action['params'].replace("'", "\""))
            except json.JSONDecodeError:
                print(f"Warning: Could not convert params to dictionary for action: {action}")
                action['params'] = {}
        if 'flag' in action:
            return TrimActionItemModel(**action)
        else:
            return ActionItemModel(**action)

    def fetch_deep_attribute_values(self, key: str, attribute: str, action_attribute: Optional[str] = None) -> List[
        Any]:
        results = []
        configuration = self.processed_configurations.get(key)

        if not configuration:
            print(f"No configuration found for key: {key}")
            return results

        if attribute not in ['actions', 'sub_function']:
            attribute_value = getattr(configuration, attribute, [])
            return list(attribute_value) if isinstance(attribute_value, (list, dict)) else [attribute_value]

        items_to_process = []
        if attribute == 'actions':
            items_to_process.extend(configuration.actions)
        elif attribute == 'sub_function' and action_attribute:
            for sub_actions in configuration.sub_function.values():
                items_to_process.extend(sub_actions)

        for item in items_to_process:
            if hasattr(item, action_attribute):
                results.append(getattr(item, action_attribute, None))

        return results