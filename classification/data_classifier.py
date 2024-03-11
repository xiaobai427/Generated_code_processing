from copy import deepcopy
from typing import List, Optional, Dict, Any, Type, Callable

from models.base import ConfigurationModel, TypeModel, ActionItemModel, TrimActionItemModel
from util import contains_arithmetic_operator, get_most_common_type

class HandlerRegistry:
    def __init__(self):
        self._registry = {}

    def register(self, model_type: Any, handler: Callable):
        self._registry[model_type] = handler

    def get_handler(self, model_type: Any):
        handler = self._registry.get(model_type)
        if not handler:
            raise ValueError(f"No handler registered for type: {model_type}")
        return handler


class DataClassifier:
    def __init__(self, data: Dict[str, Any], configuration: Optional[ConfigurationModel] = None):
        self._configuration = deepcopy(configuration)
        self.data = data
        self.values = []
        self.address = []
        self.params = None
        self.judgment_to_type = [
            (self._is_function_encapsulation, 'type_function_encapsulation'),
            (self._is_static_assignment, 'type_static_assignment'),
            (self._is_simple_judgment, 'type_simple_judgment'),
            (self._is_parameter_assignment, 'type_parameter_assignment'),
            (self._is_logic_operation, 'type_logic_operation'),
        ]

        # 初始化处理函数注册器
        self.handler_registry = HandlerRegistry()
        # 注册处理函数
        self.handler_registry.register(ConfigurationModel, self._set_configuration)
        self.handler_registry.register(ActionItemModel, self._set_configuration_from_items)
        self.handler_registry.register(TrimActionItemModel, self._set_configuration_from_items_Trim)

    def _is_static_assignment(self) -> bool:
        return not self.params

    @staticmethod
    def _has_consecutive_duplicates(addresses: List[str]) -> bool:
        return any(a == b for a, b in zip(addresses, addresses[1:]))

    def _is_simple_judgment(self) -> bool:
        return self._has_consecutive_duplicates(self.address)

    def _is_parameter_assignment(self) -> bool:
        return self.params and all(not contains_arithmetic_operator(v) for v in self.values)

    def _is_logic_operation(self) -> bool:
        return self.params and any(contains_arithmetic_operator(v) for v in self.values)

    def _is_function_encapsulation(self) -> bool:
        return bool(self._configuration and self._configuration.sub_function)

    def process_data(self) -> TypeModel:
        model = TypeModel()
        for key, config_items in self.data.items():
            self._prepare_data(config_items)
            for judgment, type_attr in self.judgment_to_type:
                if judgment():
                    getattr(model, type_attr).append(key)
                    break
        return model

    def _prepare_data(self, config_items):
        # 通过注册表获取处理函数，并调用之
        common_type = get_most_common_type(config_items)
        if common_type is not None:
            handler = self.handler_registry.get_handler(common_type)
            if handler:
                handler(config_items)
            else:
                print(f"No handler registered for {common_type}")
        else:
            print("Invalid input: config_items is neither a type nor a non-empty list.")

    def _set_configuration(self, configuration: ConfigurationModel):
        self.values = configuration.values
        self.address = configuration.address
        self.params = configuration.params
        self._configuration = deepcopy(configuration)

    def _set_configuration_from_items(self, items):
        parser_list = [str(value.params) for value in items if hasattr(value, 'params')]
        parsed_list = [eval(item) for item in parser_list if eval(item)]
        self.params = parsed_list[0] if parsed_list else {}
        if self._configuration:
            self.values = self._configuration.values[:len(items)]
            self.address = self._configuration.address[:len(items)]
            self._configuration.sub_function = {}

    def _set_configuration_from_items_Trim(self, items):
        parsed_list = self._configuration.params
        self.params = parsed_list[0] if parsed_list else {}
        if self._configuration:
            self.values = self._configuration.values[:len(items)]
            self.address = self._configuration.address[:len(items)]
            self._configuration.sub_function = {}
