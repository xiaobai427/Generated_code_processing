from copy import deepcopy
from typing import List, Optional, Dict, Any

from models.base import ConfigurationModel, TypeModel
from util import contains_arithmetic_operator


class DataClassifier:
    def __init__(self, data: Dict[str, Any], configuration: Optional[ConfigurationModel] = None):
        self._configuration = deepcopy(configuration)  # 对传入的configuration进行深拷贝
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

    def _is_static_assignment(self) -> bool:
        return not self.params

    @staticmethod
    def _has_consecutive_duplicates(addresses: List[str]) -> bool:
        return any(a == b for a, b in zip(addresses, addresses[1:]))

    def _is_simple_judgment(self) -> bool:
        return self._has_consecutive_duplicates(self.address)

    def _is_parameter_assignment(self) -> bool:
        """检查是否所有的参数赋值都不包含算术运算符。

        返回:
        bool: 如果所有值都不包含算术运算符，则为True，否则为False。
        """
        return self.params and all(not contains_arithmetic_operator(v) for v in self.values)

    def _is_logic_operation(self) -> bool:
        """检查是否任何参数赋值包含算术运算符。

        返回:
        bool: 如果任何值包含算术运算符，则为True，否则为False。
        """
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
        if isinstance(config_items, ConfigurationModel):
            self._set_configuration(config_items)
        else:
            self._set_configuration_from_items(config_items)

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