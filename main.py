import csv
import json
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union, Tuple
import re

from debug import Test_DataClassifier


def adjust_params(data):
    """
    Adjusts the 'params' field in a nested dictionary structure based on whether the 'value'
    is a key in 'params'. If 'value' is not a key, sets 'params' to "{}".

    :param data: The nested dictionary structure containing the instructions and their parameters.
    :return: The modified dictionary with adjusted 'params'.
    """
    for key, instructions in data.items():
        for instruction in instructions:
            # Convert string representation of dict to dict
            params = eval(instruction['params'])
            # Check if the 'value' is a key in 'params'
            if instruction['value'] not in params:
                instruction['params'] = "{}"
            else:
                # Ensure params is properly formatted as a string
                instruction['params'] = str(params)
    return data


class PseudocodeParser:
    def __init__(self):
        self.functions = {}
        self.current_sub_function = {'name': None, 'active': False}

    def handle_start(self, action, function_name, sub_func_key):
        self.current_sub_function['name'] = sub_func_key
        self.current_sub_function['active'] = True
        self.functions[function_name]['sub_function'].setdefault(sub_func_key, []).append(action)

    def handle_end(self, action, function_name):
        if self.current_sub_function['name']:
            self.functions[function_name]['sub_function'][self.current_sub_function['name']].append(action)
            self.current_sub_function['name'] = None
            self.current_sub_function['active'] = False

    def handle_none(self, action, function_name, sub_func_key):
        if self.current_sub_function['active']:
            # Append to the current sub_function if active
            self.functions[function_name]['sub_function'][self.current_sub_function['name']].append(action)
        elif sub_func_key:
            # This case handles encapsulation without start/end explicitly mentioned
            self.functions[function_name]['sub_function'].setdefault(sub_func_key, []).append(action)
        else:
            # Normal action handling
            self.functions[function_name]['actions'].append(action)

    def parse_encapsulation(self, action, function_name):
        encapsulation = action.get('encapsulation', '')
        sub_func_key = encapsulation.split(': ')[-1].strip() if encapsulation else None

        if 'start' in encapsulation:
            self.handle_start(action, function_name, sub_func_key)
        elif 'end' in encapsulation:
            self.handle_end(action, function_name)
        else:
            self.handle_none(action, function_name, sub_func_key)

    def parse_csv_to_pseudocode(self, file_path):
        with open(file_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)

            for row in reader:
                if row and row[0].strip().startswith('function Name'):
                    function_name = row[1].strip()
                    headers = next(reader)

                    valid_fields = [field for field in headers if field.lower() != 'na']
                    field_to_index = {field: index for index, field in enumerate(headers) if field.lower() != 'na'}
                    para_names_map = {field: field.replace('para_', '') for field in valid_fields if
                                      field.startswith('para')}

                    self.functions[function_name] = {'params': list(para_names_map.values()),
                                                     'actions': [],
                                                     'sub_function': {},
                                                     'values': [],
                                                     'address': []
                                                     }

                    for func_row in reader:
                        if not func_row or not func_row[0].strip():
                            break

                        action = {field: func_row[index] for field, index in field_to_index.items() if
                                  not field.startswith('para')}
                        params = {para_names_map[field]: func_row[index] for field, index in field_to_index.items() if
                                  field.startswith('para') and func_row[index] != '/'}

                        if params:
                            action['params'] = str(params)

                        action['value'] = re.sub(r'\bpara_([a-zA-Z0-9_]+)\b', r'\1', action.get('value', ''))
                        address = action.get('address', None)
                        self.functions[function_name]['values'].append(action['value'])
                        self.functions[function_name]['address'].append(address)
                        self.parse_encapsulation(action, function_name)

            # Post-process to adjust the structure
            for func in self.functions.values():
                if func['sub_function']:
                    func['sub_function'] = adjust_params(func['sub_function'])
                else:
                    del func['sub_function']

            return self.functions


class TypeModel(BaseModel):
    type_static_assignment: List[str] = Field(default_factory=list)
    type_simple_judgment: List[str] = Field(default_factory=list)
    type_parameter_assignment: List[str] = Field(default_factory=list)
    type_logic_operation: List[str] = Field(default_factory=list)
    type_function_encapsulation: List[str] = Field(default_factory=list)


# Action item model for actions and sub-functions
class ActionItemModel(BaseModel):
    step: str
    instruction: str
    address: str
    value: str
    encapsulation: Optional[str] = None
    comment: Optional[str] = None
    params: Optional[Dict[str, str]] = None

    def __str__(self):
        return (f"Step: {self.step}, Instruction: {self.instruction}, Address: {self.address}, "
                f"Value: {self.value}, Encapsulation: {self.encapsulation}, Comment: {self.comment}, "
                f"Params: {self.params}")


class SubFunctionModel(BaseModel):
    items: List[ActionItemModel] = Field(default_factory=list)

    def __str__(self):
        items_str = ", ".join(str(item) for item in self.items)
        return f"SubFunction Items: [{items_str}]"


class ConfigurationModel(BaseModel):
    params: List[str] = Field(default_factory=list)
    actions: List[ActionItemModel] = Field(default_factory=list)
    sub_function: Dict[str, List[ActionItemModel]] = Field(default_factory=dict)
    values: List[str] = Field(default_factory=list)
    address: List[str] = Field(default_factory=list)

    def __str__(self):
        params_str = ", ".join(self.params)
        actions_str = "; ".join(str(action) for action in self.actions)
        sub_functions_str = "; ".join(
            f"{k}: [{', '.join(str(v) for v in vals)}]" for k, vals in self.sub_function.items())
        values_str = ", ".join(self.values)
        address_str = ", ".join(self.address)

        return (f"Params: [{params_str}], Actions: [{actions_str}], "
                f"SubFunctions: [{sub_functions_str}], Values: [{values_str}], "
                f"Address: [{address_str}]")


class DataClassifier:
    _configuration: ConfigurationModel

    def __init__(self, data: Dict[str, Any], configuration: ConfigurationModel = None):
        self._configuration = None
        self.data = data
        self.configuration = configuration
        self.values = []
        self.address = []
        self.params = None
        self.judgment_to_type = [
            (self.is_function_encapsulation, 'type_function_encapsulation'),
            (self.is_static_assignment, 'type_static_assignment'),
            (self.is_simple_judgment, 'type_simple_judgment'),
            (self.is_parameter_assignment, 'type_parameter_assignment'),
            (self.is_logic_operation, 'type_logic_operation'),
        ]

    def is_static_assignment(self) -> bool:
        return not bool(self.params)

    @staticmethod
    def has_consecutive_duplicates(addresses: List[str]) -> bool:
        return any(addresses[i] == addresses[i + 1] for i in range(len(addresses) - 1))

    def is_simple_judgment(self) -> bool:
        return self.has_consecutive_duplicates(self.address)

    def is_parameter_assignment(self) -> bool:
        return self.params and all(not re.search(r'[\+\-\*/]', v) for v in self.values)

    def is_logic_operation(self) -> bool:
        return self.params and any(re.search(r'[\+\-\*/]', v) for v in self.values)

    def is_function_encapsulation(self) -> bool:
        return bool(self._configuration.sub_function)

    def process_data(self) -> TypeModel:
        model = TypeModel()
        for key, configuration_items in self.data.items():
            if isinstance(configuration_items, ConfigurationModel):
                self.values = configuration_items.values
                self.address = configuration_items.address
                self.params = configuration_items.params
                self._configuration = configuration_items
            else:
                parser_list = []
                self.values = self.configuration.values[:len(configuration_items)]
                self.address = self.configuration.address[:len(configuration_items)]
                for i, value in enumerate(configuration_items):
                    parser_list.append(str(value.params))
                parsed_list = [eval(item) for item in parser_list if eval(item)]
                self.params = parsed_list[0] if len(parsed_list) == 1 else {}
                self.configuration.sub_function = {}
                self._configuration = self.configuration
            for judgment, type_attr in self.judgment_to_type:
                if judgment():
                    getattr(model, type_attr).append(key)
                    break  # 如果需要一个项目只能归类到一个类型，添加break; 否则，删除break以允许多重分类
        return model


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


def find_common_elements_to_params(params, values, addresses, element_type='uint8_t'):
    combined_expressions = ' '.join(values)

    # 使用正则表达式提取变量名
    # 假设变量名由字母、下划线或数字组成，并且不以数字开头
    variables_in_expressions = set(re.findall(r'\b[a-zA-Z_][a-zA-Z_0-9]*\b', combined_expressions))

    # 找出两个集合的共有元素
    common_elements = set(params).intersection(variables_in_expressions)

    # 为每个共有元素指定类型
    common_elements_with_type = {element: element_type for element in common_elements}

    if not common_elements_with_type:
        if not common_elements and any(addresses[i] == addresses[i + 1] for i in range(len(addresses) - 1)):
            return {element: element_type for element in params}

    return common_elements_with_type


# 策略接口
class ActionStrategy(ABC):
    @abstractmethod
    def execute_action(self,
                       action: ActionItemModel,
                       params: Optional[Dict[str, str]] = None,
                       state: bool = None) -> List[str]:
        raise NotImplementedError

    def generate_function_signature(self, action_item: ActionItemModel, function_name: str, return_type: str,
                                    class_name: str,
                                    params: Optional[Dict[str, str]], state=None) -> str:
        param_str = ", ".join([f"{type_} {name}" for name, type_ in params.items()]) if params else ""
        if action_item.value in params.keys():
            param_str = param_str
        else:
            param_str = ''
        function_signature = f"{return_type} {class_name}::{function_name}({param_str})"
        pseudocode_lines = [f"{function_signature}\n{{"]
        self.execute_action(action_item, params)
        action_lines = self.execute_action(action_item, params, state)
        pseudocode_lines.extend(action_lines)
        pseudocode_lines.append("}\n")
        return "\n".join(pseudocode_lines)


# 静态赋值策略
class StaticAssignmentStrategy(ActionStrategy):
    def execute_action(self,
                       action: ActionItemModel,
                       params: Optional[Dict[str, str]] = None,
                       state: bool = None) -> List[str]:
        return [f"\treg_write({action.address}, {action.value}); // {action.comment}"]


class SimpleJudgmentStrategy(ActionStrategy):
    def execute_action(self, action: ActionItemModel, params: Optional[Dict[str, str]] = None,
                       state: bool = None) -> List[str]:

        lines = []
        condition = f"if (enable == {action.params['enable']})"
        if action.params['enable'] and action.instruction == "write":
            if state:  # 使用 else if 替换之后的 if 条件
                condition = "else " + condition

            action_line = f"\treg_write({action.address}, {action.value}); // {action.comment}"
            lines.append(f"\t{condition}")
            lines.append(f"\t{{")
            lines.append(f"\t{action_line}")
            lines.append(f"\t}}")
        elif action.instruction == "delay":
            lines.append(f"\tSleep({action.address} {action.value}); // {action.comment}")
        return lines


# 参数赋值策略
class ParameterAssignmentStrategy(ActionStrategy):
    def execute_action(self,
                       action: ActionItemModel,
                       params: Optional[Dict[str, str]] = None,
                       state: bool = None) -> List[str]:
        param_name = action.value.strip("{}")
        return [f"\treg_write({action.address}, {param_name}); // {action.comment}"]


class LogicOperationStrategy(ActionStrategy):
    def execute_action(self,
                       action: ActionItemModel,
                       params: Optional[Dict[str, str]] = None,
                       state: bool = None) -> List[str]:
        lines = []
        if action.instruction == 'write':
            # 处理包含逻辑运算的写入指令
            evaluated_value = self._evaluate_logic_expression(action.value, params)
            line = f"\treg_write({action.address}, {evaluated_value}); // {action.comment}"
            lines.append(line)
        elif action.instruction == 'delay':
            # 处理延时
            lines.append(f"\tSleep({action.address}, {action.value}); // {action.comment}")
        return lines

    @staticmethod
    def _evaluate_logic_expression(expression: str, params: Dict[str, str]) -> str:
        # 这个方法用于将表达式中的参数替换为实际的参数值，并返回处理后的表达式
        for param_name, param_type in params.items():
            if param_name in expression:
                # 这里简单地替换参数名为其类型表示，需要根据实际情况进行调整
                expression = expression.replace(param_name, param_type)
        return expression


class FunctionEncapsulationStrategy:
    @staticmethod
    def execute_action(configuration: ConfigurationModel, params: Optional[Dict[str, str]] = None) -> List[str]:
        main_function_lines = []
        sub_functions_code = {}

        # 处理每个子功能
        for sub_function_name, actions in configuration.sub_function.items():
            sub_function_lines = []
            for action in actions:
                # 根据动作类型生成代码行
                if action.instruction == "write":
                    # 这里仅示例处理写入指令，其他指令类型可类似处理
                    line = f"\treg_write({action.address}, {action.value}); // {action.comment}"
                    sub_function_lines.append(line)
            # 保存子函数代码
            sub_functions_code[sub_function_name] = sub_function_lines

        # 生成主函数调用子函数的代码
        for sub_function_name in configuration.sub_function.keys():
            call_line = f"\trf_{sub_function_name}();"
            main_function_lines.append(call_line)

        # 将子函数代码转换为字符串形式
        sub_functions_code_str = []
        for name, lines in sub_functions_code.items():
            sub_function_def = f"void DRIVER_KUNLUN::rf_{name}()\n{{\n" + "\n".join(lines) + "\n}\n"
            sub_functions_code_str.append(sub_function_def)

        return main_function_lines + [""] + sub_functions_code_str


class ActionStrategyFactory:
    def __init__(self, type_model: TypeModel):
        self.strategies = {
            "simple_judgment": SimpleJudgmentStrategy(),
            "logic_operation": LogicOperationStrategy(),
            "static_assignment": StaticAssignmentStrategy(),
            "parameter_assignment": ParameterAssignmentStrategy(),
            "function_encapsulation": FunctionEncapsulationStrategy(),  # 新增策略
        }
        self.type_model = type_model

    def get_strategy(self, function_name: str) -> ActionStrategy:
        # 遍历所有类型模型的属性，查找匹配的函数名称
        for type_attribute, strategy_key in [
            (self.type_model.type_simple_judgment, "simple_judgment"),
            (self.type_model.type_logic_operation, "logic_operation"),
            (self.type_model.type_static_assignment, "static_assignment"),
            (self.type_model.type_parameter_assignment, "parameter_assignment"),
        ]:
            if function_name in type_attribute:
                return self.strategies[strategy_key]

        # 如果没有找到匹配的策略，抛出异常
        raise ValueError(f"No strategy found for function: {function_name}")

    @staticmethod
    def get_dynamic_strategy(
            action_item,
            addresses,
            configuration_params) -> tuple[ActionStrategy, Any]:
        if not list(action_item.params.keys())[0] == action_item.value:
            return StaticAssignmentStrategy(), action_item.value
        if list(action_item.params.keys())[0] == action_item.value:
            return ParameterAssignmentStrategy(), action_item.value
        if any(addresses[i] == addresses[i + 1] for i in range(len(addresses) - 1)):
            return SimpleJudgmentStrategy(), action_item.value
        if configuration_params and any(re.search(r'[\+\-\*/]', v) for v in configuration_params.values()):
            return LogicOperationStrategy(), action_item.value
        # 如果没有符合的条件，可以返回一个默认策略或抛出异常
        raise ValueError("No suitable strategy found.")


class CodeGenerator:
    def __init__(self, configurations, type_model, processor):
        self.configurations = configurations
        self.type_model = type_model
        self.strategy_factory = ActionStrategyFactory(type_model)
        self.processor = processor

    def generate_pseudocode(self, function_name, return_type="void", class_name=""):
        configuration = self.configurations.get(function_name, None)
        if configuration is None:
            return f"// No configuration found for function: {function_name}"

        params = self._find_params(configuration)
        function_signature = self._generate_function_signature(function_name, return_type, class_name, params)
        pseudocode_lines = [f"{function_signature}\n{{"]
        sub_function_definitions = '\n'
        if configuration.sub_function:
            test = DataClassifier(configuration.sub_function, configuration)
            result = test.process_data()
            sub_function_calls, sub_function_definitions = self._handle_sub_functions(configuration, return_type,
                                                                                      class_name)
            pseudocode_lines += sub_function_calls
        else:
            pseudocode_lines += self._execute_actions(configuration.actions, params, function_name)

        pseudocode_lines.append("}\n")
        pseudocode_lines += sub_function_definitions
        return "\n".join(pseudocode_lines)

    @staticmethod
    def _find_params(configuration):
        return find_common_elements_to_params(configuration.params, configuration.values, configuration.address)

    def _handle_sub_functions(self, configuration, return_type, class_name):
        sub_function_calls, sub_function_definitions = [], []
        for sub_func_name, action_items in configuration.sub_function.items():
            sub_params = self._find_params_from_actions(action_items, configuration)
            call_line, definition = self._generate_sub_function(sub_func_name, return_type, action_items, sub_params,
                                                                class_name, configuration)
            sub_function_calls.append(call_line)
            sub_function_definitions.append(definition)
        return sub_function_calls, sub_function_definitions

    @staticmethod
    def _find_params_from_actions(action_items, configuration):
        values, address = zip(*[(action.value, action.address) for action in action_items])
        return find_common_elements_to_params(configuration.params, values, address)

    def _generate_sub_function(self, function_name, return_type, action_items, params, class_name, configuration):
        function_signature = self._generate_function_signature(function_name, return_type, class_name, params)
        call_line = f"\t{function_signature};"
        values, address = zip(*[(action.value, action.address) for action in action_items])
        strategy, _ = self.strategy_factory.get_dynamic_strategy(action_items[0], address, configuration)
        function_lines = [f"{function_signature}\n{{"]
        for action_item in action_items:
            action_lines = strategy.execute_action(action_item, params)
            function_lines += action_lines
        function_lines.append("}\n")
        return call_line, "\n".join(function_lines)

    def _execute_actions(self, actions, params, function_name):
        lines = []
        for index, action in enumerate(actions):
            strategy = self.strategy_factory.get_strategy(function_name)
            action_lines = strategy.execute_action(action, params, index > 0)
            lines.extend(action_lines)
        return lines

    @staticmethod
    def _generate_function_signature(function_name, return_type, class_name, params):
        param_str = ", ".join([f"{ptype} {pname}" for pname, ptype in params.items()]) if params else ""
        if class_name:
            return f"{return_type} {class_name}::{function_name}({param_str})"
        return f"{function_name}({','.join(params.keys())})"


if __name__ == '__main__':
    # 使用示例
    parser = PseudocodeParser()
    functions_dict = parser.parse_csv_to_pseudocode("Assign_Function.csv")
    processor = DataProcessor(functions_dict)
    processed_configurations = processor.process()
    classifier = DataClassifier(processed_configurations)
    type_model = classifier.process_data()
    print(type_model)

    code_generator = CodeGenerator(processed_configurations, type_model, processor)
    # pseudocode_static = code_generator.generate_pseudocode('set_radio_init', "void", "DRIVER_KUNLUN")
    # print(pseudocode_static)
    #
    # pseudocode_parameter = code_generator.generate_pseudocode('set_rx_tpana', "void", "DRIVER_KUNLUN")
    #
    # print(pseudocode_parameter)
    # #
    # # 生成简单判断类型的伪代码
    # pseudocode_simple_judgment = code_generator.generate_pseudocode('tx0_ldo_on', "void", "DRIVER_KUNLUN")
    # #
    # # 调用generate_pseudocode生成代码
    # function_name = 'set_rx0_tia_vcm'
    # return_type = 'void'
    # class_name = 'DRIVER_KUNLUN'
    #
    # # 生成伪代码
    # pseudocode = code_generator.generate_pseudocode(function_name, return_type, class_name)
    # print(pseudocode)

    # 调用generate_pseudocode生成代码
    function_name = 'rx0_on'
    return_type = 'void'
    class_name = 'DRIVER_KUNLUN'

    # 生成伪代码
    pseudocode = code_generator.generate_pseudocode(function_name, return_type, class_name)
    print(pseudocode)
    # processed_configurations = processor.process()
    # classifier = DataClassifier(functions_dict)
    # instruction = processor.fetch_deep_attribute_values('set_radio_init', 'actions', 'instruction')
    # address = processor.fetch_deep_attribute_values('set_radio_init', 'address')
    # values = processor.fetch_deep_attribute_values('set_radio_init', 'values')
    # comments = processor.fetch_deep_attribute_values('set_radio_init', 'actions', 'comment')
    # print(classifier.process_data())
    #
    # print(instruction)
    # print(address)
    # print(values)
    # print(comments)

    # for function_name, function in processed_configurations.items():
    #     for action in function.actions:
    #         if action.params:
    #             for param in function.params:
    #                 print(action.params.get(param, None))

    # classifier = DataClassifier(functions_dict)
#     print(functions_dict)
#     # model_result = classifier.process_data()
#     # print(model_result)
#     for function_name, function in functions_dict.items():
#         print(function_name)
#         print(function)
