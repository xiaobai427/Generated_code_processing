import csv
import json

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
import re


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
                    pass
                else:
                    del func['sub_function']

            return self.functions


class TypeModel(BaseModel):
    type_static_assignment: List[str] = Field(default_factory=list)
    type_simple_judgment: List[str] = Field(default_factory=list)
    type_parameter_assignment: List[str] = Field(default_factory=list)
    type_logic_operation: List[str] = Field(default_factory=list)
    type_function_encapsulation: List[str] = Field(default_factory=list)


class DataClassifier:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.judgment_to_type = [
            (self.is_function_encapsulation, 'type_function_encapsulation'),
            (self.is_static_assignment, 'type_static_assignment'),
            (self.is_simple_judgment, 'type_simple_judgment'),
            (self.is_parameter_assignment, 'type_parameter_assignment'),
            (self.is_logic_operation, 'type_logic_operation'),
        ]

    @staticmethod
    def is_static_assignment(value: Dict[str, Any]) -> bool:
        return not value['params']

    @staticmethod
    def has_consecutive_duplicates(addresses: List[str]) -> bool:
        return any(addresses[i] == addresses[i + 1] for i in range(len(addresses) - 1))

    @classmethod
    def is_simple_judgment(cls, value: Dict[str, Any]) -> bool:
        return cls.has_consecutive_duplicates(value['address'])

    @staticmethod
    def is_parameter_assignment(value: Dict[str, Any]) -> bool:
        return value['params'] and all(not re.search(r'[\+\-\*/]', v) for v in value['values'])

    @staticmethod
    def is_logic_operation(value: Dict[str, Any]) -> bool:
        return value['params'] and any(re.search(r'[\+\-\*/]', v) for v in value['values'])

    @staticmethod
    def is_function_encapsulation(value: Dict[str, Any]) -> bool:
        return 'sub_function' in value

    def process_data(self) -> TypeModel:
        model = TypeModel()
        for key, value in self.data.items():
            for judgment, type_attr in self.judgment_to_type:
                if judgment(value):
                    getattr(model, type_attr).append(key)
                    break  # 如果需要一个项目只能归类到一个类型，添加break; 否则，删除break以允许多重分类
        return model


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

    def generate_pseudocode(self, function_name: str) -> str:
        configuration = self.processed_configurations.get(function_name)
        if not configuration:
            return f"// No configuration found for function: {function_name}"

        pseudocode_lines = [f"void DRIVER_KUNLUN::{function_name}()\n{{"]

        for action in configuration.actions:
            if action.instruction == 'write':
                pseudocode_lines.append(f"\treg_write({action.address}, {action.value}); // {action.comment}")
            elif action.instruction == 'delay':
                pseudocode_lines.append(f"\tSleep({action.address} {action.value}); // {action.comment}")

        pseudocode_lines.append("}\n")

        return "\n".join(pseudocode_lines)


class Type1Model(BaseModel):
    instruction: List[str] = Field(default_factory=list)
    address: List[str] = Field(default_factory=list)
    values: List[str] = Field(default_factory=list)
    comments: List[str] = Field(default_factory=list)


if __name__ == '__main__':
    # 使用示例
    parser = PseudocodeParser()
    functions_dict = parser.parse_csv_to_pseudocode("Assign_Function.csv")
    processor = DataProcessor(functions_dict)
    processed_configurations = processor.process()
    pseudocode = processor.generate_pseudocode('set_radio_init')
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
