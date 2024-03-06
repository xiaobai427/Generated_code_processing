from abc import ABC, abstractmethod
from strategy.Interface import SubFunctionHandler
from util import find_common_elements_to_params


class CodeGeneratorBase(ABC):
    def __init__(self, csv_path, function_type="void", driver="DRIVER_KUNLUN"):
        self.csv_path = csv_path
        self.function_type = function_type
        self.driver = driver
        self.strategy_factory_sub = None
        self.strategy_factory_action = None
        self.generated_code = ''

    def run(self):
        functions_dict = self.parse_csv()
        processed_configurations = self.process_data(functions_dict)
        action_type_model, sub_type_model = self.classify_data(processed_configurations)
        self.setup_strategy_factory(action_type_model, sub_type_model)
        self.before_generate_code(processed_configurations)  # 钩子方法
        self.generate_code(processed_configurations)
        self.after_generate_code(processed_configurations)  # 钩子方法

    @abstractmethod
    def parse_csv(self):
        pass

    @abstractmethod
    def process_data(self, functions_dict):
        pass

    @abstractmethod
    def classify_data(self, processed_configurations):
        pass

    @abstractmethod
    def setup_strategy_factory(self, action_type_model, sub_type_model):
        pass

    @abstractmethod
    def generate_code(self, processed_configurations):
        pass

    # 钩子方法，子类可以根据需要覆盖它们
    def before_generate_code(self, processed_configurations):
        # 默认实现为空
        pass

    def after_generate_code(self, processed_configurations):
        # 默认实现为空
        pass


class CodeGenerator:
    def __init__(self, configurations, strategy_factory_action, strategy_factory_sub):
        self.configurations = configurations
        self.strategy_factory_action = strategy_factory_action
        self.strategy_factory_sub = strategy_factory_sub

    def generate_pseudocode(self, function_name, return_type="void", class_name=""):
        configuration = self.configurations.get(function_name, None)
        if configuration is None:
            return f"// No configuration found for function: {function_name}"
        params = self._find_params(configuration)

        function_signature = self._generate_function_signature(function_name, return_type, class_name, params)
        pseudocode_lines = [f"{function_signature}\n{{"]
        sub_function_definitions = '\n'
        if configuration.sub_function:
            sub_function_handler = SubFunctionHandler(self.strategy_factory_sub)
            sub_function_calls, sub_function_definitions = sub_function_handler.handle_sub_functions(configuration,
                                                                                                     return_type,
                                                                                                     class_name)
            pseudocode_lines += sub_function_calls
        else:
            pseudocode_lines += self._execute_actions(configuration.actions, params, function_name)

        pseudocode_lines.append("}\n")
        for actions in configuration.actions:
            lines = []
            if actions.flag:
                for pseudocode_line in pseudocode_lines:
                    pseudocode_line_str = pseudocode_line.replace("uint8_t", "string")
                    lines.append(pseudocode_line_str)
                pseudocode_lines += lines
        pseudocode_lines += sub_function_definitions
        return "\n".join(pseudocode_lines)

    @staticmethod
    def _find_params(configuration):
        return find_common_elements_to_params(configuration.params, configuration.values, configuration.address)

    def _execute_actions(self, actions, params, function_name):
        lines = []
        for index, action in enumerate(actions):
            strategy = self.strategy_factory_action.get_strategy(function_name)
            action_lines = strategy.execute_action(action, params, index > 0)
            lines.extend(action_lines)
        return lines

    @staticmethod
    def _generate_function_signature(function_name, return_type, class_name, params):
        param_str = ", ".join([f"{ptype} {pname}" for pname, ptype in params.items()]) if params else ""
        if class_name:
            return f"{return_type} {class_name}::{function_name}({param_str})"
        return f"{function_name}({','.join(params.keys())})"
