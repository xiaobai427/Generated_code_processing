from classification.data_classifier import DataClassifier
from strategy.Interface import SubFunctionHandler
from strategy.factory import ActionStrategyFactory
from util import find_common_elements_to_params


class CodeGeneratorBasics:
    def __init__(self, configurations, processor, strategy_factory_action, strategy_factory_sub):
        self.configurations = configurations
        self.strategy_factory_action = strategy_factory_action
        self.strategy_factory_sub = strategy_factory_sub
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
            sub_function_handler = SubFunctionHandler(self.strategy_factory_sub)
            sub_function_calls, sub_function_definitions = sub_function_handler.handle_sub_functions(configuration,
                                                                                                     return_type,
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


if __name__ == '__main__':
    from classification.data_classifier import DataClassifier
    from handle.data_processor import DataProcessor
    from parser.pseudocode_parser import PseudocodeParser
    from strategy.factory import ActionStrategyFactory

    if __name__ == '__main__':
        # 使用示例
        parser = PseudocodeParser()
        functions_dict = parser.parse_csv_to_pseudocode(r"C:\Users\shibo.huang\Desktop\Generated_code_processing\Assign_Function.csv")
        processor = DataProcessor(functions_dict)
        processed_configurations = processor.process()

        action_type_model = None
        sub_type_model = None
        for function_name, function_configurations in processed_configurations.items():
            if function_configurations.sub_function:
                sub_classifier = DataClassifier(function_configurations.sub_function, function_configurations)
                sub_type_model = sub_classifier.process_data()
        classifier = DataClassifier(processed_configurations)
        action_type_model = classifier.process_data()
        strategy_factory_sub = ActionStrategyFactory(sub_type_model)
        strategy_factory_action = ActionStrategyFactory(action_type_model)
        code_generator = CodeGeneratorBasics(
            processed_configurations,
            processor,
            strategy_factory_action=strategy_factory_action,
            strategy_factory_sub=strategy_factory_sub)

        for function_name, function_configurations in processed_configurations.items():
            pseudocode_static = code_generator.generate_pseudocode(function_name, "void", "DRIVER_KUNLUN")
            print(pseudocode_static)
