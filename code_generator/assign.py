from classification.data_classifier import DataClassifier
from code_generator.base import CodeGeneratorBase, CodeGenerator
from handle.data_processor import DataProcessor
from parser.pseudocode_parser import PseudocodeParser
from strategy.factory import ActionStrategyFactory


class CodeGeneratorAssign(CodeGeneratorBase):
    def parse_csv(self):
        parser = PseudocodeParser()
        return parser.parse_csv_to_pseudocode(self.csv_path)

    def process_data(self, functions_dict):
        processor = DataProcessor(functions_dict)
        processed_configurations = processor.process()
        # print(processor.fetch_deep_attribute_values("set_radio_init", "actions", "instruction"))
        # print(processor.extract_parameters_from_actions_or_sub_functions("set_radio_init", "actions"))
        return processed_configurations, processor

    def classify_data(self, processed_configurations):
        action_type_model = None
        sub_type_model = None
        for function_name, function_configurations in processed_configurations.items():
            if function_configurations.sub_function:
                sub_classifier = DataClassifier(function_configurations.sub_function, function_configurations)
                sub_type_model = sub_classifier.process_data()
            classifier = DataClassifier(processed_configurations)
            action_type_model = classifier.process_data()
        return action_type_model, sub_type_model

    def setup_strategy_factory(self, action_type_model, sub_type_model):
        self.strategy_factory_sub = ActionStrategyFactory(sub_type_model)
        self.strategy_factory_action = ActionStrategyFactory(action_type_model)

    def generate_code(self, processed_configurations):
        code_generator = CodeGenerator(
            processed_configurations,
            self.strategy_factory_action,
            self.strategy_factory_sub)
        for function_name, function_configurations in processed_configurations.items():
            pseudocode_static = code_generator.generate_pseudocode(function_name, self.function_type, self.driver)
            self.generated_code += pseudocode_static
