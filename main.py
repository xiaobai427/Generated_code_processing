from classification.data_classifier import DataClassifier
from code_generator.code_generator_basics import CodeGeneratorBasics
from handle.data_processor import DataProcessor
from parser.pseudocode_parser import PseudocodeParser
from strategy.factory import ActionStrategyFactory

if __name__ == '__main__':
    # 使用示例
    parser = PseudocodeParser()
    functions_dict = parser.parse_csv_to_pseudocode("Assign_Function.csv")
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

    # pseudocode_static = code_generator.generate_pseudocode('set_radio_cal_cbc', "void", "DRIVER_KUNLUN")
    # print(pseudocode_static)
    # print(processed_configurations.keys())
    # pseudocode_static = code_generator.generate_pseudocode('set_radio_init', "void", "DRIVER_KUNLUN")
    # print(pseudocode_static)


    #
    # pseudocode_parameter = code_generator.generate_pseudocode('set_rx_tpana', "void", "DRIVER_KUNLUN")
    #
    # print(pseudocode_parameter)
    # #
    # # 生成简单判断类型的伪代码
    # pseudocode_simple_judgment = code_generator.generate_pseudocode('tx0_ldo_on', "void", "DRIVER_KUNLUN")
    # print(pseudocode_simple_judgment)
    # #
    # # 调用generate_pseudocode生成代码
    # function_name = 'set_rx0_tia_vcm'
    # return_type = 'void'
    # class_name = 'DRIVER_KUNLUN'
    #
    # # 生成伪代码
    # pseudocode = code_generator.generate_pseudocode(function_name, return_type, class_name)
    # print(pseudocode)
    #
    # 调用generate_pseudocode生成代码
    # function_name = 'rx0_on'
    # return_type = 'void'
    # class_name = 'DRIVER_KUNLUN'
    # #
    # # 生成伪代码
    # pseudocode = code_generator.generate_pseudocode(function_name, return_type, class_name)
    # print(pseudocode)

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