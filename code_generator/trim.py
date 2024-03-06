from code_generator.assign import CodeGeneratorAssign
from handle.data_processor import DataProcessor


class CodeGeneratorTrim(CodeGeneratorAssign):
    def process_data(self, functions_dict):
        processor = DataProcessor(functions_dict)
        processed_configurations = processor.process()

        # Adapted from the data processing part of run()
        for function_name, function_configurations in processed_configurations.items():
            if function_configurations.sub_function:
                for _function_name_sub, _function_configurations in function_configurations.sub_function.items():
                    for item in _function_configurations:
                        item.params_value = item.value
                        item.params = {function_configurations.params[0]: item.value}
                        item.value = function_configurations.params[0]
                continue
            for item in function_configurations.actions:
                item.params = {function_configurations.params[0]: item.value}
                item.value = function_configurations.params[0]
            function_configurations.values.append(function_configurations.params[0])

        return processed_configurations

    def before_generate_code(self, processed_configurations):
        values = []
        function_names = []
        for function_name, function_configurations in processed_configurations.items():
            function_names.append(function_name)
            if function_configurations.sub_function:
                for _function_name_sub, _function_configurations in function_configurations.sub_function.items():
                    function_names.append(_function_name_sub)
            _values = [item for item in function_configurations.values if item != "trim_value"]
            _values = self.remove_duplicates_keep_order(list(set(_values)))
            values += _values
        print(values)
        print(function_names)
        # values = self.remove_duplicates_keep_order(list(set(values)))
        # print(values)

    @staticmethod
    def remove_duplicates_keep_order(seq):
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

