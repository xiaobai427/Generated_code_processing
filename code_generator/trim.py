import re

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
                        item.params_value = f"\"{str(item.value)}_ATE\""
                        item.params = {function_configurations.params[0]: item.value}
                        item.value = function_configurations.params[0]
                continue
            for item in function_configurations.actions:
                item.params = {function_configurations.params[0]: item.value}
                item.value = function_configurations.params[0]
            function_configurations.values.append(function_configurations.params[0])

        return processed_configurations, processor

    def after_generate_code(self, processed_configurations):
        for function_name, function_configurations in processed_configurations.items():
            if not function_configurations.sub_function:
                self.function_names.append(function_name)
                for item in function_configurations.actions:
                    if item.params[item.value]:
                        self.generated_code += '\n' + self._generate_function_main(self.driver, function_name,
                                                                                   item.params[item.value])
                    if item.trim_idx:
                        self.flags.append(item.flag)
                        self.trim_value_list.append(item.trim_idx)
            if function_configurations.sub_function:
                for _function_name_sub, _function_configurations in function_configurations.sub_function.items():
                    for configurations in _function_configurations:
                        if configurations.trim_idx:
                            self.function_names.append(_function_name_sub)
                            self.flags.append(configurations.flag)
                            self.trim_value_list.append(configurations.trim_idx)
            _values = [item for item in function_configurations.values if item != "trim_value"]
            _values = self.remove_duplicates_keep_order(list(set(_values)))
            self.values += _values
        runTimeVal_functions = self.generate_runTimeVal_function(self.driver, 'set_GL_OTP_ALL', self.values)
        function_pointers = self.generate_function_pointers_array(self.driver, self.function_names)
        self.generated_code += '\n' + runTimeVal_functions + '\n' + function_pointers + '\n'
        self.generate_flag_index_code()
        self.generate_value_ate()

    @staticmethod
    def remove_duplicates_keep_order(seq):
        seen = set()
        return [x for x in seq if not (x in seen or seen.add(x))]

    @staticmethod
    def generate_runTimeVal_function(class_name, function_name, otp_list):
        sorted_otp_list = sorted(otp_list)
        function_body = f"void {class_name}::{function_name}()\n{{\n"
        for otp in sorted_otp_list:
            function_call = f'\trdi.runTimeVal("{otp}_ATE",{otp}_ATE);'
            function_body += function_call + "\n"
        function_body += "}\n"
        return function_body

    @staticmethod
    def generate_function_pointers_array(class_name, function_names):
        array_declaration = f"void ({class_name}::*functionPointers[400])(uint8_t) = \n{{\n"
        array_declaration += f"\t&{class_name}::set_driver_index, // this line is only to take index-0;\n"
        for name in function_names:
            array_declaration += f"\t&{class_name}::{name},\n"
        array_declaration = array_declaration.rstrip(',\n') + "\n\n};\n"
        return array_declaration

    @staticmethod
    def _generate_function_main(class_name, function_name, value):
        function_body = f"void {class_name}::{function_name}()\n{{\n"
        function_call = f'\t{function_name}("{value}_ATE");'
        function_body += function_call + "\n"
        function_body += "}\n"
        return function_body

    def generate_flag_index_code(self):
        # 验证长度是否一致
        if len(self.flags) != len(self.trim_value_list):
            raise ValueError("The length of flags and trim_value_list does not match.")

        # 初始化代码字符串
        cpp_code = ""
        hpp_code = ""

        for index, flag in enumerate(self.flags):
            # 使用正则表达式提取数字部分，如果需要
            match = re.search(r'\d+', flag)
            number = match.group() if match else str(index + 1)

            # 生成cpp和hpp代码
            cpp_code += f"const int {flag}_INDEX={number};\n"
            hpp_code += f"extern const int {flag}_INDEX;\n"

        self.variable["flag_index_cpp"] = cpp_code
        self.variable["flag_index_hpp"] = hpp_code

    def generate_value_ate(self):
        cpp_code = ""
        hpp_code = ""

        # Iterate over each value in the provided list
        for value in self.values:
            # Append "_ATE" to the value for the variable name
            variable_name = f"{value}_ATE"

            # Generate the .cpp code line for the current value
            cpp_code_line = f"ARRAY_D {variable_name}(TOTAL_SITE_NUM);\n"
            cpp_code += cpp_code_line

            # Generate the .hpp code line for the current value
            hpp_code_line = f"extern ARRAY_D {variable_name};\n"
            hpp_code += hpp_code_line

        self.variable["value_ate_cpp"] = cpp_code
        self.variable["value_ate_hpp"] = hpp_code