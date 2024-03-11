import re
from pathlib import Path

from code_generator.assign import CodeGeneratorAssign
from code_generator.trim import CodeGeneratorTrim


class CodeGeneratorController:
    def __init__(self):
        # This list will hold tuples of (GeneratorClass, csv_path) to process
        self.generators_info = []
        self.generated_code = ""  # To accumulate all generated codes
        self.generated_code_dict = {}
        self.variables_dict = {}

    def add_generator(self, generator_class, csv_path):
        """Add a generator to the processing list."""
        self.generators_info.append((generator_class, csv_path))

    def run_all(self):
        """Run each added generator and collect their generated code."""
        for generator_class, csv_path in self.generators_info:
            generator_instance = generator_class(csv_path)
            generator_instance.run()
            # Assuming each generator instance has a `generated_code` attribute
            self.generated_code_dict.update({str(Path(csv_path).stem): generator_instance.generated_code})
            self.generated_code += generator_instance.generated_code + "\n\n"
            if generator_instance.variable:
                self.variables_dict = generator_instance.variable


class RadioGenerator:
    def __init__(self, template_h_path, output_h_path, template_cpp_path, output_cpp_path, functions_dict):
        self.template_h_path = template_h_path
        self.output_h_path = output_h_path
        self.template_cpp_path = template_cpp_path
        self.output_cpp_path = output_cpp_path
        self.functions_dict = functions_dict

    @staticmethod
    def extract_functions(cpp_content):
        function_regex = re.compile(r'(\w+)\s+(\w+)::([a-zA-Z0-9_]+)\((.*?)\)', re.S)
        functions = function_regex.findall(cpp_content)
        return functions

    def process_header(self):
        updated_sections_h = {}
        for key, value in self.functions_dict.items():
            functions = self.extract_functions(value)
            new_function_declarations = []
            for return_type, class_name, function_name, parameters in functions:
                if "set_GL_OTP_ALL" == function_name:
                    continue
                elif parameters == '':
                    new_function_declarations.append(f"\t{return_type} {function_name}({return_type});\n")
                else:
                    new_function_declarations.append(f"\t{return_type} {function_name}({parameters});\n")
            updated_sections_h[key] = new_function_declarations
        self.update_file(self.template_h_path, self.output_h_path, updated_sections_h, is_cpp=False)

    def process_cpp(self):
        updated_sections_cpp = {}
        for key, value in self.functions_dict.items():
            updated_sections_cpp[key] = value.split('\n\n')
        self.update_file(self.template_cpp_path, self.output_cpp_path, updated_sections_cpp, is_cpp=True)

    @staticmethod
    def update_file(template_path, output_path, updated_sections, is_cpp):
        with open(template_path, 'r', encoding='utf-8') as file:
            template_content = file.readlines()

        output_content = template_content.copy()

        for key, declarations in updated_sections.items():
            insert_index = -1
            for i, line in enumerate(output_content):
                if key in line:
                    insert_index = i + 1
                    break

            if insert_index != -1:
                if is_cpp:
                    for decl in declarations:
                        output_content.insert(insert_index, decl + "\n\n")
                        insert_index += 1
                else:
                    for decl in reversed(declarations):
                        output_content.insert(insert_index, decl)

        with open(output_path, 'w', encoding='utf-8') as file:
            file.writelines(output_content)

    def radio_header_process(self):
        self.process_header()
        self.process_cpp()


class VariableHeaderGenerator:
    def __init__(self, template_hpp_path, output_hpp_path, template_cpp_path, output_cpp_path, variable_dict):
        self.template_hpp_path = template_hpp_path
        self.output_hpp_path = output_hpp_path
        self.template_cpp_path = template_cpp_path
        self.output_cpp_path = output_cpp_path
        self.variable_dict = variable_dict

    def process_hpp(self):
        self.update_file(self.template_hpp_path, self.output_hpp_path, self.variable_dict['flag_index_hpp'],
                         'flag_index_hpp')
        self.update_file(self.output_hpp_path, self.output_hpp_path, self.variable_dict['value_ate_hpp'],
                         'value_ate_hpp')

    def process_cpp(self):
        self.update_file(self.template_cpp_path, self.output_cpp_path, self.variable_dict['flag_index_cpp'],
                         'flag_index_cpp')
        self.update_file(self.output_cpp_path, self.output_cpp_path, self.variable_dict['value_ate_cpp'],
                         'value_ate_cpp')

    @staticmethod
    def update_file(template_path, output_path, content, key):
        with open(template_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        insert_index = -1
        for i, line in enumerate(lines):
            if key in line:
                insert_index = i + 1
                break

        if insert_index != -1:
            lines = lines[:insert_index] + [content] + lines[insert_index:]

        with open(output_path, 'w', encoding='utf-8') as file:
            file.writelines(lines)


if __name__ == '__main__':
    # Create the controller instance with the .cpp file path
    code_gen_controller = CodeGeneratorController()

    code_gen_controller.add_generator(CodeGeneratorAssign,
                                      r"C:\Users\Administrator\Desktop\Generated_code_processing\Assign_Function.csv")
    # Add the code generator instances to be executed
    code_gen_controller.add_generator(CodeGeneratorTrim,
                                      r"Trim_Function.csv")

    #     # Run all added generators and write their output to the .cpp file
    code_gen_controller.run_all()

    functions_dict_ = code_gen_controller.generated_code_dict
    template_h_path_ = 'radio_DRIVER_KUNLUNm0_ATE_code.h.template'
    output_h_path_ = 'radio_DRIVER_KUNLUNm0_ATE_code.h'
    template_cpp_path_ = 'radio_DRIVER_KUNLUNm0_ATE_code.cpp.template'
    output_cpp_path_ = 'radio_DRIVER_KUNLUNm0_ATE_code.cpp'
    generator = RadioGenerator(template_h_path_, output_h_path_, template_cpp_path_, output_cpp_path_, functions_dict_)
    generator.radio_header_process()

    template_hpp_path = 'radio_variable.hpp.template'
    output_hpp_path = 'radio_variable.hpp'
    template_cpp_path = 'radio_variable.cpp.template'
    output_cpp_path = 'radio_variable.cpp'
    variable_dict = code_gen_controller.variables_dict
    generator = VariableHeaderGenerator(template_hpp_path, output_hpp_path, template_cpp_path, output_cpp_path,
                                        variable_dict)
    generator.process_hpp()
    generator.process_cpp()
