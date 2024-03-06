from code_generator.assign import CodeGeneratorAssign
from code_generator.trim import CodeGeneratorTrim


class CodeGeneratorController:
    def __init__(self, cpp_path):
        self.cpp_path = cpp_path
        # This list will hold tuples of (GeneratorClass, csv_path) to process
        self.generators_info = []
        self.generated_code = ""  # To accumulate all generated codes

    def add_generator(self, generator_class, csv_path):
        """Add a generator to the processing list."""
        self.generators_info.append((generator_class, csv_path))

    def run_all(self):
        """Run each added generator and collect their generated code."""
        for generator_class, csv_path in self.generators_info:
            generator_instance = generator_class(csv_path)
            generator_instance.run()
            # Assuming each generator instance has a `generated_code` attribute
            self.generated_code += generator_instance.generated_code + "\n\n"

        self._write_to_cpp()

    def _write_to_cpp(self):
        """Write the collected generated code to the specified .cpp file."""
        with open(self.cpp_path, 'w') as cpp_file:
            cpp_file.write(self.generated_code)


if __name__ == '__main__':
    cpp_path = r"C:\Users\shibo.huang\Desktop\Generated_code_processing\output.cpp"

    # Create the controller instance with the .cpp file path
    code_gen_controller = CodeGeneratorController(cpp_path)

    code_gen_controller.add_generator(CodeGeneratorAssign,
                                      r"C:\Users\shibo.huang\Desktop\Generated_code_processing\Assign_Function.csv")
    # Add the code generator instances to be executed
    code_gen_controller.add_generator(CodeGeneratorTrim,
                                      r"C:\Users\shibo.huang\Desktop\Generated_code_processing\Trim_Function.csv")

#     # Run all added generators and write their output to the .cpp file
    code_gen_controller.run_all()
# def generate_set_GL_OTP_ALL_function(class_name, function_name, otp_list):
#     sorted_otp_list = sorted(otp_list)
#     function_body = f"void {class_name}::{function_name}()\n\n{{\n"
#     for otp in sorted_otp_list:
#         function_call = f'\trdi.runTimeVal("{otp}_ATE",{otp}_ATE);'
#         function_body += function_call + "\n"
#     function_body += "}\n"
#     return function_body
#
# # 使用示例
# class_name = "DRIVER_KUNLUN"
# otp_list = [
#     'GL_OTP_LODIST_MLDO08_VRCAL', 'GL_OTP_ADC_LDO08_VOSEL',
#     'GL_OTP_RX_TIA_MX_VCM_L0', 'GL_OTP_FPLL_PFDLDO15_VOSEL',
#     'GL_OTP_RXBG_CTAT_8U_CAL', 'GL_OTP_TXLO_13G_LDO08_VRCAL'
# ]
# print(generate_set_GL_OTP_ALL_function(class_name, 'set_GL_OTP_ALL', otp_list))


# def generate_function_pointers_array(class_name, function_names):
#     array_declaration = f"void ({class_name}::*functionPointers[400])(uint8_t) = {{\n"
#     array_declaration += f"\t&{class_name}::set_driver_index, // this line is only to take index-0;\n"
#     for name in function_names:
#         array_declaration += f"\t&{class_name}::{name},\n"
#     array_declaration = array_declaration.rstrip(',\n') + "\n\n};\n"
#     return array_declaration
#
# # 使用示例
# class_name = "DRIVER_KUNLUN"
# function_names = [
#     'set_radio_cal', 'set_radio_cal_pll', 'set_radio_cal_lo',
#     'set_radio_cal_rx', 'set_radio_cal_adc', 'set_radio_cal_cbc',
#     'set_radio_cal_tx'
# ]
# print(generate_function_pointers_array(class_name, function_names))
