import csv
import re

from util import adjust_params


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