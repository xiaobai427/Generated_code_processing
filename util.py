import re
from collections import Counter
from typing import Union, Type, List, Any

from models.base import ConfigurationModel


def adjust_params(data):
    """
    Adjusts the 'params' field in a nested dictionary structure based on whether the 'value'
    is a key in 'params'. If 'value' is not a key, sets 'params' to "{}".

    :param data: The nested dictionary structure containing the instructions and their parameters.
    :return: The modified dictionary with adjusted 'params'.
    """
    for key, instructions in data.items():
        for instruction in instructions:
            # Convert string representation of dict to dict
            if "params" in instruction.keys():
                params = eval(instruction['params'])
                # Check if the 'value' is a key in 'params'
                instruction['params'] = str(params)
                if instruction['value'] not in params:
                    instruction['params'] = "{}"
                else:
                    # Ensure params is properly formatted as a string
                    instruction['params'] = str(params)
    return data


def contains_arithmetic_operator(value: str) -> bool:
    """检查字符串是否包含算术运算符(+, -, *, /)。

    参数:
    value: 需要检查的字符串。

    返回:
    bool: 如果字符串包含任何算术运算符，则为True，否则为False。
    """
    operators = ['+', '-', '*', '/']
    return any(op in value for op in operators)


def find_common_elements_to_params(params, values, addresses, element_type='uint8_t'):
    combined_expressions = ' '.join(values)

    # 使用正则表达式提取变量名
    variables_in_expressions = set(re.findall(r'\b[a-zA-Z_][a-zA-Z_0-9]*\b', combined_expressions))

    # 根据params的顺序找出共有元素
    common_elements = [param for param in params if param in variables_in_expressions]

    # 为每个共有元素指定类型
    common_elements_with_type = {element: element_type for element in common_elements}

    if not common_elements_with_type:
        if not common_elements and any(addresses[i] == addresses[i + 1] for i in range(len(addresses) - 1)):
            return {element: element_type for element in params}

    return common_elements_with_type


def get_most_common_type(items: Union[ConfigurationModel, List[Any]]) -> Type:
    """
    确定传入参数的类型。
    如果是直接传递类型本身，则直接返回该类型。
    如果是传递类型实例的列表，则确定列表中最常见的元素类型。
    """
    # 直接传递类型本身
    if isinstance(items, ConfigurationModel):
        return type(items)
    # 传递类型实例的列表
    elif isinstance(items, list) and items:
        types = [type(item) for item in items]
        most_common_type, _ = Counter(types).most_common(1)[0]
        return most_common_type
