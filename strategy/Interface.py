from typing import Optional, Dict, List

from models.base import ActionItemModel
from strategy.base import ActionStrategy
from util import find_common_elements_to_params


class StaticAssignmentStrategy(ActionStrategy):
    def execute_action(self,
                       action: ActionItemModel,
                       params: Optional[Dict[str, str]] = None,
                       state: bool = None) -> List[str]:
        lines = []
        if action.instruction == "write":
            lines.append(f"\treg_write({action.address}, {action.value}); // {action.comment}")
        elif action.instruction == "delay":
            lines.append(f"\tSleep({action.address} {action.value}); // {action.comment}")
        return lines


class SimpleJudgmentStrategy(ActionStrategy):
    def execute_action(self, action: ActionItemModel, params: Optional[Dict[str, str]] = None,
                       state: bool = None) -> List[str]:

        lines = []
        condition = f"if (enable == {action.params['enable']})"
        if action.params['enable'] and action.instruction == "write":
            if state:  # 使用 else if 替换之后的 if 条件
                condition = "else " + condition

            action_line = f"\treg_write({action.address}, {action.value}); // {action.comment}"
            lines.append(f"\t{condition}")
            lines.append(f"\t{{")
            lines.append(f"\t{action_line}")
            lines.append(f"\t}}")
        elif action.instruction == "delay":
            lines.append(f"\tSleep({action.address} {action.value}); // {action.comment}")
        return lines


# 参数赋值策略
class ParameterAssignmentStrategy(ActionStrategy):
    def execute_action(self,
                       action: ActionItemModel,
                       params: Optional[Dict[str, str]] = None,
                       state: bool = None) -> List[str]:
        param_name = action.value.strip("{}")
        return [f"\treg_write({action.address}, {param_name}); // {action.comment}"]


class LogicOperationStrategy(ActionStrategy):
    def execute_action(self,
                       action: ActionItemModel,
                       params: Optional[Dict[str, str]] = None,
                       state: bool = None) -> List[str]:
        lines = []
        if action.instruction == 'write':
            # 处理包含逻辑运算的写入指令
            evaluated_value = self._evaluate_logic_expression(action.value, params)
            line = f"\treg_write({action.address}, {evaluated_value}); // {action.comment}"
            lines.append(line)
        elif action.instruction == 'delay':
            # 处理延时
            lines.append(f"\tSleep({action.address}, {action.value}); // {action.comment}")
        return lines

    @staticmethod
    def _evaluate_logic_expression(expression: str, params: Dict[str, str]) -> str:
        # 这个方法用于将表达式中的参数替换为实际的参数值，并返回处理后的表达式
        for param_name, param_type in params.items():
            if param_name in expression:
                # 这里简单地替换参数名为其类型表示，需要根据实际情况进行调整
                expression = expression.replace(param_name, param_type)
        return expression


class SubFunctionHandler:
    def __init__(self, strategy_factory_sub):
        self.strategy_factory_sub = strategy_factory_sub

    @staticmethod
    def find_params_from_actions(action_items, configuration):
        values, address = zip(*[(action.value, action.address) for action in action_items])
        return find_common_elements_to_params(configuration.params, values, address)

    def handle_sub_functions(self, configuration, return_type, class_name):
        sub_function_calls, sub_function_definitions = [], []
        for sub_func_name, action_items in configuration.sub_function.items():
            sub_params = self.find_params_from_actions(action_items, configuration)
            call_line, definition = self.generate_sub_function(sub_func_name, return_type, action_items, sub_params,
                                                               class_name)
            sub_function_calls.append(call_line)
            sub_function_definitions.append(definition)
        return sub_function_calls, sub_function_definitions

    def generate_sub_function(self, function_name, return_type, action_items, params, class_name):
        _function_signature = self.generate_function_signature(function_name, return_type, None, params)
        function_signature = self.generate_function_signature(function_name, return_type, class_name, params)
        call_line = f"\t{_function_signature};"
        strategy = self.strategy_factory_sub.get_strategy(function_name)
        function_lines = [f"{function_signature}\n{{"]
        for action_item in action_items:
            action_lines = strategy.execute_action(action_item, params)
            function_lines += action_lines
        function_lines.append("}\n")
        return call_line, "\n".join(function_lines)

    @staticmethod
    def generate_function_signature(function_name, return_type, class_name, params):
        param_str = ", ".join([f"{ptype} {pname}" for pname, ptype in params.items()]) if params else ""
        if class_name:
            return f"{return_type} {class_name}::{function_name}({param_str})"
        return f"{function_name}({','.join(params.keys())})"
