from models.base import TypeModel
from strategy.Interface import SimpleJudgmentStrategy, LogicOperationStrategy, StaticAssignmentStrategy, \
    ParameterAssignmentStrategy
from strategy.base import ActionStrategy


class ActionStrategyFactory:
    def __init__(self, type_model: TypeModel):
        self.strategies = {
            "simple_judgment": SimpleJudgmentStrategy(),
            "logic_operation": LogicOperationStrategy(),
            "static_assignment": StaticAssignmentStrategy(),
            "parameter_assignment": ParameterAssignmentStrategy(),
        }
        self.type_model = type_model

    def get_strategy(self, function_name: str) -> ActionStrategy:
        # 遍历所有类型模型的属性，查找匹配的函数名称
        for type_attribute, strategy_key in [
            (self.type_model.type_simple_judgment, "simple_judgment"),
            (self.type_model.type_logic_operation, "logic_operation"),
            (self.type_model.type_static_assignment, "static_assignment"),
            (self.type_model.type_parameter_assignment, "parameter_assignment"),
        ]:
            if function_name in type_attribute:
                return self.strategies[strategy_key]

        # 如果没有找到匹配的策略，抛出异常
        raise ValueError(f"No strategy found for function: {function_name}")