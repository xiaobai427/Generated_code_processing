from typing import List, Optional, Dict

from pydantic import BaseModel, Field


class TypeModel(BaseModel):
    type_static_assignment: List[str] = Field(default_factory=list)
    type_simple_judgment: List[str] = Field(default_factory=list)
    type_parameter_assignment: List[str] = Field(default_factory=list)
    type_logic_operation: List[str] = Field(default_factory=list)
    type_function_encapsulation: List[str] = Field(default_factory=list)


# Action item model for actions and sub-functions
class ActionItemModel(BaseModel):
    step: str
    instruction: str
    address: str
    value: str
    encapsulation: Optional[str] = None
    comment: Optional[str] = None
    params: Optional[Dict[str, str]] = None

    def __str__(self):
        return (f"Step: {self.step}, Instruction: {self.instruction}, Address: {self.address}, "
                f"Value: {self.value}, Encapsulation: {self.encapsulation}, Comment: {self.comment}, "
                f"Params: {self.params}")


class TrimActionItemModel(ActionItemModel):
    flag: Optional[str] = None
    index_start: Optional[str] = None
    index_stop: Optional[str] = None
    index_step: Optional[str] = None
    spec_low: Optional[str] = None
    target: Optional[str] = None
    spec_high: Optional[str] = None
    unit_scale: Optional[str] = None
    in_target: Optional[str] = None
    out_target: Optional[str] = None


class ConfigurationModel(BaseModel):
    params: List[str] = Field(default_factory=list)
    actions: List[ActionItemModel] = Field(default_factory=list)
    sub_function: Dict[str, List[ActionItemModel]] = Field(default_factory=dict)
    values: List[str] = Field(default_factory=list)
    address: List[str] = Field(default_factory=list)

    def __str__(self):
        params_str = ", ".join(self.params)
        actions_str = "; ".join(str(action) for action in self.actions)
        sub_functions_str = "; ".join(
            f"{k}: [{', '.join(str(v) for v in vals)}]" for k, vals in self.sub_function.items())
        values_str = ", ".join(self.values)
        address_str = ", ".join(self.address)

        return (f"Params: [{params_str}], Actions: [{actions_str}], "
                f"SubFunctions: [{sub_functions_str}], Values: [{values_str}], "
                f"Address: [{address_str}]")
