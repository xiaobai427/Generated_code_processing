from typing import List, Optional, Dict, Union

from pydantic import BaseModel, Field


class TypeModel(BaseModel):
    type_static_assignment: List[str] = Field(default_factory=list)
    type_simple_judgment: List[str] = Field(default_factory=list)
    type_parameter_assignment: List[str] = Field(default_factory=list)
    type_logic_operation: List[str] = Field(default_factory=list)
    type_function_encapsulation: List[str] = Field(default_factory=list)


# Action item model for actions and sub-functions
class ActionItemModel(BaseModel):
    step: Union[str, List[str]]
    instruction: Union[str, List[str]]
    address: Union[str, List[str]]
    value: Union[str, List[str]]
    encapsulation: Optional[Union[str, List[str]]] = None
    comment: Optional[Union[str, List[str]]] = None
    params: Optional[Union[Dict[str, Union[str, List[str]]], List[Union[str, None]]]] = None
    flag: Optional[Union[str, List[Union[str, None]]]] = None
    params_value: Optional[Union[str, List[Union[str, None]]]] = None


class TrimActionItemModel(ActionItemModel):
    trim_idx: Optional[Union[str, List[Union[str, None]]]] = None
    index_start: Optional[Union[str, List[Union[str, None]]]] = None
    index_stop: Optional[Union[str, List[Union[str, None]]]] = None
    index_step: Optional[Union[str, List[Union[str, None]]]] = None
    spec_low: Optional[Union[str, List[Union[str, None]]]] = None
    target: Optional[Union[str, List[Union[str, None]]]] = None
    spec_high: Optional[Union[str, List[Union[str, None]]]] = None
    unit_scale: Optional[Union[str, List[Union[str, None]]]] = None
    in_target: Optional[Union[str, List[Union[str, None]]]] = None
    out_target: Optional[Union[str, List[Union[str, None]]]] = None


class ConfigurationModel(BaseModel):
    params: List[str] = Field(default_factory=list)
    actions: List[Optional[Union[ActionItemModel, TrimActionItemModel]]] = Field(default_factory=list)
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

    # def __repr__(self):
    #     return self.__str__()

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, item):
        return hasattr(self, item)

    # def __len__(self):
    #     return len(self.__dict__)
    #
    # def __iter__(self):
    #     return iter(self.__dict__)
    #
    # def __dir__(self):
    #     return list(self.__dict__)
    #
    # def __bool__(self):
    #     return bool(self.__dict__)
    #
    # def __eq__(self, other):
    #     if not isinstance(other, ConfigurationModel):
    #         return False
    #     return self.__dict__ == other.__dict__