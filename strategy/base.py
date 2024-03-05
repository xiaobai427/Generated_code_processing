# 策略接口
from abc import abstractmethod, ABC
from typing import Optional, List, Dict

from models.base import ActionItemModel


class ActionStrategy(ABC):
    @abstractmethod
    def execute_action(self,
                       action: ActionItemModel,
                       params: Optional[Dict[str, str]] = None,
                       state: bool = None) -> List[str]:
        raise NotImplementedError