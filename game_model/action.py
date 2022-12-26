from enum import Enum

from game_model.resource_types import ResourceType

class Action_Type(Enum):
    TAKE_UNIQUE = 1,
    TAKE_THREE = 2,
    BUY_CARD = 3,
    RESERVE_CARD = 4

class Action:
    def __init__(self, action_type: Action_Type, resources: list[ResourceType] = None, card_id: int = None):
        self.action_type = action_type
        self.resources = resources
        self.card_id = card_id
    
    def validate(self):
        raise "not implemented"