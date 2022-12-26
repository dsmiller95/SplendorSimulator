from game_model.resource_types import ResourceType

class Card:
    def __init__(self, card_level: int, resource_cost: list[int], reward_resource: ResourceType, reward_points: int):
        self.tier = card_level
        self.costs = resource_cost
        self.reward = reward_resource
        self.points = reward_points
    