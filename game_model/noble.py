

from game_model.resource_types import ResourceType

class Noble:
    def __init__(self, resource_cost: list[int], reward_points: int):
        self.costs = resource_cost
        self.points = reward_points
    
    