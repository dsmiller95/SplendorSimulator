

from game_model.resource_types import ResourceType

class Noble:
    def __init__(self, resource_cost: list[int], reward_points: int):
        self.costs = resource_cost
        self.points = reward_points
    
    def describe_self(self) -> str:
        result = "Points: " + str(self.points).ljust(3)
        result += "Cost: ["
        for index, cost in enumerate(self.costs):
            if cost <= 0:
                continue
            resource_name = ResourceType(index)
            result = result + str(cost) + " " + resource_name.name + ", "
        return result + "]"
    