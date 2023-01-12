

from game_model.resource_types import ResourceType

class Noble:
    def __init__(self, resource_cost: list[int], reward_points: int, card_id: int):
        self.id = card_id
        self.costs = resource_cost
        self.points = reward_points
    
    def satisfied_by(self, resource_bank: list[int]) -> bool:
        for i, req in enumerate(self.costs):
            if resource_bank[i] < req:
                return False
        return True

    def describe_self(self) -> str:
        result = "Points: " + str(self.points).ljust(3)
        result += "Cost: ["
        for index, cost in enumerate(self.costs):
            if cost <= 0:
                continue
            resource_name = ResourceType(index)
            result = result + str(cost) + " " + resource_name.name + ", "
        return result + "]"
    
    def as_serializable_data(self) -> dict:
        return {
            "id": self.id,
            "costs": self.costs,
            "points": self.points
        }