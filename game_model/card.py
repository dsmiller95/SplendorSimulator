from game_model.resource_types import ResourceType

class Card:
    def __init__(
        self, 
        card_level: int,
        resource_cost: list[int],
        reward_resource: ResourceType,
        reward_points: int,
        card_id: int):
        self.tier = card_level
        self.costs = resource_cost
        self.returns = reward_resource
        self.points = reward_points
        self.id = card_id
        
    def describe_state(self) -> str:
        result = ""
        result += "Points: " + str(self.points).ljust(3)
        result += "Cost: ["
        for index, cost in enumerate(self.costs):
            if cost <= 0:
                continue
            resource_name = ResourceType(index)
            result = result + str(cost) + " " + resource_name.name + ", "
        result += "] Resource: " + self.returns.name
        return result
    
    def as_serializable_data(self) -> dict:
        return {
            "id": self.id,
            "costs": self.costs,
            "points": self.points,
            "returns": self.returns,
            "tier": self.tier
        }