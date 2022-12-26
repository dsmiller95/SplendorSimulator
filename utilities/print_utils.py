
from game_model.resource_types import ResourceType

def stringify_resources(resources: list[int], ignore_empty: bool = False) -> str:
    result_str = ""
    for idx, amount in enumerate(resources):
        if ignore_empty and amount <= 0:
            continue
        result_str += "[" + str(amount).ljust(2) + ResourceType(idx).name + "] "
    return result_str