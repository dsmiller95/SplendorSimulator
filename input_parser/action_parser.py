

import asyncio
from game_model.resource_types import ResourceType


from game_model.turn import Action_Type, Turn
from utilities.subsamples import parse_int

async def get_input():
    return (await asyncio.get_running_loop().run_in_executor(None, input))

resource_type_description = """
0: Ruby
1: Emerald
2: Sapphire
3: Diamond
4: Onyx"""

def get_action_from_user() -> Turn:
    print(
"""
Enter the type of action you would like to take, or "exit":
1: Take 3 tokens
2: take 2 of the same token
3: buy a card
4: reserve a card
""")
    action_type: Action_Type = None
    try:
        input_val = input()
        if input_val == "exit":
            return False
        action_type = Action_Type(int(input_val))
    except:
        print("Enter a number in the range [1, 4]")
        return None
    
    match action_type:
        case Action_Type.TAKE_THREE_UNIQUE:
            print("which three resources will you claim, separated by commas?\n" + resource_type_description)
            try:
                claimed = [ResourceType(int(x)) for x in input().split(",")]
                claimed = [x for x in claimed if x != ResourceType.GOLD]
            except:
                print("enter a valid resource type, one of the 5 listed")
                return None
            resources = [(1 if ResourceType(x) in claimed else 0) for x in range(0, 5)]
            return Turn(Action_Type.TAKE_THREE_UNIQUE, resources)
        case Action_Type.TAKE_TWO:
            print("which resources will you claim two of?\n" + resource_type_description)
            try:
                claimed = ResourceType(int(input()))
            except:
                print("enter a valid resource type, one of the 5 listed")
            if(claimed == ResourceType.GOLD):
                print("enter a valid resource type, one of the 5 listed")
                return None
            resources = [0 for x in range(0, 5)]
            resources[claimed.value] = 2
            return Turn(Action_Type.TAKE_TWO, resources)
        case other:
            print("unimplemented/unknown action")
    
            
