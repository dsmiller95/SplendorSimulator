

import asyncio
from game_model.actor import Actor
from game_model.game import Game
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

def get_action_from_user(player: Actor, game: Game) -> Turn:
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
        case Action_Type.BUY_CARD:
            if player.has_reserved_cards() and make_choice("purchase a reserved card?"):
                print("which reserved card will you purchase? 1/2/3")
                for i, card in enumerate(player.reserved_cards):
                    print(str(i + 1) + ": " + ("None" if card is None else card.describe_state()))
                reserved_index = int(get_option(["1", "2", "3"], "enter a number in the range [1-3]")) - 1
                complete_index = game.config.total_available_card_indexes() + reserved_index
                return Turn(Action_Type.BUY_CARD, card_index=complete_index)
            selected_index = select_game_board_card(game, include_decks=False)
            return Turn(Action_Type.BUY_CARD, card_index=selected_index)
        case Action_Type.RESERVE_CARD:
            selected_index = select_game_board_card(game, include_decks=True)
            return Turn(Action_Type.RESERVE_CARD, card_index=selected_index)
        case other:
            print("unimplemented/unknown action")
    

def select_game_board_card(game: Game, include_decks: bool) -> int:
    print("select a card tier 1/2/3")
    tier = int(get_option(["1", "2", "3"], "enter a valid card tier, 1 2 or 3")) - 1
    print("select a card in the tier:")
    valid_choices: list[int] = []
    if include_decks:
        print("0: " + "Top of Deck")
        valid_choices.append("0")
    for i, card in enumerate(game.open_cards[tier]):
        print(str(i + 1) + ": " + card.describe_state())
        valid_choices.append(str(i + 1))
    in_tier = int(get_option(valid_choices, "select a valid card id"))
    return tier * (game.config.open_cards_per_tier + 1) + in_tier

def get_option(valid_selection: list[str], reminder: str):
    while True:
        next_select = input()
        if next_select in valid_selection:
            return next_select
        print(reminder)

def make_choice(question: str) -> bool:
    while True:
        print(question + " Y/N")
        txt = input().lower()
        if txt.startswith("y"):
            return True
        elif txt.startswith("n"):
            return False