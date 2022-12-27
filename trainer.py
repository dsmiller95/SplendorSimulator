from game_data.game_config_data import GameConfigData
from game_model.turn import Turn, Action_Type
from game_model.game import Game
from game_model.resource_types import ResourceType
from game_model.AI_model.model import SplendidSplendorModel
from game_model.AI_model.loss import *

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SplendidSplendorModel.to(device)

    game_config = GameConfigData.read_file("./game_data/cards.csv")
    game = Game(player_count=3, game_config=game_config)
    print(game.describe_common_state())
    first_action = Turn(
        action_type=Action_Type.TAKE_TWO,
        resources=[ResourceType.DIAMOND, ResourceType.ONYX, ResourceType.EMERALD]
        )