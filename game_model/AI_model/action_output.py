
class ActionOutput:
    def __init__(self):
        self.action_choice: list[float] = [0] * 4
        self.card_buy: list[float] = [0] * 15
        self.reserve_buy: list[float]= [0] * 3
        self.resource_token_draw: list[float] = [0] * 5
        self.noble_choice: list[float] = [0] * 5
        self.discard_choice: list[float] = [0]
        self.discard_amounts: list[float] = [0] * 6
