from game_model.action import Action


class Actor:
    def __init__(self):
        print("actor created")
    
    def execute_action(self, action : Action):
        raise "not implemented"
    