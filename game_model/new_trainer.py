import torch
import random
from game_data.game_config_data import GameConfigData
from game_model.game import Game
from game_model.AI_model.model import SplendidSplendorModel
from game_model.AI_model.gamestate_input import GamestateInputVector
from game_model.AI_model.maps import map_to_AI_input
from game_model.AI_model.action_output import ActionOutput
from game_model.AI_model.maps import map_from_AI_output
from game_model.game_runner import step_game
from game_model.turn import Action_Type, Turn

# Hyperparameters
N_HIDDEN_LAYERS = 5
HIDDEN_LAYER_WIDTH = 100
LEARNING_RATE = 0.001
GAMMA = 0.9  # Discount factor
MEMORY_SIZE = 10000  # Replay memory size
BATCH_SIZE = 32  # Mini-batch size

# Load game configuration data
game_config = GameConfigData.read_file("./game_data/cards.csv")

# Create game model
game = Game(player_count=4, game_config=game_config)

# Map game state to AI input
ai_input = map_to_AI_input(game)

# Create model
input_shape_dict = ai_input
output_shape_dict = ActionOutput().in_dict_form()
model = SplendidSplendorModel(input_shape_dict, output_shape_dict, HIDDEN_LAYER_WIDTH, N_HIDDEN_LAYERS)

# Define loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Initialize replay memory
memory: list[tuple[dict[str, torch.Tensor], Turn, int, dict[str, torch.Tensor]]] = []

def train(steps: int = 20):
    """Train the model for a given number of steps."""
    for step in range(steps):
        # Get current game state
        ai_input = map_to_AI_input(game)

        # Get model's predicted action
        forward_result = model.forward(ai_input)
        next_action = _get_next_action_from_forward_result(forward_result, game)
        if not isinstance(next_action, Turn):
            print(next_action)
            return

        # Take action and get reward
        next_player = game.get_current_player()
        step_status = step_game(game, next_action)
        reward = _get_reward(game, step_status)

        # Get next game state
        next_ai_input = map_to_AI_input(game)

        # Store transition in replay memory
        memory.append((ai_input, next_action, reward, next_ai_input))
        if len(memory) > MEMORY_SIZE:
            memory.pop(0)
        
        # Sample mini-batch from replay memory
        batch = random.sample(memory, min(len(memory), BATCH_SIZE))
        states, actions, rewards, next_states = zip(*batch)

        # Convert inputs to tensors
        states = [GamestateInputVector(x).to_tensor() for x in states]
        states = torch.cat(states, dim=0)
        actions = [ActionOutput.from_turn(x).to_tensor() for x in actions]
        actions = torch.cat(actions, dim=0)
        rewards = torch.tensor(rewards)
        next_states = [GamestateInputVector(x).to_tensor() for x in next_states]
        next_states = torch.cat(next_states, dim=0)

        # Get model's predicted Q-values for next states
        next_q_values = model.forward(next_states).values()
        next_q_values = torch.cat(list(next_q_values), dim=1)

        # Compute expected Q-values
        expected_q_values = rewards + GAMMA * next_q_values.max(dim=1)[0].unsqueeze(1)

        # Compute loss
        q_values = model.forward(states).values()
        q_values = torch.cat(list(q_values), dim=1)
        q_values = q_values.gather(1, actions)
        loss = loss_fn(q_values, expected_q_values)

        # Backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def _get_next_action_from_forward_result(forward: dict[str, torch.Tensor], game: Game) -> Turn | str:
    """Get the next action from the model's forward pass result."""
    next_action = ActionOutput()
    next_action.map_dict_into_self(forward)
    return map_from_AI_output(next_action, game, game.get_current_player())

def _get_reward(game: Game, step_status: str) -> float:
    """Determine the reward for the current game state."""
    # If the game ended, return a large negative or positive reward
    if step_status == "game_over":
        return -100.0
    if step_status == "victory":
        return 100.0

    # Otherwise, return a small positive reward for making progress
    return 1.0

def play_single_game():
    """Play a single game using the trained model."""
    game.reset()
    while True:
        # Get current game state
        ai_input = map_to_AI_input(game)

        # Get model's predicted action
        forward_result = model.forward(ai_input)
        next_action = _get_next_action_from_forward_result(forward_result, game)
        if not isinstance(next_action, Turn):
            print(next_action)
            return

        # Take action and get reward
        next_player = game.get_current_player()
        step_status = step_game(game, next_action)
        reward = _get_reward(game, step_status)

        # If the game ended, return the reward
        if step_status is not None:
            return reward

# Train the model
train(1000)

# Play a single game using the trained model
reward = play_single_game()
print("Game reward:", reward)

