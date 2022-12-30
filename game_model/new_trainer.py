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
mem_slice_type = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], int, dict[str, torch.Tensor]]
memory: list[mem_slice_type] = []

def train(steps: int = 100):
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
        step_status = step_game(game, next_action)
        reward = _get_reward(game, step_status)

        # Get next game state
        next_ai_input = map_to_AI_input(game)

        # Store transition in replay memory
        memory.append((ai_input, forward_result, reward, next_ai_input))
        if len(memory) > MEMORY_SIZE:
            memory.pop(0)
        if len(memory) < 32:
            continue
        
        # Sample mini-batch from replay memory
        batch = random.sample(memory, min(len(memory), BATCH_SIZE))
        states, actions, rewards, next_states = zip(*batch)

        # Convert inputs to tensors
        states = concat_dict_list_to_dict_tensors(states)
        actions = concat_dict_list_to_dict_tensors(actions)
        rewards = torch.tensor(rewards)
        next_states = concat_dict_list_to_dict_tensors(next_states)

        # Get model's predicted Q-values for next states
        next_q_values = model.forward(next_states).values()
        next_q_values = torch.cat(list(next_q_values), dim=1)

        # Compute expected Q-values
        expected_q_values = rewards + GAMMA * next_q_values.max(dim=1)[0].unsqueeze(1)

        # Compute loss
        q_values = model.forward(states).values()
        q_values = torch.cat(list(q_values), dim=1)
        ##q_values = {name: val.gather(1, actions[name].long()) for name, val in q_values.items()}
        action_flat = concat_dict_tensors_to_flat(actions)
        _, action_indices = action_flat.max(dim=1)
        action_indices_unsqueezed = action_indices.unsqueeze(1)
        q_values = q_values.gather(1, action_indices_unsqueezed)
        loss = loss_fn(q_values, expected_q_values)

        # Backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def concat_dict_tensors_to_flat(input: dict[str, torch.Tensor]):
    return torch.concat(list(input.values()), dim=1)

def concat_dict_list_to_dict_tensors(dict_list: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(dict_list) <= 0:
        return None
    return {key:torch.stack([x[key] for x in dict_list], dim=0) for key in dict_list[0]}
    

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
    game = Game(player_count=4, game_config=game_config)
    run_count = 0
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
        step_status = step_game(game, next_action)
        reward = _get_reward(game, step_status)

        run_count += 1
        # If the game ended, return the reward
        if step_status is not None or run_count > 100:
            return reward

# Train the model
train(100)

# Play a single game using the trained model
reward = play_single_game()
print("Game reward:", reward)

