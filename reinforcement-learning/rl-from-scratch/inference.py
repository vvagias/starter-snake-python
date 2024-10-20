import numpy as np
import torch
import torch.nn as nn
import typing

class DQN(nn.Module):
    def __init__(self, board_size):
        super(DQN, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * board_size * board_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def move(game_state: typing.Dict) -> typing.Dict:
    # Parse the game_state to create the state tensor
    board_height = game_state['board']['height']
    board_width = game_state['board']['width']
    board_size = max(board_height, board_width)

    # Initialize the state tensor
    state = np.zeros((policy_net.board_size, policy_net.board_size, 4), dtype=np.float32)

    # Get our snake
    you = game_state['you']
    you_body = you['body']
    you_head = you['head']
    # Get other snakes
    snakes = game_state['board']['snakes']
    # Get food
    food_list = game_state['board']['food']
    # Get hazards (if any)
    hazards = game_state['board']['hazards']

    # Channel 0: Our snake's head
    state[you_head['x'], you_head['y'], 0] = 1

    # Channel 1: All snake bodies (including ours)
    for snake in snakes:
        for segment in snake['body']:
            state[segment['x'], segment['y'], 1] = 1

    # Channel 2: Food
    for food in food_list:
        state[food['x'], food['y'], 2] = 1

    # Channel 3: Hazards (assuming walls are hazards)
    for hazard in hazards:
        state[hazard['x'], hazard['y'], 3] = 1

    # Convert state to tensor and adjust axes
    state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

    # Get action from the policy network
    with torch.no_grad():
        action = policy_net(state_tensor).max(1)[1].item()

    # Map action to move
    # The action indices are 0: left, 1: straight, 2: right
    # We need to determine the current direction and map the action to a move

    # First, get our current direction
    you_body = you['body']
    if len(you_body) >= 2:
        head = you_body[0]
        neck = you_body[1]
        dx = head['x'] - neck['x']
        dy = head['y'] - neck['y']
        if dx == 1:
            current_direction = 'right'
        elif dx == -1:
            current_direction = 'left'
        elif dy == -1:
            current_direction = 'up'    # Corrected
        elif dy == 1:
            current_direction = 'down'  # Corrected
        else:
            current_direction = 'up'  # Default
    else:
        current_direction = 'up'  # Default

    # Map action to new direction
    directions = ['up', 'right', 'down', 'left']
    idx = directions.index(current_direction)
    if action == 0:  # Turn left
        new_direction = directions[(idx - 1) % 4]
    elif action == 1:  # Straight
        new_direction = current_direction
    elif action == 2:  # Turn right
        new_direction = directions[(idx + 1) % 4]

    move_response = {"move": new_direction}
    print(f"MOVE {game_state['turn']}: {move_response['move']}")
    return move_response

def info() -> typing.Dict:
    return {
        "apiversion": "1",
        "author": "YourName",
        "color": "#FF0000",
        "head": "default",
        "tail": "default",
    }

def start(game_state: typing.Dict):
    print("GAME START")

def end(game_state: typing.Dict):
    print("GAME OVER")

if __name__ == "__main__":
    from server import run_server

    BOARD_SIZE = 11  # Should match the board_size used in training

    # Load the trained model
    policy_net = DQN(BOARD_SIZE)
    policy_net.load_state_dict(torch.load('dqn_model.pth', map_location=torch.device('cpu')))
    policy_net.eval()

    run_server({"info": info, "start": start, "move": move, "end": end})