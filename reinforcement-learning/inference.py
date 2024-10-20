import tensorflow as tf
import random
import typing
import heapq
from collections import deque
import time
import numpy as np

MAX_DEPTH = 3  # Depth of the minimax search tree
TIME_LIMIT = 0.25  # Maximum time in seconds per move

# Load the pre-trained model
model = tf.keras.models.load_model("./vVv54.h5")

def move(game_state: typing.Dict) -> typing.Dict:
    # Preprocess the game state into the expected input shape
    #print(game_state)
    input_data = preprocess_game_state(game_state)

    # Run inference
    predictions = model.predict(input_data)
    print(predictions)

    # Determine the current direction of the snake
    my_head = game_state["you"]["body"][0]
    my_neck = game_state["you"]["body"][1]
    
    if my_neck["x"] < my_head["x"]:
        current_direction = "right"
    elif my_neck["x"] > my_head["x"]:
        current_direction = "left"
    elif my_neck["y"] < my_head["y"]:
        current_direction = "up"
    else:
        current_direction = "down"

    # Remap Q-values to actual directions based on current direction
    if current_direction == 'up':
        move_direction = ['left', 'up', 'right'][np.argmax(predictions)]
    elif current_direction == 'down':
        move_direction = ['right', 'down', 'left'][np.argmax(predictions)]
    elif current_direction == 'left':
        move_direction = ['down', 'left', 'up'][np.argmax(predictions)]
    elif current_direction == 'right':
        move_direction = ['up', 'right', 'down'][np.argmax(predictions)]

    print(f"MOVE {game_state['turn']}: {move_direction}")
    return {"move": move_direction}

import numpy as np


def preprocess_game_state(game_state):
    board_width = game_state['board']['width']
    board_height = game_state['board']['height']
    
    # Initialize a 21x21x4 grid
    input_grid = np.zeros((21, 21, 4), dtype=np.float32)
    
    # Calculate padding to center the board around the snake’s head
    pad_x = (21 - board_width) // 2
    pad_y = (21 - board_height) // 2

    # Channel 1: Snake heads (negated values)
    for snake in game_state['board']['snakes']:
        head = snake['body'][0]
        head_x = head['x'] + pad_x
        head_y = head['y'] + pad_y
        head_value = -(len(snake['body']) - len(game_state['you']['body']) + 0.5) * (1 / max(board_width, board_height))
        input_grid[head_y, head_x, 0] = head_value

    # Channel 2: Obstacles (walls and bodies, negated values)
    for snake in game_state['board']['snakes']:
        for i, segment in enumerate(snake['body'][:-1]):  # Exclude the tail
            segment_x = segment['x'] + pad_x
            segment_y = segment['y'] + pad_y
            turns_left = len(snake['body']) - i - 1
            input_grid[segment_y, segment_x, 1] = -turns_left * (1 / max(board_width, board_height))

    # Add walls as obstacles (negated)
    input_grid[:pad_y, :, 1] = -1  # Top wall
    input_grid[-pad_y:, :, 1] = -1  # Bottom wall
    input_grid[:, :pad_x, 1] = -1  # Left wall
    input_grid[:, -pad_x:, 1] = -1  # Right wall

    # Channel 3: Food (valued 1.0)
    for food in game_state['board']['food']:
        food_x = food['x'] + pad_x
        food_y = food['y'] + pad_y
        input_grid[food_y, food_x, 2] = 1.0

    # Channel 4: Snake health (relative health of each snake)
    for snake in game_state['board']['snakes']:
        for segment in snake['body']:
            segment_x = segment['x'] + pad_x
            segment_y = segment['y'] + pad_y
            input_grid[segment_y, segment_x, 3] = snake['health'] * (1 / 100)  # Health scaled between 0 and 1

    return np.expand_dims(input_grid, axis=0)  # Add a batch dimension

def info() -> typing.Dict:
    print("INFO")
    return {
        "apiversion": "1",
        "author": "vagias",
        "color": "#FF0000",  # Hex color for red
        "head": "space-helmet",
        "tail": "bolt",
    }


# Directions and their respective x, y coordinate changes
directions = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}



if __name__ == "__main__":
    from server import run_server

    run_server({"info": info, "start": start, "move": move, "end": end})