import numpy as np

# Reward Variables
REWARD_FOOD = 100       # Reward for eating food
REWARD_COLLISION = -100 # Penalty for collision with wall or self
REWARD_STEP = -0.1      # Small penalty for each step to encourage shorter paths
REWARD_CLOSER = 1       # Reward for moving closer to food
REWARD_FARTHER = -1     # Penalty for moving away from food
REWARD_SURVIVAL = 0.1   # Reward for staying alive each step

class SnakeEnv:
    def __init__(self, board_size=11):
        self.board_size = board_size
        self.reset()

    def reset(self):
        # Reset snake and place it in the middle
        self.snake = [(self.board_size // 2, self.board_size // 2)]
        self.direction = (0, -1)  # Initially moving up
        self.done = False

        # Initialize walls before placing food
        self.walls = self._place_walls()
        self.food = self._place_food()

        return self._get_state()

    def _place_food(self):
        while True:
            food = (np.random.randint(0, self.board_size), np.random.randint(0, self.board_size))
            if food not in self.snake and food not in self.walls:
                return food

    def _place_walls(self):
        walls = []
        num_walls = np.random.randint(0, 5)  # Random number of walls
        for _ in range(num_walls):
            wall_pos = (np.random.randint(0, self.board_size), np.random.randint(0, self.board_size))
            if wall_pos != self.snake[0]:
                walls.append(wall_pos)
        return walls

    def _get_state(self):
        # Create the 3D tensor representing the current state
        state = np.zeros((self.board_size, self.board_size, 4), dtype=np.float32)

        # Channel 0: Snake head
        head_x, head_y = self.snake[0]
        state[head_x, head_y, 0] = 1

        # Channel 1: Snake body
        for segment in self.snake[1:]:
            state[segment[0], segment[1], 1] = 1

        # Channel 2: Food
        food_x, food_y = self.food
        state[food_x, food_y, 2] = 1

        # Channel 3: Walls
        for wall in self.walls:
            wall_x, wall_y = wall
            state[wall_x, wall_y, 3] = 1

        return state

    def get_valid_actions(self):
        # Returns a list of valid actions at the current state
        valid_actions = []
        for action in range(3):  # Actions: 0 - Left, 1 - Straight, 2 - Right
            new_direction = self._get_new_direction(action)
            new_head = (self.snake[0][0] + new_direction[0], self.snake[0][1] + new_direction[1])
            if not self._is_collision(new_head):
                valid_actions.append(action)
        return valid_actions

    def step(self, action):
        # Apply small penalty each step to encourage shorter paths
        reward = REWARD_STEP

        # Compute distance to food before moving
        distance_prev = self._compute_distance(self.snake[0], self.food)

        # Convert action into new direction (0: left, 1: straight, 2: right)
        self.direction = self._get_new_direction(action)

        # Move the snake
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])

        # Check for collision
        if self._is_collision(new_head):
            reward += REWARD_COLLISION  # Penalty for collision
            self.done = True
            return self._get_state(), reward, self.done

        # Update the snake's position
        self.snake.insert(0, new_head)

        # Check if food is eaten
        if new_head == self.food:
            reward += REWARD_FOOD  # Reward for eating food
            self.food = self._place_food()
        else:
            # Remove tail segment
            self.snake.pop()

        # Survival reward
        reward += REWARD_SURVIVAL

        # Compute distance to food after moving
        distance_curr = self._compute_distance(self.snake[0], self.food)

        # Reward for moving closer or farther from food
        if distance_curr < distance_prev:
            reward += REWARD_CLOSER
        else:
            reward += REWARD_FARTHER

        # Check for self-collision after moving
        if self.snake[0] in self.snake[1:]:
            reward += REWARD_COLLISION  # Penalty for self-collision
            self.done = True

        return self._get_state(), reward, self.done

    def _compute_distance(self, pos1, pos2):
        # Manhattan distance
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_collision(self, head):
        return (head in self.snake or
                head[0] < 0 or head[0] >= self.board_size or
                head[1] < 0 or head[1] >= self.board_size or
                head in self.walls)

    def _get_new_direction(self, action):
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        current_idx = directions.index(self.direction)
        if action == 0:  # Turn left
            new_idx = (current_idx - 1) % 4
        elif action == 2:  # Turn right
            new_idx = (current_idx + 1) % 4
        else:  # Straight
            new_idx = current_idx
        return directions[new_idx]