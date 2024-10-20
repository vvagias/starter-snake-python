import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from snake_env import SnakeEnv
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

np.random.seed(0)
torch.manual_seed(0)

BATCH_SIZE = 64
GAMMA = 0.95
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 20000
TARGET_UPDATE = 10
NUM_EPISODES = 5000
MEMORY_SIZE = 500000
LEARNING_RATE = 1e-4
BOARD_SIZE = 11
SAVE_EVERY = 100
CHECKPOINT_DIR = 'checkpoints'
CHECKPOINT_FILENAME = 'checkpoint.pth'

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
        self.fc2 = nn.Linear(512, 3)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.view(-1, 128 * self.board_size * self.board_size)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def rotate_state(state, rotation):
    """Rotate the state by 90, 180, or 270 degrees."""
    return np.rot90(state, k=rotation)

def rotate_action(action, rotation):
    """Rotate the integer action (0: left, 1: straight, 2: right) according to the board rotation."""
    if rotation == 1:  # 90 degrees clockwise
        if action == 0:  # Left becomes Up (90 degree rotation)
            return 0
        elif action == 1:  # Straight remains Straight
            return 1
        elif action == 2:  # Right becomes Down (90 degree rotation)
            return 2
    elif rotation == 2:  # 180 degrees clockwise
        if action == 0:  # Left becomes Right (180 degree rotation)
            return 2
        elif action == 1:  # Straight remains Straight
            return 1
        elif action == 2:  # Right becomes Left (180 degree rotation)
            return 0
    elif rotation == 3:  # 270 degrees clockwise
        if action == 0:  # Left becomes Down (270 degree rotation)
            return 2
        elif action == 1:  # Straight remains Straight
            return 1
        elif action == 2:  # Right becomes Up (270 degree rotation)
            return 0
    return action  # No rotation for 0 degrees

env = SnakeEnv(board_size=BOARD_SIZE)

policy_net = DQN(BOARD_SIZE).to(device)
target_net = DQN(BOARD_SIZE).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

memory = deque(maxlen=MEMORY_SIZE)
steps_done = 0
start_episode = 0

def load_checkpoint():
    global start_episode
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_episode = checkpoint['episode'] + 1
            print(f"Resumed training from episode {start_episode}")
        else:
            print(f"Checkpoint file does not contain 'model_state_dict' or 'optimizer_state_dict'. Starting from scratch.")
    else:
        print("No checkpoint found, starting from scratch.")

load_checkpoint()  # Load the checkpoint at the start of training

def select_action(state, epsilon):
    global steps_done
    sample = np.random.rand()
    steps_done += 1

    if sample > epsilon:
        # Convert state to tensor and permute to the correct shape: [batch_size, channels, height, width]
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension
        state_tensor = state_tensor.permute(0, 3, 1, 2)  # From [batch_size, height, width, channels] to [batch_size, channels, height, width]
        
        # Get the action with the maximum Q-value
        return policy_net(state_tensor).max(1)[1].item()
    else:
        return np.random.randint(0, 3)  # Explore: Random action (0: left, 1: straight, 2: right)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return None

    # Sample transitions from memory
    transitions = np.random.choice(len(memory), BATCH_SIZE, replace=False)
    batch = [memory[idx] for idx in transitions]

    # Separate out the components of the batch
    state_batch = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32, device=device)
    action_batch = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device).unsqueeze(1)
    reward_batch = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
    next_state_batch = torch.tensor(np.stack([item[3] for item in batch]), dtype=torch.float32, device=device)
    done_batch = torch.tensor([item[4] for item in batch], dtype=torch.bool, device=device)

    # Permute the state and next state tensors to match the expected input shape for the model
    state_batch = state_batch.permute(0, 3, 1, 2)  # From [batch_size, height, width, channels] to [batch_size, channels, height, width]
    next_state_batch = next_state_batch.permute(0, 3, 1, 2)

    # Compute current Q values for actions taken
    q_values = policy_net(state_batch).gather(1, action_batch)

    # Compute next Q values for non-terminal states
    next_q_values = torch.zeros(BATCH_SIZE, device=device)
    next_q_values[~done_batch] = target_net(next_state_batch[~done_batch]).max(1)[0].detach()

    # Compute expected Q values
    expected_q_values = reward_batch + (GAMMA * next_q_values)

    # Compute loss
    loss = nn.functional.mse_loss(q_values.squeeze(), expected_q_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def rotate_experiences(state, action, next_state, reward):
    """Generate the 3 rotated states, actions, and update the experience memory."""
    for rotation in range(1, 4):  # Rotations 90, 180, 270 degrees
        rotated_state = rotate_state(state, rotation)
        rotated_action = rotate_action(action, rotation)
        rotated_next_state = rotate_state(next_state, rotation)
        memory.append((rotated_state, rotated_action, reward, rotated_next_state, done))

def save_checkpoint(episode):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)
    torch.save({
        'episode': episode,
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    print(f"Checkpoint saved at episode {episode}")

epsilon = EPSILON_START

for episode in range(start_episode, NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    episode_loss = 0
    steps = 0

    while True:
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)

        # Store the transition in memory
        memory.append((state, action, reward, next_state, done))

        # Also store the 3 other rotations of the state-action pair
        rotate_experiences(state, action, next_state, reward)

        # Perform one step of the optimization
        loss = optimize_model()
        if loss is not None:
            episode_loss += loss
            steps += 1

        total_reward += reward
        state = next_state

        if done:
            break

    # Update target network every TARGET_UPDATE episodes
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon - (EPSILON_START - EPSILON_END) / EPSILON_DECAY)

    # Logging
    print(f"Episode {episode + 1}/{NUM_EPISODES}, Total Reward: {total_reward:.2f}, Average Loss: {episode_loss / steps if steps > 0 else 0:.4f}, Epsilon: {epsilon:.4f}")

    # Save checkpoint every SAVE_EVERY episodes
    if episode % SAVE_EVERY == 0:
        save_checkpoint(episode)

# Save the final model
save_checkpoint(NUM_EPISODES)