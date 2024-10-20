import numpy as np
from snake_env import SnakeEnv

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.exploration_min = 0.01

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.randint(self.action_size)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        # Decay exploration rate
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

def flatten_state(state):
    # Convert the 3D state representation to a flat index
    return hash(state.tostring()) % 10000  # Simplified hashing

# Initialize environment and agent
env = SnakeEnv(board_size=11)
agent = QLearningAgent(state_size=10000, action_size=3)  # 3 actions (left, straight, right)

episodes = 1000000
max_steps = 1000  # Max steps per episode

for e in range(episodes):
    state = flatten_state(env.reset())  # Reset the environment and flatten the state
    total_reward = 0
    done = False

    for step in range(max_steps):
        # Choose action using epsilon-greedy policy
        action = agent.choose_action(state)

        # Take the action, get next state, reward, and done flag
        next_state, reward, done = env.step(action)
        next_state = flatten_state(next_state)

        # Learn from the experience
        agent.learn(state, action, reward, next_state)

        # Update state
        state = next_state
        total_reward += reward

        if done:
            break

    print(f"Episode {e+1}/{episodes} - Total Reward: {total_reward}")

print("Training complete.")
import pickle

# After training is complete
with open("q_table.pkl", "wb") as f:
    pickle.dump(agent.q_table, f)

print("Training complete. Q-table saved to q_table.pkl.")
