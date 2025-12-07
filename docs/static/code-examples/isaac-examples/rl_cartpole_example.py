#!/usr/bin/env python3

"""
Reinforcement Learning CartPole Example for Isaac Sim
This script demonstrates how to train a simple RL agent using Isaac Gym.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from gym import spaces


class CartPoleAgent(nn.Module):
    """Simple neural network for CartPole RL agent"""
    def __init__(self, input_size, hidden_size, output_size):
        super(CartPoleAgent, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.network(x)


class SimpleRLTrainer:
    """Simple RL trainer for CartPole environment"""
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n

        # Create agent
        self.agent = CartPoleAgent(input_size=self.state_size, hidden_size=64, output_size=self.action_size)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        # Training parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()  # Explore

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.agent(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())  # Exploit

    def train_step(self, state, action, reward, next_state, done):
        """Train the agent on a single step"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        target = reward
        if not done:
            next_q_values = self.agent(next_state_tensor)
            target = reward + self.gamma * torch.max(next_q_values).item()

        target_f = self.agent(state_tensor).clone()
        target_f[0][action] = target

        self.optimizer.zero_grad()
        predictions = self.agent(state_tensor)
        loss = self.loss_fn(predictions, target_f.detach())
        loss.backward()
        self.optimizer.step()

    def train(self, episodes=1000):
        """Train the agent for specified episodes"""
        scores = []
        for episode in range(episodes):
            state = self.env.reset()
            if isinstance(state, tuple):  # Handle new gym versions
                state = state[0]

            total_reward = 0
            done = False

            while not done:
                action = self.get_action(state)
                step_result = self.env.step(action)

                # Handle different gym versions
                if len(step_result) == 4:
                    next_state, reward, done, info = step_result
                else:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated

                self.train_step(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            scores.append(total_reward)

            # Decay exploration rate
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Print progress
            if episode % 100 == 0:
                avg_score = sum(scores[-100:]) / min(len(scores), 100)
                print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {self.epsilon:.2f}")

        print("Training completed!")


def main():
    """Main function"""
    print("Starting Isaac Sim RL Example")

    try:
        # Create CartPole environment (for demonstration - in Isaac Sim you'd use a real robot env)
        import gym
        env = gym.make('CartPole-v1')

        # Create and train RL agent
        trainer = SimpleRLTrainer(env)
        trainer.train(episodes=1000)

        # Close environment
        env.close()

        print("RL Training Example completed successfully")

    except ImportError:
        print("Gym not available, showing conceptual code only")
        print("In Isaac Sim, you would use Isaac Gym environments instead of standard gym")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()