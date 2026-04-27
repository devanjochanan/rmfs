import os
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DeepQNetwork:
    def __init__(self, state_size, action_size, model_name=None, load_existing_model=False):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_existing_model:
            self.model = self.load_model()
        else:
            self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_model(self):
        """Load model from file if exists."""
        model_path = f"saved_models/{self.model_name}.pt"
        if os.path.exists(model_path):
            return torch.load(model_path).to(self.device)
        else:
            print(f"No model found at {model_path}. Building a new model.")
            return self._build_model()

    def _build_model(self):
        """Neural Network for Deep Q learning Model."""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)

        states = torch.tensor([m[0] for m in minibatch], dtype=torch.float32).to(self.device)
        actions = torch.tensor([m[1] for m in minibatch]).to(self.device)
        rewards = torch.tensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.tensor([m[3] for m in minibatch], dtype=torch.float32).to(self.device)
        dones = torch.tensor([m[4] for m in minibatch], dtype=torch.float32).to(self.device)

        # Predict Q-values for current states
        q_values = self.model(states)
        q_values_next = self.model(next_states).detach()

        targets = rewards + (self.gamma * torch.max(q_values_next, dim=1)[0] * (1 - dones))
        targets_full = q_values.clone()
        targets_full[range(batch_size), actions] = targets

        self.optimizer.zero_grad()
        loss = nn.MSELoss()(q_values, targets_full)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, model_name, tick):
        torch.save(self.model, f"saved_models/{model_name}-{tick}.pt")
