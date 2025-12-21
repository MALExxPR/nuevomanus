import json
from pathlib import Path
import numpy as np

class QLearningTradingAgent:
    """Agente de trading basado en Q-learning simple."""

    def __init__(self, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.995, epsilon_min=0.01, model_path=None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = {}
        self.model_path = Path(model_path) if model_path else None

    def _state_key(self, state):
        return tuple(int(x) for x in state)

    def choose_action(self, state, training=True):
        key = self._state_key(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(3)  # hold, buy, sell
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(3)
        return int(np.argmax(self.q_table[key]))

    def update(self, state, action, reward, next_state, done):
        key = self._state_key(state)
        next_key = self._state_key(next_state)
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(3)
        target = reward
        if not done:
            target += self.gamma * np.max(self.q_table[next_key])
        self.q_table[key][action] += self.alpha * (target - self.q_table[key][action])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path=None):
        path = Path(path) if path else self.model_path
        if path is None:
            return
        with open(path, 'w') as f:
            json.dump({str(k): v.tolist() for k, v in self.q_table.items()}, f)

    def load(self, path=None):
        path = Path(path) if path else self.model_path
        if path is None or not path.exists():
            return False
        with open(path, 'r') as f:
            data = json.load(f)
        self.q_table = {tuple(map(int, k.strip('()').split(','))): np.array(v) for k, v in data.items()}
        return True

    @staticmethod
    def state_from_row(row):
        """Discretiza un estado a partir de una fila de datos con indicadores."""
        a = 1 if row['close'] > row['sma_5'] else -1
        b = 1 if row['close'] > row['sma_20'] else -1
        return (a, b)
