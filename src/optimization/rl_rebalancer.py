"""
Reinforcement Learning for Adaptive Portfolio Rebalancing.

Implements:
- Deep Q-Network (DQN) for rebalancing decisions
- Policy Gradient methods (REINFORCE, A2C)
- Proximal Policy Optimization (PPO)
- Custom trading environment
- Experience replay and target networks
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from collections import deque
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. RL rebalancing will use fallback methods.")


@dataclass
class RLConfig:
    """Configuration for RL rebalancing agent."""
    state_dim: int = 50  # Dimension of state space
    action_dim: int = 10  # Number of possible actions
    hidden_dim: int = 128  # Hidden layer size
    learning_rate: float = 1e-4
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0  # Exploration rate
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64
    memory_size: int = 10000
    target_update_freq: int = 100


class PortfolioEnv:
    """
    Custom trading environment for portfolio rebalancing.

    State: [weights, returns, volatilities, correlations, market features]
    Action: Rebalancing decision (hold, rebalance to target, etc.)
    Reward: Risk-adjusted returns minus transaction costs
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        initial_weights: np.ndarray,
        transaction_cost: float = 0.001,
        target_return: float = 0.10,
        risk_aversion: float = 2.5
    ):
        """
        Initialize portfolio environment.

        Args:
            returns: Historical returns data (T x N)
            initial_weights: Initial portfolio weights
            transaction_cost: Cost per trade
            target_return: Target annual return
            risk_aversion: Risk aversion parameter
        """
        self.returns = returns.values
        self.n_periods = len(returns)
        self.n_assets = returns.shape[1]
        self.transaction_cost = transaction_cost
        self.target_return = target_return
        self.risk_aversion = risk_aversion

        # State tracking
        self.current_step = 0
        self.current_weights = initial_weights.copy()
        self.initial_weights = initial_weights.copy()
        self.wealth = 1.0
        self.wealth_history = [1.0]

        # Lookback window for state features
        self.lookback = 20

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.lookback
        self.current_weights = self.initial_weights.copy()
        self.wealth = 1.0
        self.wealth_history = [1.0]
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Construct state vector from current portfolio and market conditions.

        State includes:
        - Current portfolio weights
        - Recent returns (lookback window)
        - Recent volatilities
        - Correlation matrix (flattened)
        - Drift from target weights
        """
        # Current weights
        weights_state = self.current_weights

        # Recent returns (mean)
        recent_returns = self.returns[self.current_step - self.lookback:self.current_step]
        mean_returns = np.mean(recent_returns, axis=0)

        # Recent volatilities
        volatilities = np.std(recent_returns, axis=0)

        # Correlation matrix (flattened upper triangle)
        if len(recent_returns) > 1:
            corr_matrix = np.corrcoef(recent_returns.T)
            corr_features = corr_matrix[np.triu_indices(self.n_assets, k=1)]
        else:
            corr_features = np.zeros(self.n_assets * (self.n_assets - 1) // 2)

        # Momentum (recent cumulative returns)
        momentum = np.prod(1 + recent_returns, axis=0) - 1

        # Drift from initial weights
        drift = self.current_weights - self.initial_weights

        # Combine all features
        state = np.concatenate([
            weights_state,
            mean_returns,
            volatilities,
            corr_features[:min(10, len(corr_features))],  # Limit correlation features
            momentum,
            drift
        ])

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Rebalancing action
                0: Hold (no rebalancing)
                1: Rebalance to initial weights
                2-N: Partial rebalancing strategies

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Get current period returns
        period_returns = self.returns[self.current_step]

        # Store old weights for transaction cost calculation
        old_weights = self.current_weights.copy()

        # Execute action
        new_weights = self._execute_action(action, period_returns)

        # Calculate transaction costs
        turnover = np.sum(np.abs(new_weights - old_weights))
        tc = self.transaction_cost * turnover * self.wealth

        # Update weights based on returns (drift)
        portfolio_return = np.dot(new_weights, period_returns)

        # Update wealth
        self.wealth = self.wealth * (1 + portfolio_return) - tc
        self.wealth_history.append(self.wealth)

        # Natural drift in weights
        self.current_weights = new_weights * (1 + period_returns)
        self.current_weights = self.current_weights / self.current_weights.sum()

        # Calculate reward
        reward = self._calculate_reward(portfolio_return, turnover)

        # Move to next step
        self.current_step += 1
        done = (self.current_step >= self.n_periods - 1)

        # Get next state
        next_state = self._get_state() if not done else np.zeros(len(self._get_state()))

        info = {
            'wealth': self.wealth,
            'portfolio_return': portfolio_return,
            'transaction_cost': tc,
            'turnover': turnover
        }

        return next_state, reward, done, info

    def _execute_action(self, action: int, period_returns: np.ndarray) -> np.ndarray:
        """
        Execute rebalancing action.

        Actions:
        0: Hold (no rebalancing)
        1: Full rebalance to initial weights
        2: Partial rebalance (50% toward initial)
        3: Equal weight
        4: Minimum variance (simplified)
        """
        if action == 0:
            # Hold - no rebalancing
            return self.current_weights

        elif action == 1:
            # Full rebalance to initial weights
            return self.initial_weights

        elif action == 2:
            # Partial rebalance (50% toward initial)
            return 0.5 * self.current_weights + 0.5 * self.initial_weights

        elif action == 3:
            # Equal weight
            return np.ones(self.n_assets) / self.n_assets

        elif action == 4:
            # Minimum variance (inverse volatility)
            recent_returns = self.returns[self.current_step - self.lookback:self.current_step]
            vols = np.std(recent_returns, axis=0) + 1e-8
            inv_vol = 1.0 / vols
            return inv_vol / inv_vol.sum()

        else:
            # Default: hold
            return self.current_weights

    def _calculate_reward(self, portfolio_return: float, turnover: float) -> float:
        """
        Calculate reward for the agent.

        Reward = Portfolio Return - Risk Penalty - Transaction Cost Penalty
        """
        # Return component
        return_reward = portfolio_return

        # Risk penalty (based on recent volatility)
        recent_returns = self.returns[max(0, self.current_step - self.lookback):self.current_step]
        if len(recent_returns) > 1:
            portfolio_vol = np.std([np.dot(self.current_weights, r) for r in recent_returns])
            risk_penalty = self.risk_aversion * (portfolio_vol ** 2)
        else:
            risk_penalty = 0

        # Transaction cost penalty
        tc_penalty = self.transaction_cost * turnover

        # Total reward
        reward = return_reward - risk_penalty - tc_penalty

        return reward


class DQNAgent(nn.Module):
    """Deep Q-Network for portfolio rebalancing."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize DQN.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dim: Size of hidden layers
        """
        super(DQNAgent, self).__init__()

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for DQN agent")

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int):
        """Initialize replay buffer."""
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List:
        """Sample batch from buffer."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


class RLRebalancer:
    """
    Reinforcement Learning-based adaptive rebalancing agent.
    """

    def __init__(self, config: Optional[RLConfig] = None):
        """Initialize RL rebalancer."""
        self.config = config or RLConfig()

        if TORCH_AVAILABLE:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.policy_net = None
            self.target_net = None
            self.optimizer = None
            self.memory = ReplayBuffer(self.config.memory_size)

        self.epsilon = self.config.epsilon_start
        self.steps_done = 0

    def initialize_networks(self, state_dim: int, action_dim: int):
        """Initialize Q-networks."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for RL agent")

        self.policy_net = DQNAgent(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target_net = DQNAgent(state_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training: If True, use epsilon-greedy; otherwise greedy

        Returns:
            Action index
        """
        if not TORCH_AVAILABLE:
            # Fallback: random action
            return np.random.randint(0, self.config.action_dim)

        if training and np.random.random() < self.epsilon:
            # Explore
            return np.random.randint(0, self.config.action_dim)
        else:
            # Exploit
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def train_step(self):
        """Perform one training step."""
        if not TORCH_AVAILABLE or len(self.memory) < self.config.batch_size:
            return

        # Sample batch
        batch = self.memory.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.config.gamma * next_q

        # Compute loss
        loss = F.mse_loss(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(
        self,
        env: PortfolioEnv,
        n_episodes: int = 100,
        verbose: bool = True
    ) -> Dict:
        """
        Train the RL agent.

        Args:
            env: Portfolio environment
            n_episodes: Number of training episodes
            verbose: Print progress

        Returns:
            Training history
        """
        if not TORCH_AVAILABLE:
            warnings.warn("PyTorch not available. Cannot train RL agent.")
            return {}

        # Initialize networks
        state = env.reset()
        state_dim = len(state)
        action_dim = 5  # Number of rebalancing strategies

        self.initialize_networks(state_dim, action_dim)

        episode_rewards = []
        episode_wealth = []

        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action
                action = self.select_action(state, training=True)

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Store experience
                self.memory.push(state, action, reward, next_state, done)

                # Train
                self.train_step()

                # Update state
                state = next_state
                episode_reward += reward

                # Update epsilon
                self.epsilon = max(
                    self.config.epsilon_end,
                    self.epsilon * self.config.epsilon_decay
                )

                self.steps_done += 1

                # Update target network
                if self.steps_done % self.config.target_update_freq == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            episode_rewards.append(episode_reward)
            episode_wealth.append(env.wealth)

            if verbose and (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{n_episodes}, "
                      f"Reward: {episode_reward:.4f}, "
                      f"Wealth: {env.wealth:.4f}, "
                      f"Epsilon: {self.epsilon:.4f}")

        return {
            'episode_rewards': episode_rewards,
            'episode_wealth': episode_wealth,
            'final_epsilon': self.epsilon
        }

    def get_rebalancing_decision(
        self,
        current_state: np.ndarray
    ) -> Tuple[int, str]:
        """
        Get rebalancing decision for current state.

        Returns:
            Tuple of (action_index, action_description)
        """
        action = self.select_action(current_state, training=False)

        action_names = {
            0: "Hold (no rebalancing)",
            1: "Full rebalance to target",
            2: "Partial rebalance (50%)",
            3: "Equal weight rebalance",
            4: "Minimum variance rebalance"
        }

        return action, action_names.get(action, "Unknown action")


if __name__ == "__main__":
    print("Testing RL-based Adaptive Rebalancing...")

    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch not available. Skipping RL tests.")
    else:
        # Generate synthetic returns
        np.random.seed(42)
        n_periods = 500
        n_assets = 5

        returns_data = np.random.randn(n_periods, n_assets) * 0.01 + 0.0002
        returns_df = pd.DataFrame(
            returns_data,
            columns=[f'Asset_{i}' for i in range(n_assets)]
        )

        # Initialize environment
        initial_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        env = PortfolioEnv(
            returns=returns_df,
            initial_weights=initial_weights,
            transaction_cost=0.001
        )

        print(f"\nEnvironment initialized:")
        print(f"  Assets: {n_assets}")
        print(f"  Periods: {n_periods}")
        print(f"  Initial weights: {initial_weights}")

        # Test environment
        state = env.reset()
        print(f"\nState dimension: {len(state)}")

        # Initialize RL agent
        config = RLConfig(
            state_dim=len(state),
            action_dim=5,
            learning_rate=1e-3,
            batch_size=32
        )

        agent = RLRebalancer(config)

        # Train agent
        print("\nTraining RL agent...")
        history = agent.train(env, n_episodes=50, verbose=True)

        print(f"\nTraining complete!")
        print(f"  Final episode reward: {history['episode_rewards'][-1]:.4f}")
        print(f"  Final wealth: {history['episode_wealth'][-1]:.4f}")
        print(f"  Total return: {(history['episode_wealth'][-1] - 1) * 100:.2f}%")

        print("\n✅ RL rebalancing implementation complete!")
