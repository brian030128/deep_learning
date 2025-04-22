import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Conv2d
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import matplotlib.pyplot as plt
gym.register_envs(ale_py)
torch.set_printoptions(linewidth=1000)
torch.set_printoptions(threshold=float('inf'))
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.size = 0  # Track number of added elements

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, s):
        idx = 0
        while idx < self.capacity - 1:
            left = 2 * idx + 1
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def total(self):
        return self.tree[0]

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.8, beta=0.4, beta_increment_per_sampling=1e-6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = 1e-5
        self.max_priority = 1.0

    def add(self, transition, error=None):
        # Use max_priority if error not provided
        priority = self.max_priority if error is None else (abs(error) + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)

    def __len__(self):
        return len(self.tree)

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        
        # Add this code to log priority distribution
        if len(self.tree) > 1000:  # Only log when buffer has sufficient entries
            all_priorities = []
            for i in range(min(self.tree.size, 1000)):  # Sample 1000 priorities
                idx = self.tree.capacity - 1 + i
                if idx < len(self.tree):
                    all_priorities.append(self.tree[idx])
            
            if len(all_priorities) > 0:
                wandb.log({
                    "Priority_Mean": np.mean(all_priorities),
                    "Priority_Std": np.std(all_priorities),
                    "Priority_Max": np.max(all_priorities),
                    "Priority_Min": np.min(all_priorities)
                })

        total_priority = self.tree.total()
        segment = total_priority / batch_size

        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            idx, prio, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(prio)

        sampling_probabilities = np.array(priorities) / total_priority
        weights = (self.tree.size * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        total_priority = self.tree.total()
        segment = total_priority / batch_size
        
        # Rest of your sample code...
        
        # At the end of sample method, add this to track sample uniqueness
        # Count how many unique samples we have in the batch
        unique_states = len(set([hash(state.cpu().numpy().tobytes()) for state in states]))
        wandb.log({
            "Unique_Samples_Ratio": unique_states / batch_size,
            "Beta_Value": self.beta
        })
        
        return (
            torch.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            torch.stack(next_states),
            np.array(dones, dtype=np.uint8),
            idxs,
            weights.astype(np.float32)
        )


    def update_priorities(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)


class NStepTransitionBuffer:
    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        self.buffer = deque()

    def append(self, transition):
        self.buffer.append(transition)

    def is_ready(self):
        return len(self.buffer) >= self.n

    def get(self):
        R = 0
        for i, (_, _, r, _, d) in enumerate(self.buffer):
            R += (self.gamma ** i) * r
            if d:  # 提早結束
                break

        state, action, _, _, _ = self.buffer[0]
        _, _, _, next_state, done = self.buffer[-1]
        return state, action, R, next_state, done

    def pop(self):
        self.buffer.popleft()

    def reset(self):
        self.buffer.clear()

class DQN(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims=512):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self._conv = nn.Sequential(
            self.conv1, nn.ReLU(),
            self.conv2, nn.ReLU(),
            self.conv3, nn.ReLU()
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_dims)
            conv_out_size = self._conv(dummy_input).view(1, -1).size(1)

        # Dueling: separate streams
        self.fc_value = nn.Sequential(
            nn.Linear(conv_out_size, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, 1)  # Value stream
        )

        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_out_size, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, n_actions)  # Advantage stream
        )

    def forward(self, x):
        x = self._conv(x)
        x = x.view(x.size(0), -1)

        value = self.fc_value(x)
        advantage = self.fc_advantage(x)

        # Combine V(s) and A(s, a)
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals


class AtariPreprocessor:
    """
    Preprocess Atari observations for DQN:
    - Converts to grayscale
    - Resizes to 84x84
    - Normalizes to [0.0, 1.0]
    - Maintains a stack of past frames
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        # Convert to grayscale
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        # Normalize to [0, 1] and convert to float32
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return torch.from_numpy(np.stack(self.frames, axis=0)).to(device)  # shape: (frame_stack, 84, 84)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return torch.from_numpy(np.stack(self.frames, axis=0)).to(device=device)  # shape: (frame_stack, 84, 84)


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):

        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        
        self.env.action_space.seed(args.seed)
        self.env.observation_space.seed(args.seed)
        self.test_env.action_space.seed(args.seed)
        self.test_env.observation_space.seed(args.seed)

        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        obs, _ = self.env.reset()
        state = self.preprocessor.reset(obs)
        obs_space_dim = state.shape
        
        self.q_net = DQN(obs_space_dim, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(obs_space_dim, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.memory = PrioritizedReplayBuffer(args.memory_size)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        
        self.debug_dir = "./debug_image"
        self.n_step = args.n_step
        self.n_step_buffer = NStepTransitionBuffer(self.n_step, self.gamma)

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def select_action(self, state):
        if random.random() < self.epsilon:
            # Get ball and paddle positions from the state
            frame = state.cpu().numpy()[-1]  # Get the most recent frame
            game_info = self.extract_game_info(frame)
            
            if game_info is not None and game_info["ball_y"] is not None and game_info["paddle_y"] is not None: 
                ball_y = game_info['ball_y']
                paddle_y = game_info['paddle_y']
                #print(ball_y, paddle_y)
                
                # Determine if we should move up or down
                if ball_y < paddle_y:  # Ball is above paddle (with some threshold)
                    return 2  # Move down (assuming action 2 moves paddle up in Pong)
                elif ball_y > paddle_y:  # Ball is below paddle
                    return 3  # Move up (assuming action 3 moves paddle down in Pong)
                else:
                    return 0  # No movement needed, ball is aligned with paddle
            else:
                # Fall back to random if we couldn't extract game info
                return random.randint(0, self.num_actions - 1)
        else:
            # Use the network for exploitation
            state_tensor = state.unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return q_values.argmax().item()
        
    def extract_game_info(self, frame):
        import matplotlib.pyplot as plt
        import os
        import numpy as np
        from datetime import datetime
        
        frame = frame.copy()
        frame = torch.from_numpy(frame) * 10
        frame = frame.int()
        
        # Remove score area
        frame[:14, :] = 3
        frame[-7:, :] = 3
        mask = frame != 3

        # Use the mask to set all non-3 elements to 9
        frame[mask] = 9

        subset = frame[:, 70:]
        mask = (subset == 9)  # Mask for elements equal to 9
        # Apply the mask only to the slice [:,70:]
        indices = torch.nonzero(mask, as_tuple=True)
        paddle_y = None
        if len(indices[0]) != 0:
            paddle_y = torch.mean(indices[0].to(torch.float)).item()

        subset = frame[:, 15:71]
        mask = (subset == 9)  # Mask for elements equal to 9
        indices = torch.nonzero(mask, as_tuple=True)
        ball_y = None
        if len(indices[0]) != 0:
            ball_y = torch.mean(indices[0].to(torch.float)).item()

        if ball_y is not None and paddle_y is not None:
            frame[int(ball_y)][40] = 5
            frame[int(paddle_y)][40] = 5
        # Create debug visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(frame)
        ax.set_title('Game Frame Analysis')
        

        debug_path = os.path.join(os.path.curdir, self.debug_dir)
        os.makedirs(self.debug_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_path = os.path.join(debug_path, f"frame_analysis_{timestamp}.png")
        plt.savefig(debug_path)
        plt.close(fig)
        
        #print(ball_y, paddle_y)
        return {
            "ball_y": ball_y,
            "paddle_y": paddle_y    
        } if ball_y is not None and paddle_y is not None else None
    
    def run(self, episodes=2000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                next_state = self.preprocessor.step(next_obs)
                self.n_step_buffer.append((state, action, reward, next_state, done))

                if self.n_step_buffer.is_ready():
                    state, action, R, next_state, done = self.n_step_buffer.get()

                    state_tensor = state.unsqueeze(0).to(self.device)
                    next_state_tensor = next_state.unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        next_action = self.q_net(next_state_tensor).argmax(1, keepdim=True)
                        next_q = self.target_net(next_state_tensor).gather(1, next_action).squeeze().item()

                    target = R + (self.gamma ** self.n_step) * next_q * (0 if done else 1)
                    with torch.no_grad():
                        current_q = self.q_net(state_tensor).gather(1, torch.tensor([[action]], device=self.device)).squeeze().item()

                    td_error = abs(target - current_q)

                    self.memory.add((state, action, R, next_state, done), error=td_error)
                    self.n_step_buffer.pop()

                if done:
                    self.n_step_buffer.reset()

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                if self.env_count % 10000 == 0:
                    self.visualize_priority_distribution(self.env_count)
                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                if self.env_count % 50000 == 0:
                    model_path = os.path.join(self.save_dir, f"model_step_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved model checkpoint to {model_path}")
 
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            if ep % 10 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })
    def visualize_priority_distribution(self, step):
        if len(self.memory.tree) < 1000:
            return
            
        # Sample 1000 priorities from the tree
        priorities = []
        for i in range(min(self.memory.tree.size, 1000)):
            idx = self.memory.tree.capacity - 1 + i
            if idx < len(self.memory.tree.tree):
                priorities.append(self.memory.tree.tree[idx])
        
        if len(priorities) > 0:
            # Log histogram data for wandb
            wandb.log({
                "Priority_Histogram": wandb.Histogram(priorities),
                "Steps": step
            })
    def evaluate(self):
        glob_reward = 0
        
        runs = 10
        for i in range(runs):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            while not done:
                
                state_tensor = state.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = self.preprocessor.step(next_obs)
            
            glob_reward += total_reward

        return glob_reward / runs


    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1).to(self.device)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            q_targets = rewards + (self.gamma ** self.n_step) * next_q_values * (1 - dones)

        q_values = self.q_net(states).gather(1, actions)
        bellman_errors = q_values - q_targets
        loss = (weights * (bellman_errors ** 2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10)
        self.optimizer.step()

       # 更新 priority
        errors = bellman_errors.detach().cpu().numpy().squeeze()
        self.memory.update_priorities(indices, errors)

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f}")
            wandb.log({
                "Train Loss": loss.item(),
               "Train Step": self.train_count
            })
        if self.train_count % 100 == 0:
            wandb.log({
                "TD_Error_Mean": np.mean(np.abs(errors)),
                "TD_Error_Std": np.std(np.abs(errors)),
                "TD_Error_Max": np.max(np.abs(errors)),
                "TD_Error_Min": np.min(np.abs(errors)),
                "Weight_Mean": weights.mean().item(),
                "Weight_Std": weights.std().item()
            })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results/pong2/")
    parser.add_argument("--wandb-run-name", type=str, default="pong-run")
    parser.add_argument("--batch-size", type=int, default=150)
    parser.add_argument("--memory-size", type=int, default=70000)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999998)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=10000)
    parser.add_argument("--max-episode-steps", type=int, default=1000000)
    parser.add_argument("--train-per-step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=313551076, help="Random seed for evaluation")
    parser.add_argument("--n-step", type=int, default=8, help="Steps for multi-step return")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    wandb.init(project="DLP-Lab5-DQN-Pong", name=args.wandb_run_name, save_code=True)

    agent = DQNAgent(args=args)
    agent.run()
