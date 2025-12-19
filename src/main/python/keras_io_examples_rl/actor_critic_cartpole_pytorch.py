"""
PyTorch port of actor_critic_cartpole_pytorch.py from keras-io
(@see: https://github.com/keras-team/keras-io/blob/master/examples/rl/actor_critic_cartpole_pytorch.py)

Has no TensorFlow dependence:
- Uses torch.autograd.grad 

### References

- [Environment documentation](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [CartPole paper](http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf)
- [Actor Critic Method](https://hal.inria.fr/hal-00840470/document)
"""

from keras import layers
from keras import ops
import keras
import numpy as np
import gymnasium as gym
import os
import torch
os.nice(10)  # Be nice!
os.environ["KERAS_BACKEND"] = "torch"

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
# Adding `render_mode='human'` will show the attempts of the agent
env = gym.make("CartPole-v0")  # Create the environment
env.reset(seed=seed)
# Smallest number such that 1.0 + eps != 1.0
eps = np.finfo(np.float32).eps.item()

"""
## Implement Actor Critic network

This network learns two functions:

1. Actor: This takes as input the state of our environment and returns a
probability value for each action in its action space.
2. Critic: This takes as input the state of our environment and returns
an estimate of total rewards in the future.

In our implementation, they share the initial layer.
"""

num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])

"""
## Train
"""

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:  # Run until solved
    state = env.reset()[0]
    episode_reward = 0
    for timestep in range(1, max_steps_per_episode):

        state = ops.convert_to_tensor(state)
        state = ops.expand_dims(state, 0)

        # Predict action probabilities and estimated future rewards
        # from environment state
        action_probs, critic_value = model(state)
        critic_value_history.append(critic_value[0, 0])

        # Sample action from action probability distribution
        action = torch.multinomial(action_probs, 1, replacement=True).item()
        action_probs_history.append(ops.log(action_probs[0, action]))

        # Apply the sampled action in our environment
        state, reward, done, *_ = env.step(action)
        rewards_history.append(reward)
        episode_reward += reward

        if done:
            break

    # Update running reward to check condition for solving
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    # Calculate expected value from rewards
    # - At each timestep what was the total reward received after that timestep
    # - Rewards in the past are discounted by multiplying them with gamma
    # - These are the labels for our critic
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
    returns = returns.tolist()

    # Calculating loss values to update our network
    history = zip(action_probs_history, critic_value_history, returns)
    actor_losses = []
    critic_losses = []
    for log_prob, value, ret in history:
        # At this point in history, the critic estimated that we would get a
        # total reward = `value` in the future. We took an action with log probability
        # of `log_prob` and ended up receiving a total reward = `ret`.
        # The actor must be updated so that it predicts an action that leads to
        # high rewards (compared to critic's estimate) with high probability.
        diff = ret - value
        actor_losses.append(-log_prob * diff)  # actor loss

        # The critic must be updated so that it predicts a better estimate of
        # the future rewards.
        critic_losses.append(
            huber_loss(ops.expand_dims(value, 0), ops.expand_dims(ret, 0))
        )

    # Backpropagation
    optimizer.zero_grad()  # Clear previous gradients
    loss_value = sum(actor_losses) + sum(critic_losses)
    loss_value.backward()
    optimizer.step()

    # Clear the loss and reward history
    action_probs_history.clear()
    critic_value_history.clear()
    rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break

