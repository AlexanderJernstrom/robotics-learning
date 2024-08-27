import numpy as np
import random
import matplotlib.pyplot as plt
import math

epsilon = 0.1
bandit_probs = np.random.uniform(0.1, 0.9, 10)
print(bandit_probs)

def pullArm(i):
    r = random.random()
    if r <= bandit_probs[i]:
        return 1
    else:
        return 0

def actionEstimation(n, rewards):

    # Estimate reward from action based sample-average method
    estimated_value = sum(rewards)
    if n != 0:
        estimated_value = estimated_value / n
    return estimated_value 

def ucbEstimation(n, rewards, c, timestep):
    q = actionEstimation(n, rewards=rewards)
    uncertainty = c * math.sqrt(math.log(timestep) / n)
    return q + uncertainty


def actionSelection(state):
    action_values = []
    for i in range(10):
        val = actionEstimation(state[f'{i}']["n_chosen"], state[f'{i}']["rewards"]) 
        action_values.append(val)
    best_action = action_values.index(max(action_values))
    return best_action

def epsilonGreedySelection(state):
    best_action = random.randint(0, 9)
    r = random.random()
    if r <= epsilon:
        return random.randint(0, 9)
    for i in range(10):
        val = actionEstimation(state[f'{i}']["n_chosen"], state[f'{i}']["rewards"]) 
        best_action_val = actionEstimation(state[f'{best_action}']["n_chosen"], state[f'{best_action}']["rewards"])
        if val > best_action_val:
            best_action = i
    return best_action


def game_loop(timestep, state):
    action = actionSelection(state)
    reward = pullArm(action)
    state[f'{action}']["n_chosen"] =  state[f'{action}']["n_chosen"] + 1
    state[f'{action}']["rewards"].append(reward)
    timestep = timestep + 1

def game_loop_epsilon(timestep, state):
    action = epsilonGreedySelection(state)
    reward = pullArm(action)
    state[f'{action}']["n_chosen"] =  state[f'{action}']["n_chosen"] + 1
    state[f'{action}']["rewards"].append(reward)
    timestep = timestep + 1
    
def calculate_total_rewards(state):
    total_rewards = 0
    for key, value in state.items():
        total_rewards += sum(value['rewards'])
    return total_rewards


greedy_rewards = []
epsilon_rewards = []
epsilon_state = {
    "0": {
        "n_chosen": 0,
        "rewards": []
    },
    "1": {
        "n_chosen": 0,
        "rewards": []
    },
    "2": {
        "n_chosen": 0,
        "rewards": []
    },
    "3": {
        "n_chosen": 0,
        "rewards": []
    },
    "4": {
        "n_chosen": 0,
        "rewards": []
    },
    "5": {
        "n_chosen": 0,
        "rewards": []
    },
    "6": {
        "n_chosen": 0,
        "rewards": []
    },
    "7": {
        "n_chosen": 0,
        "rewards": []
    },
    "8": {
        "n_chosen": 0,
        "rewards": []
    },
    "9": {
        "n_chosen": 0,
        "rewards": []
    },
}
greedy_state = {
    "0": {
        "n_chosen": 0,
        "rewards": []
    },
    "1": {
        "n_chosen": 0,
        "rewards": []
    },
    "2": {
        "n_chosen": 0,
        "rewards": []
    },
    "3": {
        "n_chosen": 0,
        "rewards": []
    },
    "4": {
        "n_chosen": 0,
        "rewards": []
    },
    "5": {
        "n_chosen": 0,
        "rewards": []
    },
    "6": {
        "n_chosen": 0,
        "rewards": []
    },
    "7": {
        "n_chosen": 0,
        "rewards": []
    },
    "8": {
        "n_chosen": 0,
        "rewards": []
    },
    "9": {
        "n_chosen": 0,
        "rewards": []
    },
}
timestep = 0
iters = 5000 
for i in range(iters):
    game_loop_epsilon(timestep=timestep, state=epsilon_state)
    game_loop(timestep=timestep, state=greedy_state)

    avg_reward_epsilon = calculate_total_rewards(state=epsilon_state)
    avg_reward_greedy =calculate_total_rewards(state=greedy_state) 

    if i != 0:
        avg_reward_epsilon = avg_reward_epsilon/ i

        avg_reward_greedy = avg_reward_greedy/ i
    epsilon_rewards.append(avg_reward_epsilon)
    greedy_rewards.append(avg_reward_greedy)

x_greedy = range(iters)
y_greedy = [greedy_rewards[i] for i in range(iters)]

x_epsilon = range(iters)
y_epsilon = [epsilon_rewards[i] for i in range(iters)]


plt.plot(x_greedy, y_greedy, color="blue")

plt.plot(x_epsilon, y_epsilon, color="red")
plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title('Average Reward vs Iteration')
plt.show()



