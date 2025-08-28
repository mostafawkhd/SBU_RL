import numpy as np
import matplotlib.pyplot as plt
import itertools
from utils import bandit

SEED = 50
np.random.seed(SEED)

def softmax_preferences(pref: np.ndarray) -> np.ndarray:
    exps = np.exp(pref)
    return exps / exps.sum()

def adjust_preferences(pref: np.ndarray, 
                       policy: np.ndarray, 
                       alpha: float, 
                       action: int, 
                       reward: float, 
                       avg_reward: float) -> np.ndarray:
    one_hot = np.zeros_like(pref, dtype=np.float32)
    one_hot[action] = 1.0
    pref += alpha * (reward - avg_reward) * (one_hot - policy)
    return pref

def run_policy_gradient(K: int, 
                        q_star: np.ndarray, 
                        rewards: np.ndarray, 
                        optimal_ratio: np.ndarray,
                        alpha: float,
                        use_baseline: bool,
                        steps: int = 1000) -> None:
    
    pref = np.zeros(K, dtype=np.float32)
    policy = np.ones(K, dtype=np.float32) / K
    cumulative_reward = 0
    cumulative_optimal = 0
    
    for t in range(steps):
        action = np.random.choice(K, p=policy)
        reward, is_optimal = bandit(q_star, action)
        
        avg_reward = cumulative_reward / t if use_baseline and t > 0 else reward

        pref = adjust_preferences(pref, policy, alpha, action, reward, avg_reward)
        policy = softmax_preferences(pref)

        cumulative_reward += reward
        cumulative_optimal += is_optimal
        rewards[t] = reward
        optimal_ratio[t] = cumulative_optimal / (t + 1)

if __name__ == "__main__":
    K = 10
    alphas = [0.1, 0.4]
    baselines = [False, True]
    hyper_params = list(itertools.product(baselines, alphas))

    steps = 1000
    rounds = 2000

    rewards = np.zeros((len(hyper_params), rounds, steps))
    optimal_ratio = np.zeros((len(hyper_params), rounds, steps))
    q_star = np.random.normal(4.0, 1.0, size=K)

    for idx, (baseline_flag, alpha) in enumerate(hyper_params):
        for rnd in range(rounds):
            run_policy_gradient(K, q_star, rewards[idx, rnd], optimal_ratio[idx, rnd],
                                alpha, baseline_flag, steps)

    avg_optimal_ratio = optimal_ratio.mean(axis=1)

    for ratio in avg_optimal_ratio:
        plt.plot(ratio)
    plt.xlabel("Steps")
    plt.ylabel("Average Optimal Action %")
    plt.show()
