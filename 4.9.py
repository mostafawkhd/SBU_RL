import numpy as np
import matplotlib.pyplot as plt

def train_random_walk(prob=0.4, threshold=1e-6):
    values = np.random.rand(100) * 1000
    values[0] = 0
    policy = [0]*100
    iteration = 1

    while True:
        delta = 0
        for state in range(1, 100):
            old_value = values[state]
            action_values = [0]*51
            for action in range(1, min(state, 100 - state) + 1):
                action_values[action] = 0
                if state + action < 100:
                    action_values[action] += prob * values[state + action]
                    action_values[action] += (1 - prob) * values[state - action]
                elif state + action == 100:
                    action_values[action] += prob
                    action_values[action] += (1 - prob) * values[state - action]
            best_action = np.argmax(action_values)
            policy[state] = best_action
            values[state] = action_values[best_action]
            delta = max(delta, abs(old_value - values[state]))

        iteration += 1
        if iteration % 1000 == 0:
            print(f"Iteration {iteration}, Delta = {delta}")
        if delta < threshold:
            break

    return values[1:100], policy[1:100]

if __name__ == "__main__":
    V1, pi1 = train_random_walk(prob=0.4)
    V2, pi2 = train_random_walk(prob=0.25)
    V3, pi3 = train_random_walk(prob=0.55)

    states = np.arange(1, 100)
    plt.figure()
    plt.plot(states, V1, label='p=0.4')
    plt.plot(states, V2, label='p=0.25')
    plt.plot(states, V3, label='p=0.55')
    plt.legend()
    plt.show()

    for V, pi, p in zip([V1,V2,V3],[pi1,pi2,pi3],[0.4,0.25,0.55]):
        plt.figure()
        plt.bar(states, pi)
        plt.title(f"Policy for p={p}")
        plt.show()
