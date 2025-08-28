import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def simulate_random_walk(step_count=100, start_position=500):
    pos = start_position
    trajectory = [pos]
    for _ in range(step_count):
        move = np.random.choice([-1, 1])
        pos += move
        trajectory.append(pos)
        if pos == 0 or pos == 1000:
            break
    return trajectory

def run_experiment(total_episodes=200, steps_per_episode=1000):
    averaged_values = np.zeros(1001)
    for _ in range(total_episodes):
        path = simulate_random_walk(steps_per_episode)
        for state in path:
            averaged_values[state] += 1
    return averaged_values / total_episodes

def main():
    runs = 20
    final_values = np.zeros(1001)

    for _ in range(runs):
        results = run_experiment()
        final_values += results

    final_values /= runs

    plt.plot(final_values)
    plt.title("Random Walk State Distribution")
    plt.xlabel("State")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
