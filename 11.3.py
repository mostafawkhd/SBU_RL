import numpy as np

# Parameters
alpha = 0.1
gamma = 0.99
W = np.random.rand(9, 1)

# State encoding
state_features = {
    1:[2,0,0,0,0,0,0,1],
    2:[0,2,0,0,0,0,0,1],
    3:[0,0,2,0,0,0,0,1],
    4:[0,0,0,2,0,0,0,1],
    5:[0,0,0,0,2,0,0,1],
    6:[0,0,0,0,0,2,0,1],
    7:[0,0,0,0,0,0,1,2],
}

# Memoization for state-action features
feature_cache = {}

def get_features(state, action):
    key = (state, action)
    if key not in feature_cache:
        feat = np.array(state_features[state] + [action]).reshape(-1,1)
        feature_cache[key] = feat
    return feature_cache[key]

def Q_value(state, action):
    return W.T @ get_features(state, action)

def sample_action():
    if np.random.rand() > 1/7:
        return 0, np.random.randint(1, 7)
    else:
        return 1, 7

def semi_gradient_update(state, next_state, action):
    global W
    td_error = gamma * max(Q_value(next_state, 0), Q_value(next_state, 1)) - Q_value(state, action)
    W += alpha * td_error * get_features(state, action)

def run_simulation(episodes=30, max_steps=10000):
    global W
    for ep in range(1, episodes+1):
        W = np.random.rand(9,1)
        _, state = sample_action()
        for step in range(1, max_steps+1):
            action, next_state = sample_action()
            semi_gradient_update(state, next_state, action)
            state = next_state
            if W.max() >= 1e10:
                print(f"Episode {ep} diverged at step {step}")
                break

if __name__ == "__main__":
    run_simulation()
