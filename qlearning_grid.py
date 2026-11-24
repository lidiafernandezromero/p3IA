"""
Practical 3 - Exercici 1: Q-learning sobre el grid 4x3 (columnes x files).
Estat = (fila, columna), fila 0 = fila 1 de l'enunciat (a baix), columna 0 = columna 1.
Start = (0,0) ; Goal = (2,3).
Accions: 0=amunt, 1=avall, 2=esquerra, 3=dreta. Si l'acció surt del grid, l'agent es queda quiet.
Autores: Lidia i Maria
"""
import random
import numpy as np

N_ROWS, N_COLS = 3, 4
START, GOAL, BLOCKED = (0, 0), (2, 3), (1, 1)
ACTIONS = [(1, 0), (-1, 0), (0, -1), (0, 1)]
NAMES = ["up", "down", "left", "right"]


# ---REWARDS---
#-1 everywhere, 100 at goal.
def reward_a():
    R = -np.ones((N_ROWS, N_COLS))
    R[GOAL] = 100
    return R

# more accurate reward function, 
# - (Manhattan Distance to the goal), 100 at the goal.
def reward_b():
    R = np.zeros((N_ROWS, N_COLS))
    for r in range(N_ROWS):
        for c in range(N_COLS):
            R[r, c] = -(abs(GOAL[0] - r) + abs(GOAL[1] - c))
    R[GOAL] = 100
    return R

#---MOVEMENT---

# applies the movement. 
def move(state, action):
    nr = state[0] + ACTIONS[action][0]
    nc = state[1] + ACTIONS[action][1]
    #outside the grid or on blocked cell -> returns the same state.
    if nr < 0 or nr >= N_ROWS or nc < 0 or nc >= N_COLS or (nr, nc) == BLOCKED:
        return state
    return (nr, nc)


#adds stochasticity to the action. For the drunken sailor.
# With probability 1-p_intended the action is replaced by one of the other 3 (drunken sailor).
# p_intended=1 -> deterministic.

def step(state, action, R, p_intended=1.0): #p=1.0 means deterministic, p<1.0 means stochastic.
    if random.random() > p_intended: #when p < 1.0
        others = [a for a in range(4) if a != action]
        action = random.choice(others)
        
    next_state = move(state, action)#do the movement
    return next_state, R[next_state], next_state == GOAL



# Path followed by the greedy policy (always the best action of the Q-table)
def greedy_path(Q, max_steps=50):
    s = START
    path, acts = [s], []

    #iterate until we reach the goal or max_steps 
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        s = move(s, a)
        path.append(s)
        acts.append(NAMES[a])
        if s == GOAL:
            break
    return path, acts


# ----Q LEARNING-----

def q_learning(R, alpha=0.1, gamma=0.9, epsilon=0.2, p_intended=1.0,
               max_episodes=5000, max_steps=500,  stable_needed=20, threshold=0.1):

    # Initialize 
    Q = np.zeros((N_ROWS, N_COLS, 4)) #q-table with 4 actions for each state (zeros at first)
    history = [] #history of Q-tables 

    # Convergence: the Q-table practically stops changing (the prediction
    # matches the real reward). 
    Q_old = Q.copy()
    stable = 0
    converged = False

    #iterate over episodes
    for ep in range(max_episodes):
        #start state
        s = START

        # Run one episode.
        for t in range(max_steps):
            # Choose an action using epsilon-greedy.
            if random.random() < epsilon:
                a = random.randrange(4)
            else:
                # choose the action with the highest Q-value for the current state
                a = int(np.argmax(Q[s]))

            # Take the action and see the next state and reward.
            ns, r, done = step(s, a, R,  p_intended)

            #Q-learning update
            if done:
                target = r  # if we reach the goal, the target is just the reward
            else:
                target = r + gamma * Q[ns].max()  # otherwise, it's the reward plus the discounted max Q-value of the next state

            Q[s][a] += alpha * (target - Q[s][a]) # update the Q-value for the current state and action

            # Move to the next state.
            s = ns

            # if we reach the goal, end the episode
            if done:
                break

        #save a copy of the Q-table 
        history.append(Q.copy())

        # CONVERGENCE CHECK
        # Check if the Q-table has stabilized by comparing it to the previous Q-table.
        change = np.abs(Q - Q_old).max()
        if change < threshold:
            stable += 1
        else:
            stable = 0
        Q_old = Q.copy()

        if stable >= stable_needed:
            converged = True
            break

    return Q, history, converged


#---OUTPUT---

# Prints the Q-table in the file 
def print_q(Q, title, file):
    print(f"\n--- {title} ---", file=file)
    print("State (row,col) | " + " | ".join(f"{n:>8}" for n in NAMES), file=file)

    for r in range(N_ROWS - 1, -1, -1):
        for c in range(N_COLS):
            print(f"      ({r+1},{c+1})     | " +
                  " | ".join(f"{v:8.2f}" for v in Q[r, c]), file=file)


# Runs the experiment and saves all results to a text file.
def run_experiment(name, R, file, **params):
    print("=" * 70, file=file)
    print(name, params, file=file)
    print("=" * 70, file=file)

    Q, history, converged = q_learning(R, **params)
    n = len(history)

    # Print the first, two intermediate and final Q-tables.
    for label, ep in [("First Q-table", 1),
                      ("Intermediate Q-table 1", max(1, n // 3)),
                      ("Intermediate Q-table 2", max(1, 2 * n // 3)),
                      ("Final Q-table", n)]:
        print_q(history[ep - 1], f"{label} (episode {ep})", file)

    # Print the greedy path and actions.
    path, acts = greedy_path(Q)

    print("\nLearned path:", path, file=file)
    print("Actions:", acts, file=file)
    print("Episodes run:", n, file=file)
    if converged:
        print("Converged at episode", n, file=file)
    else:
        print("Did NOT converge within the maximum number of episodes", file=file)


# ---MAIN----
if __name__ == "__main__":

    random.seed(0) 

    # Parameters used in the experiments.
    P = dict(alpha=0.1, gamma=0.9, epsilon=0.2)

    # Open the file where all results will be saved.
    with open("resultats_grid.txt", "w", encoding="utf-8") as file:

        # 1a) Deterministic environment with reward -1 / 100.
        run_experiment("1a) Reward -1 / 100 (deterministic)",
                       reward_a(), file, **P)

        # 1b) Deterministic environment with distance-based reward.
        run_experiment("1b) Distance reward (deterministic)",
                       reward_b(), file, **P)

        # 1c) Stochastic environment with reward -1 / 100.
        run_experiment("1c) Drunken sailor, reward -1 / 100",
                       reward_a(), file, p_intended=0.99, **P)

        # 1c) Stochastic environment with distance-based reward.
        run_experiment("1c) Drunken sailor, distance reward",
                       reward_b(), file, p_intended=0.99, **P)

    print("All results have been saved in resultats_grid.txt")


