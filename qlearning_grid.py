"""
Practical 3 - Exercici 1: Q-learning sobre el grid 4x3 (columnes x files).
Estat = (fila, columna), fila 0 = fila 1 de l'enunciat (a baix), columna 0 = columna 1.
Start = (0,0) ; Goal = (2,3).
Accions: 0=amunt, 1=avall, 2=esquerra, 3=dreta. Si l'acció surt del grid, l'agent es queda quiet.
Autores: Lidia i Maria
"""
import numpy as np

N_ROWS, N_COLS = 3, 4
START, GOAL = (0, 0), (2, 3)
ACTIONS = [(1, 0), (-1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["amunt", "avall", "esq", "dreta"]


# ---REWARDS---
#-1 everywhere, 100 at goal.
def reward_a():
    R = -np.ones((N_ROWS, N_COLS))
    R[GOAL] = 100
    return R

# more accurate reward function, 
# - (Manhattan Distance to goal), 100 at goal.
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
    dr, dc = ACTIONS[action]
    nr, nc = state[0] + dr, state[1] + dc
    if not (0 <= nr < N_ROWS and 0 <= nc < N_COLS): #outside the grid -> returns the same state.
        return state
    return (nr, nc)

#adds stochasticity to the action. For the drunken sailor.
def step(state, action, R, rng, p_intended=1.0): #p=1.0 means deterministic, p<1.0 means stochastic.
    if rng.random() > p_intended: #when p < 1.0
        action = rng.choice([a for a in range(4) if a != action]) #the action is replaced by one of the other 3 actions at random 
    ns = move(state, action) #do the movement
    return ns, R[ns], ns == GOAL


# ----Q LEARNING-----
def optimal_Q(R, gamma, p_intended=1.0, iters=300):
    Q = np.zeros((N_ROWS, N_COLS, 4))
    for _ in range(iters):
        Qn = np.zeros_like(Q)
        for r in range(N_ROWS):
            for c in range(N_COLS):
                s = (r, c)
                if s == GOAL:
                    continue
                for a in range(4):
                    val = 0.0
                    probs = {a: p_intended}
                    for b in range(4):
                        if b != a:
                            probs[b] = (1 - p_intended) / 3
                    for b, p in probs.items():
                        ns = move(s, b)
                        val += p * (R[ns] + (0 if ns == GOAL else gamma * Q[ns].max()))
                    Qn[r, c, a] = val
        Q = Qn
    return Q


# ---------- Q-learning ----------
def q_learning(R, alpha=0.1, gamma=0.9, epsilon=0.2, p_intended=1.0,
               max_episodes=5000, max_steps=500, stable_window=100, seed=0,
               fixed_episodes=None, err_tol=2.0):
    """Convergència = durant `stable_window` episodis consecutius es compleix que:
      (1) el camí greedy des de START arriba al goal amb la mateixa longitud que el camí
          òptim segons Q* (value iteration), i
      (2) |max_a Q(START,a) - V*(START)| < err_tol.
    Si fixed_episodes != None, no s'atura abans (útil per imprimir Q-tables)."""
    rng = np.random.default_rng(seed)
    Qstar = optimal_Q(R, gamma, p_intended)
    opt_len = len(greedy_path(Qstar, R)[1])
    Vstar = Qstar[START].max()
    Q = np.zeros((N_ROWS, N_COLS, 4))
    history, steps_per_ep = [], []
    last_policy, stable, conv_ep = None, 0, None
    n_ep = fixed_episodes or max_episodes

    for ep in range(1, n_ep + 1):
        s = START
        for t in range(max_steps):
            a = rng.integers(4) if rng.random() < epsilon else int(np.argmax(Q[s]))
            ns, r, done = step(s, a, R, rng, p_intended)
            target = r if done else r + gamma * Q[ns].max()
            Q[s][a] += alpha * (target - Q[s][a])
            s = ns
            if done:
                break
        steps_per_ep.append(t + 1)
        history.append(Q.copy())

        path_ok = (greedy_path(Q, R)[0][-1] == GOAL and len(greedy_path(Q, R)[1]) == opt_len)
        v_ok = abs(Q[START].max() - Vstar) < err_tol
        stable = stable + 1 if (path_ok and v_ok) else 0
        if stable >= stable_window and conv_ep is None:
            conv_ep = ep - stable_window
            if fixed_episodes is None:
                break
    return Q, history, steps_per_ep, conv_ep


def greedy_path(Q, R, max_steps=50):
    rng = np.random.default_rng(0)
    s, path, acts = START, [START], []
    for _ in range(max_steps):
        a = int(np.argmax(Q[s]))
        s, _, done = step(s, a, R, rng, 1.0)
        acts.append(ACTION_NAMES[a]); path.append(s)
        if done:
            break
    return path, acts


def print_q(Q, title):
    print(f"\n--- {title} ---")
    print("Estat (fila,col) | " + " | ".join(f"{n:>8}" for n in ACTION_NAMES))
    for r in range(N_ROWS - 1, -1, -1):
        for c in range(N_COLS):
            print(f"      ({r+1},{c+1})     | " + " | ".join(f"{v:8.2f}" for v in Q[r, c]))


def run_experiment(name, R, **kw):
    print("=" * 70); print(name, kw); print("=" * 70)
    Q, hist, steps, conv = q_learning(R, **kw)
    n = len(hist)
    for label, idx in [("Q-table 1 (episodi 1)", 0),
                       (f"Q-table intermèdia 1 (episodi {max(1, n//3)})", max(1, n // 3) - 1),
                       (f"Q-table intermèdia 2 (episodi {max(1, 2*n//3)})", max(1, 2 * n // 3) - 1),
                       (f"Q-table final (episodi {n})", n - 1)]:
        print_q(hist[idx], label)
    Qs = optimal_Q(R, kw["gamma"], kw.get("p_intended", 1.0))
    mask = np.ones((N_ROWS, N_COLS), bool); mask[GOAL] = False
    err = np.abs(Q - Qs)[mask].max()
    path, acts = greedy_path(Q, R)
    print("\nConvergència (camí òptim + V(start)≈V*) des de l'episodi:", conv, "| episodis totals:", n)
    print(f"max|Q - Q*| (estats no terminals) = {err:.2f}")
    print("Camí greedy:", path)
    print("Accions:", acts)
    return Q, conv


def multi_seed(R, label, seeds=20, **kw):
    convs = [q_learning(R, seed=s, **kw)[3] for s in range(seeds)]
    ok = [c for c in convs if c is not None]
    if not ok:
        print(f"[{label}] cap seed ha convergit"); return
    print(f"[{label}] episodis fins convergir ({len(ok)}/{seeds} seeds): "
          f"mitjana={np.mean(ok):.0f}, mediana={np.median(ok):.0f}, min={min(ok)}, max={max(ok)}")


if __name__ == "__main__":
    P = dict(alpha=0.1, gamma=0.9, epsilon=0.2)

    run_experiment("1a) Reward -1 / 100 (determinista)", reward_a(), **P)
    run_experiment("1b) Reward distància (determinista)", reward_b(), **P)
    run_experiment("1c) Borratxo, reward -1/100", reward_a(), p_intended=0.99, **P)
    run_experiment("1c) Borratxo, reward distància", reward_b(), p_intended=0.99, **P)

    print("\n" + "=" * 70 + "\nResum de convergència (20 seeds)\n" + "=" * 70)
    multi_seed(reward_a(), "1a det.", **P)
    multi_seed(reward_b(), "1b det.", **P)
    multi_seed(reward_a(), "1c estoc. -1/100", p_intended=0.99, **P)
    multi_seed(reward_b(), "1c estoc. distància", p_intended=0.99, **P)
