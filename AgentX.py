import os, json, pickle, hashlib, random, time
from collections import defaultdict, deque
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ------------------- Utilities -------------------

def ensure_dirs():
    os.makedirs("qtables", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

def maze_to_hash(maze):
    flat = ",".join(str(cell) for row in maze for cell in row)
    return hashlib.sha256(flat.encode("utf-8")).hexdigest()[:12]

def save_qtable(qtable, maze_hash):
    path = os.path.join("qtables", f"q_{maze_hash}.pkl")
    data = {repr(k): v.tolist() for k,v in qtable.items()}
    with open(path, "wb") as f:
        pickle.dump(data, f)

def load_qtable(maze_hash):
    path = os.path.join("qtables", f"q_{maze_hash}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        raw = pickle.load(f)
    q = defaultdict(lambda: np.zeros(4))
    for k,v in raw.items():
        try:
            st = eval(k)
            q[st] = np.array(v)
        except Exception:
            continue
    return q

def save_history(history):
    with open("history.json","w") as f:
        json.dump(history, f, indent=2)

def load_history():
    if os.path.exists("history.json"):
        with open("history.json","r") as f:
            return json.load(f)
    return {"runs": []}

def save_figure(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

# ------------------- Maze helpers -------------------


def is_solvable(maze, start=(0,0)):
    rows, cols = len(maze), len(maze[0])
    goals = set()
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 2:
                goals.add((r,c))
    if not goals:
        return False
    sr, sc = start
    if not (0<=sr<rows and 0<=sc<cols) or maze[sr][sc] == 1:
        found = False
        for r in range(rows):
            for c in range(cols):
                if maze[r][c] == 0:
                    sr, sc = r, c
                    found = True
                    break
            if found: break
        if not found:
            return False
    q = deque(); q.append((sr,sc)); visited={ (sr,sc) }
    moves=[(-1,0),(1,0),(0,-1),(0,1)]
    while q:
        r,c = q.popleft()
        if (r,c) in goals:
            return True
        for dr,dc in moves:
            nr,nc = r+dr, c+dc
            if 0<=nr<rows and 0<=nc<cols and (nr,nc) not in visited and maze[nr][nc] != 1:
                visited.add((nr,nc)); q.append((nr,nc))
    return False

def generate_random_solvable_maze(m, n, wall_prob=0.25, max_tries=200):
    assert m>0 and n>0
    for attempt in range(max_tries):
        maze = []
        for i in range(m):
            row = []
            for j in range(n):
                row.append(1 if random.random() < wall_prob else 0)
            maze.append(row)
        maze[0][0] = 0
        while True:
            gr = random.randint(0,m-1); gc = random.randint(0,n-1)
            if not (gr==0 and gc==0): break
        maze[gr][gc] = 2
        if is_solvable(maze, start=(0,0)):
            return maze
    maze = [[0]*n for _ in range(m)]
    maze[0][0] = 0
    maze[m-1][n-1] = 2
    return maze

def find_start_and_goal(maze):
    rows,cols = len(maze), len(maze[0])
    start = None
    if maze[0][0] != 1:
        start = (0,0)
    else:
        for r in range(rows):
            for c in range(cols):
                if maze[r][c] == 0:
                    start = (r,c); break
            if start: break
    goal = None
    for r in range(rows):
        for c in range(cols):
            if maze[r][c] == 2:
                goal = (r,c); break
        if goal: break
    return start, goal

# ------------------- Q-Learning -------------------

def average_qtables_for_size(rows, cols):
    files = [f for f in os.listdir("qtables") if f.startswith("q_")]
    if not files:
        return None
    accum = {}; counts = {}
    for fname in files:
        path = os.path.join("qtables", fname)
        try:
            with open(path,"rb") as f:
                raw = pickle.load(f)
            for k,v in raw.items():
                try:
                    st = eval(k)
                except Exception:
                    continue
                if not isinstance(st, tuple) or len(st) != 2: continue
                r,c = st
                if r < rows and c < cols:
                    accum[st] = accum.get(st, np.zeros(len(v))) + np.array(v)
                    counts[st] = counts.get(st,0) + 1
        except Exception:
            continue
    if not accum: return None
    q = defaultdict(lambda: np.zeros(4))
    for st, tot in accum.items():
        q[st] = tot / counts[st]
    return q

def train_q_learning(maze, episodes=600, alpha=0.7, gamma=0.95, eps_start=1.0, eps_end=0.05,
                     max_steps_per_episode=1000, transfer_init=False, load_prev=False, maze_hash=None):
    start, goal = find_start_and_goal(maze)
    if start is None or goal is None:
        raise ValueError("Maze must have a start and a goal (value 2).")
    rows, cols = len(maze), len(maze[0])
    actions = [(-1,0),(1,0),(0,-1),(0,1)]
    Q = defaultdict(lambda: np.zeros(len(actions)))

    if load_prev and maze_hash:
        q_loaded = load_qtable(maze_hash)
        if q_loaded:
            Q = q_loaded

    if transfer_init:
        avg = average_qtables_for_size(rows, cols)
        if avg:
            for st,val in avg.items():
                Q[st] = np.array(val)

    eps = eps_start
    decay = (eps_start - eps_end) / max(1, episodes)
    rewards = []

    for ep in range(episodes):
        state = start
        total_reward = 0
        steps = 0
        done = False
        while not done and steps < max_steps_per_episode:
            if random.random() < eps:
                a = random.randint(0, len(actions)-1)
            else:
                a = int(np.argmax(Q[state]))
            nr, nc = state[0] + actions[a][0], state[1] + actions[a][1]
            if not (0 <= nr < rows and 0 <= nc < cols):
                new_state = state; rwd = -5; done=False
            elif maze[nr][nc] == 1:
                new_state = state; rwd = -5; done=False
            elif maze[nr][nc] == 2:
                new_state = (nr,nc); rwd = 10; done=True
            else:
                new_state = (nr,nc); rwd = -1; done=False
            total_reward += rwd
            Q[state][a] = Q[state][a] + alpha * (rwd + gamma * np.max(Q[new_state]) - Q[state][a])
            state = new_state
            steps += 1
        rewards.append(total_reward)
        eps = max(eps - decay, eps_end)
    return Q, rewards

# ------------------- Greedy policy & animation -------------------

def greedy_policy_path(maze, Q, max_steps=1000):
    start, goal = find_start_and_goal(maze)
    if start is None or goal is None:
        return [], 0
    actions = [(-1,0),(1,0),(0,-1),(0,1)]
    state = start; path=[state]; total=0
    for _ in range(max_steps):
        if state not in Q:
            break
        a = int(np.argmax(Q[state]))
        nr, nc = state[0]+actions[a][0], state[1]+actions[a][1]
        if not (0<=nr<len(maze) and 0<=nc<len(maze[0])) or maze[nr][nc] == 1:
            total += -5; break
        if maze[nr][nc] == 2:
            path.append((nr,nc)); total+=10; return path, total
        path.append((nr,nc)); total += -1; state = (nr,nc)
    return path, total

def animate_path(maze, path, interval=250, save_as=None):
    rows, cols = len(maze), len(maze[0])
    grid = np.array(maze)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_title("AgentX Greedy Path Animation")
    ax.imshow(grid == 1, cmap="gray_r")
    start, goal = find_start_and_goal(maze)
    if start: ax.scatter(start[1], start[0], c="green", marker="s", s=100, label="Start")
    if goal: ax.scatter(goal[1], goal[0], c="blue", marker="*", s=150, label="Goal")
    ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); ax.legend(loc="upper right")

    path2=[]
    for p in path:
        if isinstance(p,(list,tuple)) and len(p)>=2:
            path2.append((int(p[0]), int(p[1])))
    if not path2:
        print("No path to animate."); plt.show(); return

    agent_dot, = ax.plot([], [], "o", color="red", markersize=12)

    def init():
        agent_dot.set_data([], [])
        return (agent_dot,)

    def update(i):
        r,c = path2[min(i, len(path2)-1)]
        agent_dot.set_data([c], [r])  # pass sequences
        return (agent_dot,)

    ani = animation.FuncAnimation(fig, update, frames=len(path2), init_func=init,
                                  interval=interval, blit=True, repeat=False)
    if save_as:
        try:
            ani.save(save_as, writer="pillow", fps=max(1, 1000//interval))
            print("Saved animation to", save_as)
        except Exception as e:
            print("Could not save animation:", e)
    try:
        plt.show()
    except Exception as e:
        print("Animation display failed:", e)
        draw_stepwise(maze, path2)

def draw_stepwise(maze, path, pause=0.15):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(np.array(maze) == 1, cmap="gray_r")
    st,gl = find_start_and_goal(maze)
    if st: ax.scatter(st[1], st[0], c="green", marker="s", s=100)
    if gl: ax.scatter(gl[1], gl[0], c="blue", marker="*", s=150)
    ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    for (r,c) in path:
        ax.plot(c, r, "o", color="red"); plt.pause(pause)
    plt.show()

# ------------------- Analysis & Reporting -------------------

def save_rewards(rewards, label, ts):
    path = os.path.join("outputs", f"rewards_{label}_{ts}.npy")
    np.save(path, np.array(rewards))
    return path

def compute_convergence_episode(rewards, window=20, threshold_fraction=0.9):
    """
    Estimate first episode where moving average (window) reaches threshold_fraction of max moving average.
    Returns episode index or None.
    """
    if len(rewards) < window*2:
        return None
    mov = np.convolve(rewards, np.ones(window)/window, mode='valid')
    maxv = mov.max()
    if maxv == 0:
        return None
    threshold = maxv * threshold_fraction
    for i, val in enumerate(mov):
        if val >= threshold:
            return i + window - 1  # index in original rewards
    return None

def analyze_run(maze, Q, rewards, label, ts):
    """
    Create an analysis dict for a run, including success, greedy path length, stats, convergence.
    """
    path, greedy_score = greedy_policy_path(maze, Q)
    start, goal = find_start_and_goal(maze)
    success = False
    steps_to_goal = None
    if path:
        if path[-1] == goal:
            success = True
            steps_to_goal = len(path)-1
        else:
            success = False
            steps_to_goal = len(path)-1
    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    mean_last50 = float(np.mean(rewards[-50:])) if len(rewards) >= 50 else mean_reward
    convergence = compute_convergence_episode(rewards, window=20, threshold_fraction=0.9)
    analysis = {
        "label": label,
        "timestamp": ts,
        "final_greedy_score": int(greedy_score),
        "success": bool(success),
        "steps_to_goal": steps_to_goal,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_last50": mean_last50,
        "convergence_episode": int(convergence) if convergence is not None else None,
        "num_episodes": len(rewards)
    }
    return analysis

def save_report(analysis, rewards_path, training_plot, path_image):
    # JSON and TXT report
    label = analysis.get("label", "run")
    ts = analysis.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    base = os.path.join("outputs", f"report_{label}_{ts}")
    # JSON
    with open(base + ".json", "w") as f:
        json.dump({"analysis": analysis, "training_plot":training_plot, "path_image":path_image, "rewards_file":rewards_path}, f, indent=2)
    # TXT human-readable
    with open(base + ".txt", "w") as f:
        f.write(f"AgentX Run Report — {label} @ {ts}\n")
        f.write("="*40 + "\n")
        for k,v in analysis.items():
            f.write(f"{k}: {v}\n")
        f.write("\nFiles:\n")
        f.write(f"- Training plot: {training_plot}\n")
        f.write(f"- Greedy path image: {path_image}\n")
        f.write(f"- Rewards file: {rewards_path}\n")
    return base + ".json", base + ".txt"

def plot_comparison_with_history(current_rewards, label, ts):
    # Plot current rewards and overlay last few runs from history if present
    history = load_history().get("runs", [])
    fig = plt.figure(figsize=(10,5))
    plt.plot(current_rewards, label=f"{label} (current)")
    # overlay up to 3 most recent different runs
    cnt=0
    for r in reversed(history[-6:]):
        try:
            rp = r.get("rewards_file", None)
            if rp and os.path.exists(rp):
                vec = np.load(rp)
                plt.plot(vec, alpha=0.5, label=f"{r['label']} ({r['timestamp']})")
                cnt += 1
            if cnt>=3:
                break
        except Exception:
            continue
    plt.title("Training Curve Comparison")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.grid(True); plt.legend()
    out = os.path.join("outputs", f"comparison_{label}_{ts}.png")
    save_figure(fig, out)
    return out

# ------------------- Run and record (with saving rewards & analysis) -------------------

def run_on_maze(maze, label=None, episodes=600, transfer_init=False, animate=True):
    ensure_dirs()
    maze_hash = maze_to_hash(maze)
    label = label or maze_hash
    print(f"=== Running on maze '{label}' (hash={maze_hash}) ===")
    Q, rewards = train_q_learning(maze, episodes=episodes, transfer_init=transfer_init, maze_hash=maze_hash)
    save_qtable(Q, maze_hash)
    path, score = greedy_policy_path(maze, Q)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save rewards array
    rewards_path = os.path.join("outputs", f"rewards_{label}_{ts}.npy")
    np.save(rewards_path, np.array(rewards))
    # save training plot
    training_path = os.path.join("outputs", f"training_{label}_{ts}.png")
    fig = plt.figure(figsize=(8,4)); plt.plot(rewards); plt.title(f"Episode Reward over Time ({label})")
    plt.xlabel("Episode"); plt.ylabel("Total Reward"); plt.grid(True)
    save_figure(fig, training_path)
    # save greedy path image
    path_img = os.path.join("outputs", f"path_{label}_{ts}.png")
    fig2, ax = plt.subplots(figsize=(5,5)); ax.imshow(np.array(maze)==1, cmap="gray_r")
    if path: ax.scatter([p[1] for p in path],[p[0] for p in path], c="red")
    st,gl = find_start_and_goal(maze)
    if st: ax.scatter(st[1], st[0], c="green", marker="s", s=100)
    if gl: ax.scatter(gl[1], gl[0], c="blue", marker="*", s=150)
    ax.set_title("Greedy Path after Training"); plt.gca().invert_yaxis(); save_figure(fig2, path_img)
    # analyze
    analysis = analyze_run(maze, Q, rewards, label, ts)
    # comparison plot
    comp_plot = plot_comparison_with_history(rewards, label, ts)
    # save report
    rep_json, rep_txt = save_report(analysis, rewards_path, training_path, path_img)
    # update history (store pointers to rewards file)
    history = load_history()
    entry = {
        "label": label, "hash": maze_hash, "episodes": episodes,
        "final_score": int(analysis["final_greedy_score"]),
        "mean_last50": analysis["mean_last50"],
        "rewards_file": rewards_path,
        "training_plot": training_path, "path_image": path_img,
        "report_json": rep_json, "report_txt": rep_txt, "timestamp": ts
    }
    history.setdefault("runs", []).append(entry)
    save_history(history)
    print("Training complete. Final greedy score:", analysis["final_greedy_score"])
    print("Analysis saved to:", rep_txt)
    if animate:
        try:
            animate_path(maze, path, interval=250)
        except Exception as e:
            print("Animation failed:", e); draw_stepwise(maze, path)
    return entry

# ------------------- CLI & Menu -------------------

def print_menu():
    print("\nAgentX — Adaptive Maze RL (Problem Statement 1)")
    print("1) Generate random solvable maze M N (you input M N only)")
    print("2) Use example mazes (quick demo)")
    print("3) Show history summary (past runs)")
    print("4) Exit")
    print("5) Generate Auto Analysis Report (all runs) <-- NEW")
    return input("Choose (1-5): ").strip()

def input_maze_manual():
    print("Enter maze size M N (e.g. '5 7'):")
    while True:
        try:
            s = input().strip()
            m,n = map(int, s.split()); break
        except Exception:
            print("Invalid. Provide two integers like: 5 7")
    print("Now enter each row as space-separated integers (0 free, 1 wall, 2 goal):")
    maze=[] 
    for i in range(m):
        while True:
            row = input(f"Row {i+1}: ").strip().split()
            if len(row) != n:
                print(f"Need {n} values.")
                continue
            try:
                r = [int(x) for x in row]; maze.append(r); break
            except:
                print("Invalid ints.")
    return maze

def show_history_summary():
    history = load_history().get("runs", [])
    if not history:
        print("No runs yet."); return
    print("\nLast runs (most recent first):")
    for r in reversed(history[-10:]):
        print(f"- {r['timestamp']} | {r['label']} | episodes={r['episodes']} | final={r['final_score']} | mean_last50={r['mean_last50']:.2f}")

def generate_overall_report_all_runs():
    history = load_history().get("runs", [])
    if not history:
        print("No runs to analyze.")
        return
    # aggregate metrics
    rows = []
    for r in history:
        rows.append({
            "label": r.get("label"),
            "timestamp": r.get("timestamp"),
            "final_score": r.get("final_score"),
            "mean_last50": r.get("mean_last50")
        })
    # produce CSV-like txt
    out_txt = os.path.join("outputs", f"overall_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(out_txt, "w") as f:
        f.write("Overall AgentX Runs Summary\n")
        f.write("="*50 + "\n")
        for item in rows:
            f.write(f"{item['timestamp']} | {item['label']} | final={item['final_score']} | mean_last50={item['mean_last50']}\n")
    # optional: plot final scores distribution
    fig = plt.figure(figsize=(8,4))
    plt.plot([item['final_score'] for item in rows], marker='o')
    plt.title("Final Greedy Scores Across Runs"); plt.xlabel("Run Index"); plt.ylabel("Final Score"); plt.grid(True)
    out_plot = out_txt.replace(".txt", ".png")
    save_figure(fig, out_plot)
    print("Overall report saved:", out_txt, out_plot)
    return out_txt, out_plot

def interactive():
    ensure_dirs()
    while True:
        c = print_menu()
        if c == '1':
            try:
                s = input("Enter M N (e.g. '10 12'): ").strip(); m,n = map(int, s.split())
            except Exception:
                print("Invalid size."); continue
            wp = float(input("Wall probability (0.0-0.6, default 0.25): ").strip() or 0.25)
            print("Generating a solvable random maze...")
            maze = generate_random_solvable_maze(m, n, wall_prob=wp)
            print("Generated maze (rows):")
            for row in maze: print(row)
            lbl = input("Label (optional): ").strip() or f"random_{m}x{n}"
            episodes = int(input("Episodes (default 600): ").strip() or 600)
            ti = input("Transfer-init? (y/n, default n): ").strip().lower() == 'y'
            animate = input("Animate? (y/n, default y): ").strip().lower() != 'n'
            run_on_maze(maze, label=lbl, episodes=episodes, transfer_init=ti, animate=animate)
        elif c == '2':
            print("Using example mazes")
            maze1 = [
             [0,0,0,0,0,0,1,0,0,0],
             [1,1,0,1,1,0,1,0,1,0],
             [0,0,0,0,0,0,0,0,1,0],
             [0,1,1,1,1,1,1,0,1,0],
             [0,0,0,0,0,0,0,0,0,0],
             [0,1,0,1,0,1,0,1,0,1],
             [0,1,0,1,0,1,0,1,0,0],
             [0,0,0,1,0,0,0,0,1,0],
             [0,1,0,0,0,1,1,0,0,0],
             [0,0,0,1,0,0,0,0,1,2]
            ]
            maze2 = [
             [0,0,1,0,0,0,0],
             [0,1,1,0,1,1,0],
             [0,0,0,0,0,1,0],
             [1,1,0,1,0,0,0],
             [2,0,0,1,0,1,0]
            ]
            episodes = int(input("Episodes (default 600): ").strip() or 600)
            animate = input("Animate? (y/n, default y): ").strip().lower() != 'n'
            run_on_maze(maze1, label="example_10x10", episodes=episodes, transfer_init=False, animate=animate)
            run_on_maze(maze2, label="example_5x7", episodes=episodes, transfer_init=True, animate=animate)
        elif c == '3':
            show_history_summary()
        elif c == '4':
            print("Exiting. Good luck!")
            break
        elif c == '5':
            print("Generating overall auto analysis report for all runs...")
            generate_overall_report_all_runs()
        else:
            print("Choose a valid option (1-5).")

if __name__ == "__main__":
    interactive()
