"""
Snake RL — Q-Learning Agent
Fixed: server-side episode management, proper training loop, background thread
"""

from flask import Flask, render_template, jsonify, request
import random, threading, time

app = Flask(__name__)

GRID_SIZE  = 20
ACTIONS    = ['UP', 'DOWN', 'LEFT', 'RIGHT']
OPPOSITES  = {'UP':'DOWN','DOWN':'UP','LEFT':'RIGHT','RIGHT':'LEFT'}
ACTION_MAP = {'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}


class QLearningAgent:
    def __init__(self):
        self.q_table       = {}
        self.alpha         = 0.15
        self.gamma         = 0.95
        self.epsilon       = 1.0
        self.epsilon_min   = 0.01
        self.epsilon_decay = 0.997
        self.episode       = 0
        self.best_score    = 0
        self.total_steps   = 0

    def get_state(self, snake, food):
        head = snake[0]
        body = set(snake[1:])
        def danger(r, c):
            return int(r < 0 or r >= GRID_SIZE or c < 0 or c >= GRID_SIZE or (r,c) in body)
        return (
            danger(head[0]-1, head[1]),
            danger(head[0]+1, head[1]),
            danger(head[0],   head[1]-1),
            danger(head[0],   head[1]+1),
            int(food[0] < head[0]),
            int(food[0] > head[0]),
            int(food[1] < head[1]),
            int(food[1] > head[1]),
        )

    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in ACTIONS}
        return self.q_table[state]

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        q = self.get_q(state)
        return max(q, key=q.get)

    def learn(self, s, a, r, ns, done):
        q     = self.get_q(s)
        qnext = self.get_q(ns)
        target = r if done else r + self.gamma * max(qnext.values())
        q[a] += self.alpha * (target - q[a])
        self.total_steps += 1

    def end_episode(self, score):
        self.episode += 1
        if score > self.best_score:
            self.best_score = score
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    @property
    def stats(self):
        return {
            'episode':  self.episode,
            'epsilon':  round(self.epsilon, 4),
            'q_states': len(self.q_table),
            'best':     self.best_score,
            'steps':    self.total_steps,
        }


class SnakeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        mid = GRID_SIZE // 2
        self.snake     = [(mid, mid), (mid, mid-1), (mid, mid-2)]
        self.direction = 'RIGHT'
        self.score     = 0
        self.steps     = 0
        self.max_steps = GRID_SIZE * GRID_SIZE * 3
        self.food      = self._place_food()
        self.alive     = True

    def _place_food(self):
        occupied = set(self.snake)
        while True:
            p = (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
            if p not in occupied:
                return p

    def step(self, action):
        if action != OPPOSITES.get(self.direction):
            self.direction = action
        dr, dc = ACTION_MAP[self.direction]
        head   = (self.snake[0][0]+dr, self.snake[0][1]+dc)

        if (head[0] < 0 or head[0] >= GRID_SIZE or
                head[1] < 0 or head[1] >= GRID_SIZE or
                head in set(self.snake[1:])):
            self.alive = False
            return -10.0, True

        self.snake.insert(0, head)
        self.steps += 1

        if head == self.food:
            self.score += 1
            self.food   = self._place_food()
            return 10.0, False

        prev = abs(self.snake[1][0]-self.food[0]) + abs(self.snake[1][1]-self.food[1])
        curr = abs(head[0]-self.food[0])           + abs(head[1]-self.food[1])
        self.snake.pop()
        reward = 1.0 if curr < prev else -1.0
        if self.steps >= self.max_steps:
            return reward, True
        return reward, False

    def board(self):
        g = [['empty']*GRID_SIZE for _ in range(GRID_SIZE)]
        for i,(r,c) in enumerate(self.snake):
            g[r][c] = 'head' if i==0 else 'body'
        g[self.food[0]][self.food[1]] = 'food'
        return g


# ── Shared state ──────────────────────────────────────────────────────────────
agent   = QLearningAgent()
env     = SnakeEnv()
history = []   # list of {episode, score}
lock    = threading.Lock()
live    = {'board': env.board(), 'score': 0, 'episode': 0}


class Trainer:
    def __init__(self):
        self.running    = False
        self.mode       = 'idle'
        self.thread     = None
        self.step_delay = 0.08

    def _watch_loop(self):
        while self.running and self.mode == 'watch':
            with lock:
                s      = agent.get_state(env.snake, env.food)
                action = agent.choose_action(s)
                r, done = env.step(action)
                ns     = agent.get_state(env.snake, env.food)
                agent.learn(s, action, r, ns, done)
                live['board']   = env.board()
                live['score']   = env.score
                live['episode'] = agent.episode
                if done or not env.alive:
                    history.append({'episode': agent.episode, 'score': env.score})
                    agent.end_episode(env.score)
                    env.reset()
            time.sleep(self.step_delay)

    def _fast_loop(self, n):
        for _ in range(n):
            if not self.running:
                break
            with lock:
                env.reset()
            for _ in range(env.max_steps):
                if not self.running:
                    break
                with lock:
                    s      = agent.get_state(env.snake, env.food)
                    action = agent.choose_action(s)
                    r, done = env.step(action)
                    ns     = agent.get_state(env.snake, env.food)
                    agent.learn(s, action, r, ns, done)
                    if done or not env.alive:
                        history.append({'episode': agent.episode, 'score': env.score})
                        agent.end_episode(env.score)
                        live['board']   = env.board()
                        live['score']   = env.score
                        live['episode'] = agent.episode
                        break
        with lock:
            env.reset()
            live['board']   = env.board()
            live['score']   = 0
        self.running = False
        self.mode    = 'idle'

    def start_watch(self):
        self._stop_thread()
        self.running = True
        self.mode    = 'watch'
        self.thread  = threading.Thread(target=self._watch_loop, daemon=True)
        self.thread.start()

    def start_fast(self, n):
        self._stop_thread()
        self.running = True
        self.mode    = 'fast'
        self.thread  = threading.Thread(target=self._fast_loop, args=(n,), daemon=True)
        self.thread.start()

    def pause(self):
        self.running = False
        self._stop_thread()
        self.mode = 'idle'

    def _stop_thread(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)

    def set_speed(self, fps):
        self.step_delay = max(1/60, 1/max(1, fps))


trainer = Trainer()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/state')
def get_state():
    with lock:
        return jsonify({
            'board':   live['board'],
            'score':   live['score'],
            'agent':   agent.stats,
            'history': history[-100:],
            'mode':    trainer.mode,
            'running': trainer.running,
        })

@app.route('/api/start', methods=['POST'])
def start():
    trainer.start_watch()
    return jsonify({'ok': True, 'mode': 'watch'})

@app.route('/api/pause', methods=['POST'])
def pause():
    trainer.pause()
    return jsonify({'ok': True, 'mode': 'idle'})

@app.route('/api/reset', methods=['POST'])
def reset_all():
    trainer._stop_thread()
    with lock:
        agent.__init__()
        env.reset()
        history.clear()
        live['board']   = env.board()
        live['score']   = 0
        live['episode'] = 0
    trainer.running = False
    trainer.mode    = 'idle'
    return jsonify({'ok': True})

@app.route('/api/train', methods=['POST'])
def train():
    data = request.get_json() or {}
    n    = min(int(data.get('episodes', 200)), 1000)
    trainer.start_fast(n)
    return jsonify({'ok': True, 'training': n})

@app.route('/api/speed', methods=['POST'])
def speed():
    data = request.get_json() or {}
    fps  = int(data.get('fps', 10))
    trainer.set_speed(fps)
    return jsonify({'ok': True})

if __name__ == '__main__':
    print("\n  Snake RL ready → http://localhost:5000\n")
    app.run(debug=False, port=5000, threaded=True)
