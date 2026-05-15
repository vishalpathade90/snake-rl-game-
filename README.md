# 🐍 Snake RL — Q-Learning Agent

A reinforcement learning Snake game where an AI agent learns to play using **Q-Learning**.

---

## Tech Stack
- **Backend**: Python + Flask
- **RL Algorithm**: Q-Learning (tabular, state = 8-bit danger + food direction)
- **Frontend**: Vanilla JS + HTML5 Canvas (cyberpunk neon aesthetic)

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the server
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

---

## How It Works

### State Space (8 bits)
| Bit | Meaning |
|-----|---------|
| danger_up | Wall/body directly above head |
| danger_down | Wall/body directly below head |
| danger_left | Wall/body directly left |
| danger_right | Wall/body directly right |
| food_up | Food is above |
| food_down | Food is below |
| food_left | Food is to the left |
| food_right | Food is to the right |

### Rewards
| Event | Reward |
|-------|--------|
| Eat food | +10 |
| Move toward food | +1 |
| Move away from food | -0.5 |
| Die (wall/self) | -10 |

### Q-Learning Parameters
| Param | Value |
|-------|-------|
| Learning rate (α) | 0.1 |
| Discount (γ) | 0.95 |
| Epsilon start | 1.0 |
| Epsilon min | 0.01 |
| Epsilon decay | 0.995 |

---

## UI Controls
| Button | Action |
|--------|--------|
| ▶ Start Agent | Begin real-time RL training |
| ⏸ Pause | Pause/resume the agent |
| ↺ Reset Game | Reset current game |
| ⚡ Train 100 Eps | Silent fast training (100 episodes) |
| Speed slider | Adjust simulation speed (1–60 fps) |
