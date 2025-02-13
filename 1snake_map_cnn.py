import pygame
import random
import sys
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =======================
# Ustawienia gry i treningu
# =======================
CELL_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 25

MAX_EPISODES = 5000
mMAX_STEPS = 100

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.8
LR = 0.001
GAMMA = 0.9
BATCH_SIZE = 64

SHAPING_FACTOR = 4

ALWAYS_VISUALIZE = True  # Jeśli True, inicjalizujemy Pygame i wyświetlamy wizualizację

# Funkcja decydująca, czy wizualizować dany epizod
def should_visualize(episode):
    if episode < 20:
        return (episode % 2 == 0)
    elif episode < 100:
        return (episode % 10 == 0)
    elif episode < 300:
        return (episode % 30 == 0)
    elif episode < 1000:
        return (episode % 50 == 0)
    else:
        return (episode % 100 == 0)

# =======================
# Funkcje pomocnicze – sterowanie, kolizje, odległość
# =======================
def turn_left(direction):
    if direction == (0, -1):    # up
        return (-1, 0)
    elif direction == (-1, 0):  # left
        return (0, 1)
    elif direction == (0, 1):   # down
        return (1, 0)
    elif direction == (1, 0):   # right
        return (0, -1)

def turn_right(direction):
    if direction == (0, -1):    # up
        return (1, 0)
    elif direction == (1, 0):   # right
        return (0, 1)
    elif direction == (0, 1):   # down
        return (-1, 0)
    elif direction == (-1, 0):  # left
        return (0, -1)

def is_collision(pos, snake_body):
    x, y = pos
    if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
        return True
    if pos in snake_body:
        return True
    return False

def get_distance_in_direction(pos, direction, snake_body):
    steps = 0
    current = pos
    max_steps = max(GRID_WIDTH, GRID_HEIGHT)
    while True:
        current = (current[0] + direction[0], current[1] + direction[1])
        steps += 1
        if is_collision(current, snake_body):
            break
        if steps > max_steps:
            break
    return steps / max_steps

# =======================
# Reprezentacja stanu – pełna mapa (7 kanałów)
# =======================
def get_state_grid(snake, food):
    """
    Zwraca stan jako numpy array o wymiarach (7, GRID_HEIGHT, GRID_WIDTH):
      Kanał 0: Ciało węża (wszystkie segmenty poza głową)
      Kanał 1: Jedzenie
      Kanał 2: Głowa węża
      Kanały 3-6: One-hot kodowanie kierunku głowy (up, down, left, right) – tylko w komórce głowy
    """
    state = np.zeros((7, GRID_HEIGHT, GRID_WIDTH), dtype=np.float32)
    # Kanał 0: ciało (bez głowy)
    for pos in snake.body[1:]:
        x, y = pos
        state[0, y, x] = 1.0
    # Kanał 1: jedzenie
    food_x, food_y = food
    state[1, food_y, food_x] = 1.0
    # Kanał 2: głowa
    head = snake.body[0]
    head_x, head_y = head
    state[2, head_y, head_x] = 1.0
    # Kanały 3-6: one-hot kierunku (umieszczone w komórce głowy)
    if snake.direction == (0, -1):  # up
        state[3, head_y, head_x] = 1.0
    elif snake.direction == (0, 1):  # down
        state[4, head_y, head_x] = 1.0
    elif snake.direction == (-1, 0):  # left
        state[5, head_y, head_x] = 1.0
    elif snake.direction == (1, 0):  # right
        state[6, head_y, head_x] = 1.0
    return state

# =======================
# Definicja sieci konwolucyjnej (ConvQNet)
# =======================
class ConvQNet(nn.Module):
    def __init__(self, output_size):
        super(ConvQNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=3, stride=1, padding=1),  # -> (32,20,20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # -> (64,20,20)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # -> (64,20,20)
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * GRID_HEIGHT * GRID_WIDTH, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# =======================
# Agent DQN
# =======================
# Teraz wyjścia to 4 akcje: 0: góra, 1: dół, 2: lewo, 3: prawo
class DQNAgent:
    def __init__(self, output_size, lr=LR, gamma=GAMMA,
                 epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY, batch_size=BATCH_SIZE):
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=100000)
        self.model = ConvQNet(output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # kształt: (1,7,20,20)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return int(torch.argmax(q_values).item())
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        q_values = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.model(next_states).max(1)[0].unsqueeze(1)
        target = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.criterion(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# =======================
# Klasa reprezentująca węża
# =======================
class Snake:
    def __init__(self, pos, direction):
        self.body = [pos]
        self.direction = direction
        self.alive = True
        self.new_head = None
        self.growing = False

# =======================
# Funkcje środowiskowe
# =======================
def generate_food(snake):
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake.body:
            return pos

def reset_game():
    pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
    direction = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
    snake = Snake(pos, direction)
    food = generate_food(snake)
    return snake, food

def step(snake, food):
    head = snake.body[0]
    snake.new_head = (head[0] + snake.direction[0], head[1] + snake.direction[1])
    if not (0 <= snake.new_head[0] < GRID_WIDTH and 0 <= snake.new_head[1] < GRID_HEIGHT) or \
       snake.new_head in snake.body:
        snake.alive = False
        return food, -60, True
    if snake.new_head == food:
        snake.growing = True
        reward = 40
    else:
        snake.growing = False
        reward = -0.3
    if snake.growing:
        snake.body = [snake.new_head] + snake.body
        food = generate_food(snake)
    else:
        snake.body = [snake.new_head] + snake.body[:-1]
    return food, reward, False

def manhattan_distance(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def draw_board(snake, food, episode, score):
    screen.fill((0,0,0))
    for x in range(0, WINDOW_WIDTH + 1, CELL_SIZE):
        pygame.draw.line(screen, (40,40,40), (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT + 1, CELL_SIZE):
        pygame.draw.line(screen, (40,40,40), (0, y), (WINDOW_WIDTH, y))
    for segment in snake.body:
        rect = pygame.Rect(segment[0]*CELL_SIZE, segment[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (0,255,0), rect)
    food_rect = pygame.Rect(food[0]*CELL_SIZE, food[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, (255,0,0), food_rect)
    font = pygame.font.SysFont("Arial", 20)
    info = font.render(f"Epizod: {episode}  Score: {score:.2f}", True, (255,255,255))
    screen.blit(info, (10,10))
    pygame.display.flip()

# =======================
# Inicjalizacja Pygame (jeśli ALWAYS_VISUALIZE=True)
# =======================
if ALWAYS_VISUALIZE:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake RL – Trening pojedynczego węża")
    clock = pygame.time.Clock()

# =======================
# Inicjalizacja agenta – teraz wyjścia: 4 akcje: 0: góra, 1: dół, 2: lewo, 3: prawo
# =======================
output_size = 4
agent = DQNAgent(output_size=output_size)

# =======================
# Główna pętla treningowa
# =======================
mstep_helper = 0
for episode in range(1, MAX_EPISODES + 1):
    vis = should_visualize(episode) if ALWAYS_VISUALIZE else False
    snake, food = reset_game()
    step_count = 0
    done = False
    episode_score = 0.0
    while not done and step_count < mMAX_STEPS:
        step_count += 1
        if vis:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        # Stan wejściowy to pełna mapa (7,20,20)
        state = get_state_grid(snake, food)
        if snake.alive:
            head = snake.body[0]
            old_distance = manhattan_distance(head, food)
        else:
            old_distance = 0
        action = agent.get_action(state)
        # Zmiana kierunku – teraz działamy na akcjach absolutnych:
        # 0: góra, 1: dół, 2: lewo, 3: prawo
        # Zapobiegamy cofnięciu się:
        if action == 0 and snake.direction != (0, 1):
            snake.direction = (0, -1)
        elif action == 1 and snake.direction != (0, -1):
            snake.direction = (0, 1)
        elif action == 2 and snake.direction != (1, 0):
            snake.direction = (-1, 0)
        elif action == 3 and snake.direction != (-1, 0):
            snake.direction = (1, 0)
        food, reward, done = step(snake, food)
        if snake.alive and reward != 30:
            head = snake.body[0]
            new_distance = manhattan_distance(head, food)
            shaping = SHAPING_FACTOR * (old_distance - new_distance)
            reward += shaping
        episode_score += reward
        new_state = get_state_grid(snake, food)
        agent.remember(state, action, reward, new_state, done)
        agent.train()
        if vis:
            draw_board(snake, food, episode, episode_score)
            pygame.time.delay(int(1000/FPS))
    if (episode % 5 == 0):
        mstep_helper = 0 
    else:
        mstep_helper += episode_score
    if (mstep_helper > mMAX_STEPS * 4):
        mMAX_STEPS *= 1.5
    print(f"Epizod {episode}, steps: {step_count}, Score: {episode_score:.2f}, mMAX_STEPS: {mMAX_STEPS:.0f}, mstep_helper: {mstep_helper:.0f}")
    if vis:
        pygame.time.delay(500)
if ALWAYS_VISUALIZE:
    pygame.quit()
sys.exit()
