import pygame
import random
import sys
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

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

SHAPING_FACTOR = 0.8

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

def get_object_in_direction(pos, direction, snake, food):
    """
    Zwraca:
      0 – jeśli nic nie napotkano,
      1 – jeśli napotkano przeszkodę (ściana lub ciało),
      2 – jeśli napotkano jedzenie.
    """
    steps = 0
    current = pos
    max_steps = max(GRID_WIDTH, GRID_HEIGHT)
    while True:
        current = (current[0] + direction[0], current[1] + direction[1])
        steps += 1
        if not (0 <= current[0] < GRID_WIDTH and 0 <= current[1] < GRID_HEIGHT):
            return 1
        if current in snake.body:
            return 1
        if current == food:
            return 2
        if steps > max_steps:
            return 0
        if is_collision(current, snake.body):
            return 1
    return 0

def rotate_vector(vector, angle):
    """Obraca wektor o zadany kąt (w stopniach)."""
    angle_rad = math.radians(angle)
    x = vector[0]*math.cos(angle_rad) - vector[1]*math.sin(angle_rad)
    y = vector[0]*math.sin(angle_rad) + vector[1]*math.cos(angle_rad)
    return (x, y)

# =======================
# Reprezentacja stanu – funkcja get_state (22 elementy)
# =======================
def get_state(snake, food):
    """
    Zwraca wektor stanu składający się z:
      - 14 podstawowych cech:
          0: znormalizowany dystans w kierunku jazdy "prosto"
          1: znormalizowany dystans w kierunku "lewo" (względem aktualnego kierunku)
          2: znormalizowany dystans w kierunku "prawo"
          3-6: one-hot kodowanie aktualnego kierunku (kolejność: up, down, left, right)
          7-8: relatywna pozycja jedzenia względem głowy (dx, dy, znormalizowane)
          9: znormalizowana długość węża
          10-11: relatywna pozycja pierwszego segmentu (jeśli dostępna) względem głowy
          12-13: relatywna pozycja drugiego segmentu (jeśli dostępna) względem głowy
      - 8 dodatkowych cech dla 4 kierunków po 45°:
          Dla każdego z 4 kierunków (przód‑lewo, przód‑prawo, tył‑lewo, tył‑prawo):
             – pierwsza cecha: znormalizowany dystans,
             – druga cecha: typ obiektu (0 – nic, 1 – przeszkoda, 2 – jedzenie)
      Łącznie: 14 + 8 = 22 elementów.
    """
    basic = np.zeros(14, dtype=float)
    if not snake.body:
        basic[:] = 0
    else:
        head = snake.body[0]
        basic[0] = get_distance_in_direction(head, snake.direction, snake.body)
        basic[1] = get_distance_in_direction(head, turn_left(snake.direction), snake.body)
        basic[2] = get_distance_in_direction(head, turn_right(snake.direction), snake.body)
        basic[3] = 1 if snake.direction == (0, -1) else 0
        basic[4] = 1 if snake.direction == (0, 1) else 0
        basic[5] = 1 if snake.direction == (-1, 0) else 0
        basic[6] = 1 if snake.direction == (1, 0) else 0
        basic[7] = (food[0] - head[0]) / GRID_WIDTH
        basic[8] = (food[1] - head[1]) / GRID_HEIGHT
        snake_length = len(snake.body)
        basic[9] = snake_length / (GRID_WIDTH * GRID_HEIGHT)
        if snake_length >= 2:
            seg1 = snake.body[1]
            basic[10] = (seg1[0] - head[0]) / GRID_WIDTH
            basic[11] = (seg1[1] - head[1]) / GRID_HEIGHT
        else:
            basic[10:12] = 0
        if snake_length >= 3:
            seg2 = snake.body[2]
            basic[12] = (seg2[0] - head[0]) / GRID_WIDTH
            basic[13] = (seg2[1] - head[1]) / GRID_HEIGHT
        else:
            basic[12:14] = 0

    additional = np.zeros(8, dtype=float)
    if snake.body:
        forward = snake.direction
        backward = (-forward[0], -forward[1])
        directions = [
            rotate_vector(forward, -45),  # Przód-lewo
            rotate_vector(forward, 45),   # Przód-prawo
            rotate_vector(backward, -45), # Tył-lewo
            rotate_vector(backward, 45)   # Tył-prawo
        ]
        for i, d in enumerate(directions):
            norm = math.sqrt(d[0]**2 + d[1]**2)
            if norm != 0:
                d_norm = (d[0]/norm, d[1]/norm)
            else:
                d_norm = (0, 0)
            additional[i*2] = get_distance_in_direction(head, d_norm, snake.body)
            additional[i*2+1] = get_object_in_direction(head, d_norm, snake, food)
    return np.concatenate([basic, additional])  # 14 + 8 = 22

# =======================
# Definicja sieci konwolucyjnej (ConvQNet)
# =======================
class ConvQNet(nn.Module):
    def __init__(self, output_size):
        super(ConvQNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(22, 32, kernel_size=3, stride=1, padding=1),  # -> (32,20,20)
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
# Teraz wyjścia to 3 akcje: 0: prosto, 1: lewo, 2: prawo
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
        state_tensor = torch.FloatTensor(state)

        #Powtórzenie kanałów
        state_tensor = state_tensor.repeat(GRID_HEIGHT, GRID_WIDTH, 1).permute(2,0,1).unsqueeze(0)

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

        #Powtórzenie kanałów dla wsadu
        states = torch.stack([torch.FloatTensor(s).repeat(GRID_HEIGHT, GRID_WIDTH, 1).permute(2,0,1) for s in states])

        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))

        #Powtórzenie kanałów dla next_states
        next_states = torch.stack([torch.FloatTensor(ns).repeat(GRID_HEIGHT, GRID_WIDTH, 1).permute(2,0,1) for ns in next_states])

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
        return food, -15, True
    if snake.new_head == food:
        snake.growing = True
        reward = 30
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
# Inicjalizacja agenta – teraz wyjścia: 3 akcje: 0: prosto, 1: lewo, 2: prawo
# =======================
output_size = 3
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
        # Stan wejściowy to wektor 22 elementowy
        state = get_state(snake, food)
        if snake.alive:
            head = snake.body[0]
            old_distance = manhattan_distance(head, food)
        else:
            old_distance = 0
        action = agent.get_action(state)
        # Zmiana kierunku – teraz działamy na akcjach relatywnych:
        # 0: prosto, 1: lewo, 2: prawo
        if action == 1:
            snake.direction = turn_left(snake.direction)
        elif action == 2:
            snake.direction = turn_right(snake.direction)

        food, reward, done = step(snake, food)
        if snake.alive and reward != 30:
            head = snake.body[0]
            new_distance = manhattan_distance(head, food)
            shaping = SHAPING_FACTOR * (old_distance - new_distance)
            reward += shaping
        episode_score += reward
        new_state = get_state(snake, food)
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