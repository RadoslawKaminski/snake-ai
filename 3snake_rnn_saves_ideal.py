import os
import glob
import argparse
import datetime
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
# Argumenty wiersza poleceń
# =======================
parser = argparse.ArgumentParser(description="Train Snake RL Agent")
parser.add_argument("--model", type=str, default="new",
                    help="Tryb ładowania modelu: 'new' (domyślnie), 'latest' lub ścieżka do konkretnego pliku modelu")
args = parser.parse_args()
model_mode = args.model

# =======================
# Ustawienia gry i treningu
# =======================
CELL_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 30

MAX_EPISODES = 5000
mMAX_STEPS = 100

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.98
LR = 0.001
GAMMA = 0.9
BATCH_SIZE = 64

SHAPING_FACTOR = 0.8

# Parametry dla heurystyki dostępnej przestrzeni i trybu survival
SURVIVAL_THRESHOLD = 0.2   # Tryb survival, gdy dostępna przestrzeń < 20%
SURVIVAL_FACTOR = 10       # Współczynnik bonusu (lub kary) za zmianę dostępnej przestrzeni

# Parametry Lookahead
LOOKAHEAD_DEPTH = 5        # Głębokość symulacji – zmieniono z 2 do 5 (z dyskontowaniem)
LOOKAHEAD_WEIGHT = 5       # Waga bonusu dodawanego do Q-wartości

ALWAYS_VISUALIZE = True    # Jeśli True, uruchamia wizualizację za pomocą Pygame

# Parametry zapisywania modelu
SAVE_MODEL_EVERY = 10      # Zapis co 10 epizodów
MAX_MODELS_TO_SAVE = 5     # Zachowaj tylko ostatnie 5 modeli

# Jeśli tworzymy nową sesję, generujemy unikalny identyfikator treningu
if model_mode == "new":
    training_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
else:
    # Jeśli wczytujemy model (latest lub konkretny), spróbujmy wyciągnąć training_run_id z nazwy
    if model_mode == "latest":
        # Funkcja pomocnicza, aby pobrać najnowszy model:
        def get_latest_model():
            files = glob.glob("snake_model_*.pth")
            return max(files, key=os.path.getmtime) if files else None
        latest = get_latest_model()
        if latest is not None:
            training_run_id = latest.split("_")[2]
        else:
            print("Nie znaleziono modelu 'latest'. Uruchamiamy nowy model.")
            training_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_mode = "new"
    else:
        # Zakładamy, że nazwa pliku ma format: snake_model_{training_run_id}_ep{...}.pth
        if os.path.exists(model_mode):
            training_run_id = model_mode.split("_")[2]
        else:
            print("Podany plik modelu nie istnieje. Uruchamiamy nowy model.")
            training_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_mode = "new"

# =======================
# Funkcja zapisu modelu
# =======================
def save_model(model, episode, run_id, path_prefix="snake_model", max_models=MAX_MODELS_TO_SAVE):
    filename = f"{path_prefix}_{run_id}_ep{episode}.pth"
    torch.save(model.state_dict(), filename)
    print(f"Model zapisany: {filename}")
    # Usuń najstarsze pliki, jeśli jest ich za dużo
    files = sorted(glob.glob(f"{path_prefix}_{run_id}_ep*.pth"))
    if len(files) > max_models:
        os.remove(files[0])
        print(f"Usunięto najstarszy model: {files[0]}")

# =======================
# Funkcja decydująca, czy wizualizować dany epizod.
# =======================
def should_visualize(episode):
    if episode < 20:
        return (episode % 5 == 0)
    elif episode < 100:
        return (episode % 20 == 0)
    elif episode < 300:
        return (episode % 30 == 0)
    elif episode < 1000:
        return (episode % 50 == 0)
    else:
        return (episode % 200 == 0)

# =======================
# Funkcje pomocnicze – sterowanie, kolizje, odległość, obiekty
# =======================
def turn_left(direction):
    if direction == (0, -1): return (-1, 0)
    elif direction == (-1, 0): return (0, 1)
    elif direction == (0, 1): return (1, 0)
    elif direction == (1, 0): return (0, -1)

def turn_right(direction):
    if direction == (0, -1): return (1, 0)
    elif direction == (1, 0): return (0, 1)
    elif direction == (0, 1): return (-1, 0)
    elif direction == (-1, 0): return (0, -1)

def is_collision(pos, snake_body):
    x, y = pos
    return not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT) or pos in snake_body

def get_distance_in_direction(pos, direction, snake_body):
    steps = 0
    current = pos
    max_steps = max(GRID_WIDTH, GRID_HEIGHT)
    while True:
        current = (current[0] + direction[0], current[1] + direction[1])
        steps += 1
        if is_collision(current, snake_body) or steps > max_steps:
            break
    return steps / max_steps

def get_object_in_direction(pos, direction, snake, food):
    steps = 0
    current = pos
    max_steps = max(GRID_WIDTH, GRID_HEIGHT)
    while True:
        current = (current[0] + direction[0], current[1] + direction[1])
        steps += 1
        if not (0 <= current[0] < GRID_WIDTH and 0 <= current[1] < GRID_HEIGHT):
            return 1  # ściana
        if current in snake.body:
            return 1  # ciało
        if current == food:
            return 2  # jedzenie
        if steps > max_steps:
            return 0
    return 0

def rotate_vector(vector, angle):
    rad = math.radians(angle)
    return (vector[0]*math.cos(rad) - vector[1]*math.sin(rad),
            vector[0]*math.sin(rad) + vector[1]*math.cos(rad))

# =======================
# Heurystyka dostępnej przestrzeni
# =======================
def compute_available_space(head, snake_body):
    total_cells = GRID_WIDTH * GRID_HEIGHT
    visited = set()
    queue = [head]
    count = 0
    obstacles = set(snake_body)
    while queue:
        cell = queue.pop(0)
        if cell in visited:
            continue
        visited.add(cell)
        count += 1
        for n in [(cell[0]+1, cell[1]), (cell[0]-1, cell[1]), (cell[0], cell[1]+1), (cell[0], cell[1]-1)]:
            if (0 <= n[0] < GRID_WIDTH and 0 <= n[1] < GRID_HEIGHT) and (n not in obstacles) and (n not in visited):
                queue.append(n)
    return count / total_cells

# =======================
# Funkcje pomocnicze dla Lookahead
# =======================
def clone_snake(snake):
    new_snake = Snake(snake.body[0], snake.direction)
    new_snake.body = list(snake.body)
    new_snake.alive = snake.alive
    new_snake.growing = snake.growing
    return new_snake

def simulate_move(snake, food, action):
    new_snake = clone_snake(snake)
    if action == 1:
        new_snake.direction = turn_left(new_snake.direction)
    elif action == 2:
        new_snake.direction = turn_right(new_snake.direction)
    new_food, reward, done = step(new_snake, food)
    return new_snake, new_food, done, reward

def rollout_value(snake, food, depth, prev_avail):
    if not snake.alive:
        return -50  # kara za śmierć
    if depth == 0:
        return compute_available_space(snake.body[0], snake.body)
    best = -float('inf')
    for action in [0, 1, 2]:
        new_snake, new_food, done, immediate_reward = simulate_move(snake, food, action)
        if done:
            cumulative = immediate_reward
        else:
            new_avail = compute_available_space(new_snake.body[0], new_snake.body)
            avail_bonus = SURVIVAL_FACTOR * (new_avail - prev_avail)
            cumulative = immediate_reward + avail_bonus + GAMMA * rollout_value(new_snake, new_food, depth - 1, new_avail)
        best = max(best, cumulative)
    return best

# =======================
# Funkcja get_state – idealny wektor wejściowy
# =======================
def get_state(snake, food):
    features = []
    head = snake.body[0]
    # 1. Relatywna pozycja jedzenia (2)
    dx = (food[0] - head[0]) / GRID_WIDTH
    dy = (food[1] - head[1]) / GRID_HEIGHT
    features.extend([dx, dy])
    # 2. Aktualny kierunek (one-hot, 4)
    if snake.direction == (0, -1):
        features.extend([1, 0, 0, 0])
    elif snake.direction == (0, 1):
        features.extend([0, 1, 0, 0])
    elif snake.direction == (-1, 0):
        features.extend([0, 0, 1, 0])
    elif snake.direction == (1, 0):
        features.extend([0, 0, 0, 1])
    else:
        features.extend([0, 0, 0, 0])
    # 3. Globalna dostępność przestrzeni (1)
    avail = compute_available_space(head, snake.body)
    features.append(avail)
    # 4. Konfiguracja ciała – relatywne pozycje pierwszych dwóch segmentów (4 wartości)
    if len(snake.body) > 1:
        seg1 = snake.body[1]
        features.append((seg1[0] - head[0]) / GRID_WIDTH)
        features.append((seg1[1] - head[1]) / GRID_HEIGHT)
    else:
        features.extend([0, 0])
    if len(snake.body) > 2:
        seg2 = snake.body[2]
        features.append((seg2[0] - head[0]) / GRID_WIDTH)
        features.append((seg2[1] - head[1]) / GRID_HEIGHT)
    else:
        features.extend([0, 0])
    # 5. Lokalna analiza otoczenia dla 8 kierunków (0°,45°,90°,...,315°)
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    for angle in angles:
        d = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        dist = get_distance_in_direction(head, d, snake.body)
        features.append(dist)
        obj = get_object_in_direction(head, d, snake, food)
        if obj == 0:
            features.extend([1, 0, 0])
        elif obj == 1:
            features.extend([0, 1, 0])
        elif obj == 2:
            features.extend([0, 0, 1])
        else:
            features.extend([0, 0, 0])
    # 6. Pełna mapa planszy (20x20, każda komórka -> one-hot 4; 1600 elementów)
    board = np.zeros((GRID_HEIGHT, GRID_WIDTH, 4), dtype=float)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            pos = (x, y)
            if snake.body and pos == snake.body[0]:
                board[y, x, 3] = 1.0  # głowa
            elif snake.body and pos in snake.body:
                board[y, x, 2] = 1.0  # ciało
            elif pos == food:
                board[y, x, 1] = 1.0  # jedzenie
            else:
                board[y, x, 0] = 1.0  # puste
    features.extend(board.flatten().tolist())
    # Łącznie: 2+4+1+4+ (8*4=32) + 1600 = 1643 elementów
    return np.array(features, dtype=float)

# =======================
# Definicja sieci – AdvancedQNet
# =======================
class AdvancedQNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(AdvancedQNet, self).__init__()
        layers = []
        last_size = input_size
        for hs in hidden_sizes:
            layers.append(nn.Linear(last_size, hs))
            layers.append(nn.ReLU())
            last_size = hs
        layers.append(nn.Linear(last_size, output_size))
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

# =======================
# Agent DQN
# =======================
class DQNAgent:
    def __init__(self, state_size, hidden_sizes, output_size, lr=LR, gamma=GAMMA,
                 epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY, batch_size=BATCH_SIZE):
        self.state_size = state_size
        self.output_size = output_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=100000)
        self.model = AdvancedQNet(state_size, hidden_sizes, output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    def get_action(self, state, snake, food):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze().cpu().numpy()
        final_values = np.copy(q_values)
        current_avail = compute_available_space(snake.body[0], snake.body)
        for action in range(self.output_size):
            new_snake, new_food, done, immediate_reward = simulate_move(snake, food, action)
            if done:
                rollout_bonus = immediate_reward
            else:
                new_avail = compute_available_space(new_snake.body[0], new_snake.body)
                avail_bonus = SURVIVAL_FACTOR * (new_avail - current_avail)
                rollout_bonus = immediate_reward + avail_bonus + rollout_value(new_snake, new_food, LOOKAHEAD_DEPTH - 1, new_avail)
            final_values[action] += LOOKAHEAD_WEIGHT * rollout_bonus
        return int(np.argmax(final_values))
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
    direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    snake = Snake(pos, direction)
    food = generate_food(snake)
    return snake, food

def step(snake, food):
    head = snake.body[0]
    snake.new_head = (head[0] + snake.direction[0], head[1] + snake.direction[1])
    if not (0 <= snake.new_head[0] < GRID_WIDTH and 0 <= snake.new_head[1] < GRID_HEIGHT) or snake.new_head in snake.body:
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

def draw_board(snake, food, episode, score):
    screen.fill((0, 0, 0))
    for x in range(0, WINDOW_WIDTH + 1, CELL_SIZE):
        pygame.draw.line(screen, (40, 40, 40), (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT + 1, CELL_SIZE):
        pygame.draw.line(screen, (40, 40, 40), (0, y), (WINDOW_WIDTH, y))
    for segment in snake.body:
        rect = pygame.Rect(segment[0]*CELL_SIZE, segment[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, (0, 255, 0), rect)
    food_rect = pygame.Rect(food[0]*CELL_SIZE, food[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, (255, 0, 0), food_rect)
    font = pygame.font.SysFont("Arial", 20)
    info = font.render(f"Epizod: {episode}  Score: {score:.2f}", True, (255, 255, 255))
    screen.blit(info, (10, 10))
    pygame.display.flip()

# =======================
# Inicjalizacja Pygame (jeśli ALWAYS_VISUALIZE)
# =======================
if ALWAYS_VISUALIZE:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake RL – Trening pojedynczego węża")
    clock = pygame.time.Clock()

# =======================
# Inicjalizacja agenta
# =======================
state_size = 1643
hidden_sizes = [512, 256]  # Większa sieć ze względu na duże wejście
output_size = 3
agent = DQNAgent(state_size, hidden_sizes, output_size)

# Jeśli tryb nie jest "new", wczytujemy model
if model_mode != "new":
    if model_mode == "latest":
        def get_latest_model():
            files = glob.glob("snake_model_*.pth")
            return max(files, key=os.path.getmtime) if files else None
        latest = get_latest_model()
        if latest is not None:
            print(f"Ładowanie najnowszego modelu: {latest}")
            agent.model.load_state_dict(torch.load(latest))
        else:
            print("Nie znaleziono modelu 'latest'. Trening rozpocznie się od nowa.")
    else:
        if os.path.exists(model_mode):
            print(f"Ładowanie modelu z pliku: {model_mode}")
            agent.model.load_state_dict(torch.load(model_mode))
        else:
            print("Podany plik modelu nie istnieje. Trening rozpocznie się od nowa.")

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

    state = get_state(snake, food)
    old_head = snake.body[0]
    avail_old = compute_available_space(old_head, snake.body)

    while not done and step_count < mMAX_STEPS:
        step_count += 1
        if vis:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        action = agent.get_action(state, snake, food)
        if action == 1:
            snake.direction = turn_left(snake.direction)
        elif action == 2:
            snake.direction = turn_right(snake.direction)
        food, reward, done = step(snake, food)
        
        if snake.alive:
            head = snake.body[0]
            new_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
            old_distance = abs(old_head[0] - food[0]) + abs(old_head[1] - food[1])
            shaping = SHAPING_FACTOR * (old_distance - new_distance)
            reward += shaping
            avail_new = compute_available_space(head, snake.body)
            if avail_new < SURVIVAL_THRESHOLD:
                reward += SURVIVAL_FACTOR * (avail_new - avail_old)
            old_head = snake.body[0]
            avail_old = compute_available_space(old_head, snake.body)
            
        episode_score += reward
        new_state = get_state(snake, food)
        agent.remember(state, action, reward, new_state, done)
        agent.train()
        state = new_state

        if vis:
            draw_board(snake, food, episode, episode_score)
            pygame.time.delay(30)
    
    if episode % 5 == 0:
        mstep_helper = 0 
    else:
        mstep_helper += episode_score

    if mstep_helper > mMAX_STEPS * 4:
        mMAX_STEPS = int(mMAX_STEPS * 1.5)

    print(f"Epizod {episode}, steps: {step_count}, Score: {episode_score:.2f}, mMAX_STEPS: {mMAX_STEPS}, mstep_helper: {mstep_helper:.0f}")
    
    if episode % SAVE_MODEL_EVERY == 0:
        save_model(agent.model, episode, training_run_id)
        
    if vis:
        pygame.time.delay(500)
if ALWAYS_VISUALIZE:
    pygame.quit()
sys.exit()
