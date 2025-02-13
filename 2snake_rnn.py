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
SURVIVAL_THRESHOLD = 0.2  # Tryb survival: gdy dostępna przestrzeń < 20%
SURVIVAL_FACTOR = 10      # Współczynnik kształtowania nagrody związany z dostępnością przestrzeni

# Parametry Lookahead
LOOKAHEAD_DEPTH = 5       # Głębokość symulacji (ile ruchów do przodu)
LOOKAHEAD_WEIGHT = 5      # Waga bonusu dodawanego do Q-wartości na podstawie symulacji lookahead

ALWAYS_VISUALIZE = True   # Jeśli True, inicjalizujemy moduł Pygame

# Funkcja decydująca, czy wizualizować dany epizod.
def should_visualize(episode):
    if episode < 100:
        return (episode % 5 == 0)
    elif episode < 1000:
        return (episode % 20 == 0)
    else:
        return (episode % 50 == 0)

# =======================
# Funkcje pomocnicze – sterowanie, kolizje, odległość, obiekty
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
# Heurystyka dostępnej przestrzeni
# =======================
def compute_available_space(head, snake_body):
    """
    Oblicza dostępny obszar (procent planszy) osiągalny z pozycji head,
    traktując komórki zajęte przez ciało węża jako przeszkody.
    """
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
        # Sprawdzamy sąsiadów (4 kierunki)
        neighbors = [(cell[0]+1, cell[1]), (cell[0]-1, cell[1]), (cell[0], cell[1]+1), (cell[0], cell[1]-1)]
        for n in neighbors:
            if (0 <= n[0] < GRID_WIDTH and 0 <= n[1] < GRID_HEIGHT) and (n not in obstacles) and (n not in visited):
                queue.append(n)
    return count / total_cells

# =======================
# Funkcje pomocnicze dla Lookahead
# =======================
def clone_snake(snake):
    """Tworzy płytką kopię stanu węża."""
    new_snake = Snake(snake.body[0], snake.direction)
    new_snake.body = list(snake.body)
    new_snake.alive = snake.alive
    new_snake.growing = snake.growing
    return new_snake

def simulate_move(snake, food, action):
    """
    Tworzy kopię węża, modyfikuje kierunek (zgodnie z akcją) i wykonuje jeden ruch.
    Zwraca: nowego węża, nową pozycję jedzenia, flagę końca gry oraz natychmiastową nagrodę.
    """
    new_snake = clone_snake(snake)
    if action == 1:
        new_snake.direction = turn_left(new_snake.direction)
    elif action == 2:
        new_snake.direction = turn_right(new_snake.direction)
    new_food, reward, done = step(new_snake, food)
    return new_snake, new_food, done, reward

def rollout_value(snake, food, depth, prev_avail):
    """
    Rekurencyjna symulacja (DFS) – dla zadanej głębokości (liczby ruchów) zwraca
    skumulowaną wartość (suma nagród i bonusów). Jeśli wąż ginie (np. kolizja z ciałem),
    kończymy rekurencję dla tego ruchu i zwracamy natychmiastową nagrodę.
    """
    if not snake.alive:
        return -50  # silna kara za śmierć
    if depth == 0:
        return compute_available_space(snake.body[0], snake.body)
    best = -float('inf')
    for action in [0, 1, 2]:
        new_snake, new_food, done, immediate_reward = simulate_move(snake, food, action)
        # Jeśli ruch spowodował kolizję, kończymy rekurencję – nie szukamy dalej.
        if done:
            cumulative = immediate_reward
        else:
            new_avail = compute_available_space(new_snake.body[0], new_snake.body)
            avail_bonus = SURVIVAL_FACTOR * (new_avail - prev_avail)
            cumulative = immediate_reward + avail_bonus + rollout_value(new_snake, new_food, depth - 1, new_avail)
        best = max(best, cumulative)
    return best

# =======================
# Funkcja get_state – bez mapy, tylko podstawowe i dodatkowe cechy
#
# Stan składa się z:
#  - 11 podstawowych cech:
#      0: znormalizowany dystans w kierunku jazdy "prosto"
#      1: znormalizowany dystans w kierunku "lewo" (względem aktualnego kierunku)
#      2: znormalizowany dystans w kierunku "prawo"
#      3-6: one-hot kodowanie aktualnego kierunku (up, down, left, right)
#      7-8: relatywna pozycja jedzenia względem głowy (dx, dy, znormalizowane)
#      9: znormalizowana dostępna przestrzeń (0–1)
#      10: binarny wskaźnik trybu survival (1, gdy dostępna przestrzeń < SURVIVAL_THRESHOLD, 0 inaczej)
#  - 8 dodatkowych cech dla 4 kierunków po 45° (jak poprzednio)
#
# Łącznie: 11 + 8 = 19 elementów.
# =======================
def get_state(snake, food):
    basic = np.zeros(11, dtype=float)
    if not snake.body:
        basic[:] = 0
    else:
        head = snake.body[0]
        basic[0] = get_distance_in_direction(head, snake.direction, snake.body)
        basic[1] = get_distance_in_direction(head, turn_left(snake.direction), snake.body)
        basic[2] = get_distance_in_direction(head, turn_right(snake.direction), snake.body)
        basic[3] = 1 if snake.direction == (0, -1) else 0  # up
        basic[4] = 1 if snake.direction == (0, 1) else 0   # down
        basic[5] = 1 if snake.direction == (-1, 0) else 0  # left
        basic[6] = 1 if snake.direction == (1, 0) else 0   # right
        basic[7] = (food[0] - head[0]) / GRID_WIDTH
        basic[8] = (food[1] - head[1]) / GRID_HEIGHT

        # Heurystyka dostępnej przestrzeni
        avail = compute_available_space(head, snake.body)
        basic[9] = avail                   # znormalizowana dostępna przestrzeń
        basic[10] = 1 if avail < SURVIVAL_THRESHOLD else 0  # tryb survival

    # Dodatkowe cechy (8) – dla 4 kierunków (przód-lewo, przód-prawo, tył-lewo, tył-prawo)
    additional = np.zeros(8, dtype=float)
    if snake.body:
        head = snake.body[0]
        forward = snake.direction
        backward = (-forward[0], -forward[1])
        directions = [
            rotate_vector(forward, -45),  # przód-lewo
            rotate_vector(forward, 45),   # przód-prawo
            rotate_vector(backward, -45), # tył-lewo
            rotate_vector(backward, 45)   # tył-prawo
        ]
        for i, d in enumerate(directions):
            norm = math.sqrt(d[0]**2 + d[1]**2)
            if norm != 0:
                d_norm = (d[0]/norm, d[1]/norm)
            else:
                d_norm = (0, 0)
            additional[i*2] = get_distance_in_direction(head, d_norm, snake.body)
            additional[i*2+1] = get_object_in_direction(head, d_norm, snake, food)
    return np.concatenate([basic, additional])

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
        # Losowa eksploracja
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze().cpu().numpy()
        final_values = np.copy(q_values)
        current_avail = compute_available_space(snake.body[0], snake.body)
        # Dla każdej możliwej akcji wykonujemy symulację lookahead
        for action in range(self.output_size):
            new_snake, new_food, done, immediate_reward = simulate_move(snake, food, action)
            if done:
                rollout_bonus = immediate_reward  # Jeśli kolizja, nie szukamy dalej
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
# Stan: 11 cech podstawowych + 8 dodatkowych = 19
state_size = 11 + 8
hidden_sizes = [400, 400, 220, 22]
output_size = 3
agent = DQNAgent(state_size, hidden_sizes, output_size)

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

    # Stan początkowy i dostępna przestrzeń
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

        # Wybór akcji z uwzględnieniem lookahead (przekazujemy także stan węża i pozycję jedzenia)
        action = agent.get_action(state, snake, food)
        # Zmiana kierunku na podstawie akcji
        if action == 1:
            snake.direction = turn_left(snake.direction)
        elif action == 2:
            snake.direction = turn_right(snake.direction)
        # Wykonanie ruchu
        food, reward, done = step(snake, food)
        
        # Kształtowanie nagrody – zmiana dystansu do jedzenia
        if snake.alive:
            head = snake.body[0]
            new_distance = abs(head[0] - food[0]) + abs(head[1] - food[1])
            old_distance = abs(old_head[0] - food[0]) + abs(old_head[1] - food[1])
            shaping = SHAPING_FACTOR * (old_distance - new_distance)
            reward += shaping

            # Aktualizacja heurystyki dostępnej przestrzeni
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
        mMAX_STEPS *= 1.5

    print(f"Epizod {episode}, steps: {step_count}, Score: {episode_score:.2f}, mMAX_STEPS: {mMAX_STEPS:.0f}, mstep_helper: {mstep_helper:.0f}")
    if vis:
        pygame.time.delay(500)
if ALWAYS_VISUALIZE:
    pygame.quit()
sys.exit()
