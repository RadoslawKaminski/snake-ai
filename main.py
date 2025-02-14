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
import re

# =============================================================================
# Pomocnicza funkcja konwertująca string na bool
# =============================================================================
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# =============================================================================
# Argumenty wiersza poleceń
# =============================================================================
# --model: "new" (domyślnie), "latest" lub ścieżka do modelu.
# --model_name: opcjonalna nazwa dla nowego modelu (bez numeru epizodu).
# Dodatkowo: --mmax_steps, --grid_width, --grid_height, --cell_size, --always_visualize, --lookahead_depth, --visualize_rate.
parser = argparse.ArgumentParser(description="Train Snake RL Agent")
parser.add_argument("--model", type=str, default="new",
                    help="Tryb ładowania modelu: 'new' (domyślnie), 'latest' lub ścieżka do modelu")
parser.add_argument("--model_name", type=str, default="",
                    help="Opcjonalna nazwa dla nowego modelu (bez numeru epizodu)")
parser.add_argument("--mmax_steps", type=int, default=100,
                    help="Maksymalna liczba kroków w epizodzie")
parser.add_argument("--grid_width", type=int, default=20,
                    help="Liczba komórek w poziomie")
parser.add_argument("--grid_height", type=int, default=20,
                    help="Liczba komórek w pionie")
parser.add_argument("--cell_size", type=int, default=20,
                    help="Rozmiar pojedynczej komórki (w pikselach)")
parser.add_argument("--always_visualize", type=str2bool, default=True,
                    help="Czy używać wizualizacji Pygame (True/False)")
parser.add_argument("--lookahead_depth", type=int, default=5,
                    help="Głębokość symulacji lookahead")
parser.add_argument("--visualize_rate", type=str, default="20",
                    help="Wartość określająca częstotliwość wizualizacji. Jeśli '0' - każdy epizod; "
                         "jeśli zaczyna się od 0 (np. '020') - dokładna wartość; w przeciwnym razie, "
                         "dla ep <20: co 5, dla ep <100: co wartość, dla ep <300: co 50, <1000: co 100, else: co 200.")
args = parser.parse_args()
model_mode = args.model
custom_model_name = args.model_name
visualize_rate_param = args.visualize_rate

# =============================================================================
# Ustawienia – zastępujemy stałe wartości argumentami
# =============================================================================
CELL_SIZE = args.cell_size
GRID_WIDTH = args.grid_width
GRID_HEIGHT = args.grid_height
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 30

MAX_EPISODES = 5000
mMAX_STEPS = args.mmax_steps

EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.98
LR = 0.001
GAMMA = 0.9
BATCH_SIZE = 64

SHAPING_FACTOR = 0.8

SURVIVAL_THRESHOLD = 0.2
SURVIVAL_FACTOR = 10

LOOKAHEAD_DEPTH = args.lookahead_depth
LOOKAHEAD_WEIGHT = 5

ALWAYS_VISUALIZE = args.always_visualize

# Folder do zapisywania modeli
MODELS_FOLDER = "models"
os.makedirs(MODELS_FOLDER, exist_ok=True)

SAVE_MODEL_EVERY = 10
MAX_MODELS_TO_SAVE = 5

# =============================================================================
# Funkcja ekstrakcji numeru epizodu z nazwy pliku
# =============================================================================
def extract_episode(filename):
    match = re.search(r'_ep(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

# =============================================================================
# Ustalenie identyfikatora treningu (training_run_id)
# =============================================================================
if model_mode == "new":
    if custom_model_name:
        training_run_id = custom_model_name
    else:
        training_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
else:
    # Jeśli użytkownik podał nazwę bez ścieżki, przeszukujemy folder MODELS
    if not ("/" in model_mode or "\\" in model_mode):
        model_mode = os.path.join(MODELS_FOLDER, model_mode)
    base = os.path.basename(model_mode)
    ep = extract_episode(base)
    if ep is not None:
        training_run_id = base.split("_ep")[0].replace("snake_model_", "")
    else:
        training_run_id = "default"

# =============================================================================
# Funkcja should_visualize - oparta na parametrze visualize_rate
# =============================================================================
def should_visualize(episode, rate_str):
    """
    Decyduje, czy wizualizować epizod.
    Jeśli rate_str == "0", wizualizuj każdy epizod.
    Jeśli rate_str zaczyna się od '0' (np. "020"), interpretuj jako dokładną wartość (np. co 20 epizod).
    W przeciwnym razie:
       - wizualizuj co podany % aktualnego epizodu
    """
    if rate_str == "0":
        return True
    if rate_str[0] == "0":
        freq = int(rate_str)
    else:
        if episode % rate_str == 0 or episode == 0:
            freq = int(rate_str*(episode/100))

    return (episode % freq == 0)

# =============================================================================
# Funkcja save_model - zapisuje modele w folderze MODELS
# =============================================================================
def save_model(model, episode, run_id, path_prefix="snake_model", max_models=MAX_MODELS_TO_SAVE):
    """
    Zapisuje model do pliku: {MODELS_FOLDER}/{path_prefix}_{run_id}_ep{current_episode}.pth.
    Jeśli run_id zawiera numer epizodu, kontynuujemy numerację; w przeciwnym razie zaczynamy od 1.
    """
    try:
        base, ep_str = run_id.split("_ep")
        base_run = base
        last_episode = int(ep_str)
        current_episode = last_episode + episode
        new_run_id = f"{base_run}_ep{current_episode}"
    except Exception:
        current_episode = episode
        new_run_id = f"{run_id}_ep{current_episode}"
    filename = os.path.join(MODELS_FOLDER, f"{path_prefix}_{new_run_id}.pth")
    torch.save(model.state_dict(), filename)
    print(f"Model zapisany: {filename}")
    files = glob.glob(os.path.join(MODELS_FOLDER, f"{path_prefix}_{run_id}_ep*.pth"))
    files_sorted = sorted(files, key=lambda x: extract_episode(x) or 0)
    if len(files_sorted) > max_models:
        removed = files_sorted[0]
        os.remove(removed)
        print(f"Usunięto najstarszy model: {removed}")

# =============================================================================
# Funkcje pomocnicze: sterowanie, kolizje, odległość, obiekty
# =============================================================================
def turn_left(direction):
    """Zwraca nowy kierunek po skręcie w lewo."""
    if direction == (0, -1): return (-1, 0)
    elif direction == (-1, 0): return (0, 1)
    elif direction == (0, 1): return (1, 0)
    elif direction == (1, 0): return (0, -1)

def turn_right(direction):
    """Zwraca nowy kierunek po skręcie w prawo."""
    if direction == (0, -1): return (1, 0)
    elif direction == (1, 0): return (0, 1)
    elif direction == (0, 1): return (-1, 0)
    elif direction == (-1, 0): return (0, -1)

def is_collision(pos, snake_body):
    """Sprawdza, czy pozycja koliduje ze ścianą lub ciałem."""
    x, y = pos
    return not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT) or pos in snake_body

def get_distance_in_direction(pos, direction, snake_body):
    """Oblicza znormalizowany dystans od pozycji do przeszkody w danym kierunku."""
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
    """
    Identyfikuje obiekt w danym kierunku.
    Zwraca:
      0 - brak obiektu,
      1 - przeszkoda (ściana lub ciało),
      2 - jedzenie.
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
    return 0

def rotate_vector(vector, angle):
    """Obraca wektor o zadany kąt (w stopniach)."""
    rad = math.radians(angle)
    return (vector[0]*math.cos(rad) - vector[1]*math.sin(rad),
            vector[0]*math.sin(rad) + vector[1]*math.cos(rad))

# =============================================================================
# Funkcja compute_available_space
# =============================================================================
def compute_available_space(head, snake_body):
    """
    Oblicza procent dostępnych komórek osiągalnych z pozycji head.
    Używa algorytmu flood-fill.
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
        for n in [(cell[0]+1, cell[1]), (cell[0]-1, cell[1]), (cell[0], cell[1]+1), (cell[0], cell[1]-1)]:
            if (0 <= n[0] < GRID_WIDTH and 0 <= n[1] < GRID_HEIGHT) and (n not in obstacles) and (n not in visited):
                queue.append(n)
    return count / total_cells

# =============================================================================
# Funkcje pomocnicze dla Lookahead
# =============================================================================
def clone_snake(snake):
    """Tworzy kopię obiektu Snake."""
    new_snake = Snake(snake.body[0], snake.direction)
    new_snake.body = list(snake.body)
    new_snake.alive = snake.alive
    new_snake.growing = snake.growing
    return new_snake

def simulate_move(snake, food, action):
    """
    Tworzy klon węża, zmienia kierunek zgodnie z akcją (0 - brak, 1 - lewo, 2 - prawo),
    wykonuje jeden ruch i zwraca: nowego węża, nowe jedzenie, flagę zakończenia,
    natychmiastową nagrodę oraz typ kolizji (jeśli wystąpi).
    """
    new_snake = clone_snake(snake)
    if action == 1:
        new_snake.direction = turn_left(new_snake.direction)
    elif action == 2:
        new_snake.direction = turn_right(new_snake.direction)
    new_food, reward, done, collision_type = step(new_snake, food)
    return new_snake, new_food, done, reward, collision_type

def rollout_value(snake, food, depth, prev_avail):
    """
    Rekurencyjnie symuluje ruchy (lookahead) do zadanej głębokości.
    Dyskontuje przyszłe nagrody przy użyciu GAMMA.
    Jeśli wąż ginie, zwraca natychmiastową nagrodę.
    """
    if not snake.alive:
        return -50
    if depth == 0:
        return compute_available_space(snake.body[0], snake.body)
    best = -float('inf')
    for action in [0, 1, 2]:
        new_snake, new_food, done, immediate_reward, _ = simulate_move(snake, food, action)
        if done:
            cumulative = immediate_reward
        else:
            new_avail = compute_available_space(new_snake.body[0], new_snake.body)
            avail_bonus = SURVIVAL_FACTOR * (new_avail - prev_avail)
            cumulative = immediate_reward + avail_bonus + GAMMA * rollout_value(new_snake, new_food, depth - 1, new_avail)
        best = max(best, cumulative)
    return best

# =============================================================================
# Funkcja get_state
# =============================================================================
def get_state(snake, food):
    """
    Buduje wektor wejściowy opisujący stan gry bez mapy i bez wektora "do tyłu".
    Składa się z:
      1. Relatywnej pozycji jedzenia (2 wartości),
      2. Kierunku ruchu jako one-hot (4 wartości),
      3. Globalnej dostępności przestrzeni (1 wartość),
      4. Relatywnych pozycji pierwszych dwóch segmentów ciała (4 wartości),
      5. Lokalnej analizy otoczenia dla 7 względnych kierunków (7 x 4 = 28 wartości).
    Łącznie: 2 + 4 + 1 + 4 + 28 = 39 elementów.
    """
    features = []
    head = snake.body[0]
    # 1. Pozycja jedzenia
    dx = (food[0] - head[0]) / GRID_WIDTH
    dy = (food[1] - head[1]) / GRID_HEIGHT
    features.extend([dx, dy])
    # 2. Kierunek ruchu jako one-hot
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
    # 3. Globalna dostępność przestrzeni
    avail = compute_available_space(head, snake.body)
    features.append(avail)
    # 4. Pozycje pierwszych dwóch segmentów ciała
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
    # 5. Lokalna analiza otoczenia dla 7 względnych kierunków
    head_angle = math.degrees(math.atan2(snake.direction[1], snake.direction[0])) if snake.direction != (0,0) else 0
    rel_angles = [0, 45, 90, 135, 225, 270, 315]
    for r in rel_angles:
        abs_angle = head_angle + r
        d = (math.cos(math.radians(abs_angle)), math.sin(math.radians(abs_angle)))
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
    return np.array(features, dtype=float)

# =============================================================================
# Funkcja print_prev_state_no_map
# =============================================================================
def print_prev_state_no_map(vector):
    """
    Wypisuje etykietowane wartości wejścia (bez mapy – 39 elementów) w czytelnym formacie.
    Podstawowe informacje (indeksy 0-10):
      0: Food dx, 1: Food dy,
      2: Dir Up, 3: Dir Down, 4: Dir Left, 5: Dir Right,
      6: Global Avail,
      7: Seg1 dx, 8: Seg1 dy,
      9: Seg2 dx, 10: Seg2 dy.
    Następnie 7 bloków (dla względnych kątów: 0°,45°,90°,135°,225°,270°,315°), każdy po 4 elementy:
         - 1: Distance,
         - 2-4: Object (None, Ciało, Jedzenie)
    """
    basic_labels = [
        "Food dx", "Food dy",
        "Dir Up", "Dir Down", "Dir Left", "Dir Right",
        "Global Avail",
        "Seg1 dx", "Seg1 dy",
        "Seg2 dx", "Seg2 dy"
    ]
    print("Podstawowe informacje:")
    for i, lab in enumerate(basic_labels):
        print(f"{lab:12s}: {vector[i]:6.3f}")
    rel_angles = [0, 45, 90, 135, 225, 270, 315]
    print("\nLokalna analiza otoczenia (kąty względne):")
    for i, angle in enumerate(rel_angles):
        base = 11 + i * 4
        dist = vector[base]
        obj_vec = vector[base+1:base+4]
        if np.array_equal(obj_vec, [1, 0, 0]):
            obj = "None"
        elif np.array_equal(obj_vec, [0, 1, 0]):
            obj = "Ciało"
        elif np.array_equal(obj_vec, [0, 0, 1]):
            obj = "Jedzenie"
        else:
            obj = "Unknown"
        print(f"  Kąt {angle:3d}°: Distance = {dist:6.3f}, Object = {obj}")

# =============================================================================
# Definicja sieci neuronowej AdvancedQNet
# =============================================================================
class AdvancedQNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """
        Tworzy model sieci neuronowej.
        input_size - liczba wejść,
        hidden_sizes - lista rozmiarów warstw ukrytych,
        output_size - liczba akcji.
        """
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
        """Przekazuje wejście przez sieć i zwraca wyjście."""
        return self.model(x)

# =============================================================================
# Klasa agenta DQN
# =============================================================================
class DQNAgent:
    def __init__(self, state_size, hidden_sizes, output_size, lr=LR, gamma=GAMMA,
                 epsilon_start=EPSILON_START, epsilon_min=EPSILON_MIN,
                 epsilon_decay=EPSILON_DECAY, batch_size=BATCH_SIZE):
        """
        Inicjalizuje agenta DQN.
        state_size - rozmiar wektora wejściowego,
        hidden_sizes - rozmiary warstw ukrytych,
        output_size - liczba akcji.
        """
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
        """
        Wybiera akcję na podstawie stanu, łącząc Q-wartości z sieci z bonusami z lookahead.
        """
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.output_size - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze().cpu().numpy()
        final_values = np.copy(q_values)
        current_avail = compute_available_space(snake.body[0], snake.body)
        for action in range(self.output_size):
            new_snake, new_food, done, immediate_reward, _ = simulate_move(snake, food, action)
            if done:
                rollout_bonus = immediate_reward
            else:
                new_avail = compute_available_space(new_snake.body[0], new_snake.body)
                avail_bonus = SURVIVAL_FACTOR * (new_avail - current_avail)
                rollout_bonus = immediate_reward + avail_bonus + GAMMA * rollout_value(new_snake, new_food, LOOKAHEAD_DEPTH - 1, new_avail)
            final_values[action] += LOOKAHEAD_WEIGHT * rollout_bonus
        return int(np.argmax(final_values))
    def remember(self, state, action, reward, next_state, done):
        """Dodaje doświadczenie do pamięci."""
        self.memory.append((state, action, reward, next_state, done))
    def train(self):
        """Trenuje sieć na partii doświadczeń."""
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

# =============================================================================
# Klasa reprezentująca węża
# =============================================================================
class Snake:
    def __init__(self, pos, direction):
        """Inicjalizuje węża z początkową pozycją i kierunkiem."""
        self.body = [pos]
        self.direction = direction
        self.alive = True
        self.new_head = None
        self.growing = False

# =============================================================================
# Funkcje środowiskowe
# =============================================================================
def generate_food(snake):
    """Generuje nową pozycję jedzenia, która nie koliduje z ciałem węża."""
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake.body:
            return pos

def reset_game():
    """Resetuje grę – losuje pozycję i kierunek węża oraz generuje jedzenie."""
    pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
    direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
    snake = Snake(pos, direction)
    food = generate_food(snake)
    return snake, food

def step(snake, food):
    """
    Wykonuje jeden ruch.
    Aktualizuje stan węża i jedzenia.
    Zwraca: nowe jedzenie, nagrodę, flagę zakończenia (done),
            oraz typ kolizji ("wall", "body" lub None).
    """
    head = snake.body[0]
    snake.new_head = (head[0] + snake.direction[0], head[1] + snake.direction[1])
    if not (0 <= snake.new_head[0] < GRID_WIDTH and 0 <= snake.new_head[1] < GRID_HEIGHT):
        snake.alive = False
        return food, -15, True, "wall"
    if snake.new_head in snake.body:
        snake.alive = False
        return food, -15, True, "body"
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
    return food, reward, False, None

def draw_board(snake, food, episode, score):
    """Rysuje planszę, węża i jedzenie przy użyciu Pygame."""
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

# =============================================================================
# Inicjalizacja Pygame
# =============================================================================
if ALWAYS_VISUALIZE:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Snake RL – Trening pojedynczego węża")
    clock = pygame.time.Clock()

# =============================================================================
# Inicjalizacja agenta
# =============================================================================
state_size = 39            # Wejście: 39 elementów (bez mapy)
hidden_sizes = [512, 256]    # Warstwy ukryte
output_size = 3            # 3 akcje: 0 - prosto, 1 - lewo, 2 - prawo
agent = DQNAgent(state_size, hidden_sizes, output_size)

# Jeśli model_mode nie jest "new", wczytujemy model z folderu MODELS, jeśli nie podano pełnej ścieżki.
if model_mode != "new":
    if not ("/" in model_mode or "\\" in model_mode):
        model_mode = os.path.join(MODELS_FOLDER, model_mode)
    if model_mode == "latest":
        def get_latest_model():
            files = glob.glob(os.path.join(MODELS_FOLDER, "*.pth"))
            return max(files, key=os.path.getmtime) if files else None
        latest = get_latest_model()
        if latest is not None:
            print(f"Ładowanie najnowszego modelu: {latest}")
            agent.model.load_state_dict(torch.load(latest))
        else:
            print("Nie znaleziono modelu 'latest'. Trening od nowa.")
    else:
        if os.path.exists(model_mode):
            print(f"Ładowanie modelu z pliku: {model_mode}")
            agent.model.load_state_dict(torch.load(model_mode))
        else:
            print("Podany plik modelu nie istnieje. Trening od nowa.")

# =============================================================================
# Nowa funkcja should_visualize oparta o parametr visualize_rate
# =============================================================================
def should_visualize_rate(episode, rate_str):
    """
    Zwraca True, gdy epizod ma być wizualizowany.
    Jeśli rate_str == "0", wizualizuj każdy epizod.
    Jeśli rate_str zaczyna się od '0', interpretuj jako dokładną wartość (np. "020" -> co 20 epizod).
    W przeciwnym razie:
       - dla ep < 20: co 5 epizod,
       - dla ep < 100: co int(rate_str) epizod,
       - dla ep < 300: co 50,
       - dla ep < 1000: co 100,
       - inaczej: co 200.
    """
    if rate_str == "0":
        return True
    if rate_str[0] == "0":
        freq = int(rate_str)
        return (episode % freq == 0)
    else:
        if episode < 20:
            freq = 5
        elif episode < 100:
            freq = int(rate_str)
        elif episode < 300:
            freq = 50
        elif episode < 1000:
            freq = 100
        else:
            freq = 200
        return (episode % freq == 0)

# =============================================================================
# Główna pętla treningowa
# =============================================================================
mstep_helper = 0
for episode in range(1, MAX_EPISODES + 1):
    # Ustalanie numeru epizodu na podstawie run_id, jeśli zawiera "_epXXX"
    try:
        base, ep_str = training_run_id.split("_ep")
        start_episode = int(ep_str) + 1
    except Exception:
        start_episode = 1
    current_episode = start_episode + episode - 1

    vis = should_visualize_rate(current_episode, args.visualize_rate) if ALWAYS_VISUALIZE else False
    snake, food = reset_game()  # Reset gry
    step_count = 0
    done = False
    episode_score = 0.0

    state = get_state(snake, food)
    # Zachowujemy stan wejściowy (bez mapy) – 39 elementów
    prev_state_no_map = state[:39]
    old_head = snake.body[0]
    avail_old = compute_available_space(old_head, snake.body)
    
    last_action = None

    while not done and step_count < mMAX_STEPS:
        step_count += 1
        if ALWAYS_VISUALIZE:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        # Zachowujemy stan (bez mapy) przed ruchem
        prev_state_no_map = state[:39]
        action = agent.get_action(state, snake, food)
        last_action = action
        if action == 1:
            snake.direction = turn_left(snake.direction)
        elif action == 2:
            snake.direction = turn_right(snake.direction)
        food, reward, done, collision_type = step(snake, food)
        
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

        if ALWAYS_VISUALIZE:
            draw_board(snake, food, current_episode, episode_score)
            pygame.time.delay(30)
    
    # Po zakończeniu epizodu, jeśli kolizja wystąpiła, wypisujemy typ kolizji oraz ostatni ruch
    if done and collision_type is not None:
        collision_mapping = {"body": "ciało", "wall": "ściana"}
        move_mapping = {0: "prosto", 1: "lewo", 2: "prawo"}
        print("\nEpizod zakończony kolizją z:", collision_mapping.get(collision_type, "nieznane"))
        print("Ostatni ruch wykonany:", move_mapping.get(last_action, "nieznany"))
        print("Stan wejściowy (bez mapy) z jednego kroku wcześniej:")
        print_prev_state_no_map(prev_state_no_map)
    
    if episode % 5 == 0:
        mstep_helper = 0 
    else:
        mstep_helper += episode_score

    if mstep_helper > mMAX_STEPS * 4:
        mMAX_STEPS = int(mMAX_STEPS * 1.5)

    print(f"Epizod {current_episode}, steps: {step_count}, Score: {episode_score:.2f}, mMAX_STEPS: {mMAX_STEPS}, mstep_helper: {mstep_helper:.0f}")
    
    if episode % SAVE_MODEL_EVERY == 0:
        save_model(agent.model, current_episode, training_run_id)
        
    if ALWAYS_VISUALIZE:
        pygame.time.delay(500)
if ALWAYS_VISUALIZE:
    pygame.quit()
sys.exit()
