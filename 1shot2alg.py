import pygame
import random
import sys
from collections import deque

# ----- Ustawienia gry -----
CELL_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 10

# ----- Inicjalizacja Pygame -----
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Autonomiczne rywalizujące węże")
clock = pygame.time.Clock()

# ----- Pomocnicza funkcja flood fill -----
def flood_fill_area(start, obstacles):
    """
    Zwraca liczbę osiągalnych komórek zaczynając od pozycji 'start',
    omijając komórki znajdujące się w zbiorze 'obstacles'.
    """
    visited = set()
    queue = deque([start])
    area = 0
    while queue:
        pos = queue.popleft()
        if pos in visited:
            continue
        visited.add(pos)
        area += 1
        # Rozpatrujemy ruchy w 4 kierunkach
        for d in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            new_pos = (pos[0] + d[0], pos[1] + d[1])
            if (0 <= new_pos[0] < GRID_WIDTH and 0 <= new_pos[1] < GRID_HEIGHT and 
                new_pos not in obstacles and new_pos not in visited):
                queue.append(new_pos)
    return area

# ----- Definicja klasy Snake -----
class Snake:
    def __init__(self, body, direction, color, name):
        """
        body: lista krotek (x,y). Pierwszy element to głowa.
        direction: krotka (dx, dy)
        color: krotka (R,G,B)
        name: nazwa węża
        """
        self.body = body[:]  # kopia listy
        self.direction = direction
        self.color = color
        self.name = name
        self.alive = True
        self.new_head = None  # nowa pozycja głowy obliczana przy każdej aktualizacji
        self.growing = False  # True, gdy wąż zjada jedzenie
        self.score = 0       # liczba zjedzonych jedzeń

    def choose_move(self, food, snakes):
        """
        Wybiera ruch na podstawie dostępnych kierunków.
        Dla każdego ruchu sprawdzamy, czy jest on bezpieczny oraz
        wykonujemy flood fill, aby ocenić ilość wolnej przestrzeni.
        Heurystyka: free_area - manhattan_distance (+ bonus, jeśli jedzenie).
        
        Możesz tutaj zastąpić logikę heurystyczną modelem deep learning (np. DQN),
        korzystając z bibliotek takich jak PyTorch lub TensorFlow.
        """
        possible_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        # Nie pozwalamy na bezpośredni ruch przeciwny do obecnego kierunku.
        reverse = (-self.direction[0], -self.direction[1])
        allowed_dirs = [d for d in possible_directions if d != reverse]

        best_move = None
        best_heuristic = -float('inf')

        for d in allowed_dirs:
            new_head = (self.body[0][0] + d[0], self.body[0][1] + d[1])
            # Sprawdzenie granic planszy
            if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
                continue

            # Sprawdzenie kolizji z ciałami węży (dla bezpieczeństwa)
            collision = False
            for snake in snakes:
                if snake == self:
                    # Pozwalamy wejście na ostatnią komórkę, bo zostanie zwolniona, jeśli nie rośniemy
                    if new_head in snake.body[:-1]:
                        collision = True
                        break
                else:
                    if new_head in snake.body:
                        collision = True
                        break
            if collision:
                continue

            # Ustalamy, czy ruch spowoduje zjedzenie jedzenia
            candidate_eating = (new_head == food)

            # Budujemy zbiór przeszkód dla flood fill.
            obstacles = set()
            for snake in snakes:
                if snake == self:
                    if candidate_eating:
                        # Jeśli jemy, cały wąż zostaje na planszy
                        obstacles.update(snake.body)
                    else:
                        # Jeśli nie jemy, ostatnia komórka (ogon) zostanie zwolniona
                        obstacles.update(snake.body[:-1])
                else:
                    obstacles.update(snake.body)

            free_area = flood_fill_area(new_head, obstacles)
            # Obliczamy odległość Manhattan do jedzenia
            distance = abs(new_head[0] - food[0]) + abs(new_head[1] - food[1])
            # Heurystyka: im większa wolna przestrzeń i im bliżej jedzenia, tym lepiej.
            heuristic = free_area - distance
            if candidate_eating:
                heuristic += 50  # bonus za możliwość zjedzenia jedzenia

            if heuristic > best_heuristic:
                best_heuristic = heuristic
                best_move = d

        if best_move is None:
            # Gdy nie znaleziono bezpiecznego ruchu, wybieramy losowy (choć może być śmiertelny)
            best_move = random.choice(allowed_dirs)
        return best_move

# ----- Funkcje pomocnicze -----
def generate_food(snakes):
    """Generuje pozycję jedzenia, która nie koliduje z ciałem żadnego węża."""
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        collision = False
        for snake in snakes:
            if pos in snake.body:
                collision = True
                break
        if not collision:
            return pos

def draw_scores(snakes):
    """Rysuje wyniki obu węży w lewym górnym rogu."""
    font = pygame.font.SysFont("Arial", 20)
    y = 10
    for snake in snakes:
        status = " (martwy)" if not snake.alive else ""
        score_text = f"{snake.name}: {snake.score}{status}"
        text_surface = font.render(score_text, True, snake.color)
        screen.blit(text_surface, (10, y))
        y += 25

def draw_board(snakes, food):
    """Rysuje planszę, węże i jedzenie."""
    screen.fill((0, 0, 0))  # tło czarne

    # Rysowanie linii siatki (opcjonalnie)
    for x in range(0, WINDOW_WIDTH, CELL_SIZE):
        pygame.draw.line(screen, (40, 40, 40), (x, 0), (x, WINDOW_HEIGHT))
    for y in range(0, WINDOW_HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, (40, 40, 40), (0, y), (WINDOW_WIDTH, y))

    # Rysowanie żywych węży
    for snake in snakes:
        if snake.alive:
            for segment in snake.body:
                rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, snake.color, rect)

    # Rysowanie jedzenia (czerwony kwadrat)
    food_rect = pygame.Rect(food[0] * CELL_SIZE, food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    pygame.draw.rect(screen, (255, 0, 0), food_rect)

    # Rysowanie wyników
    draw_scores(snakes)

    pygame.display.flip()

def update_game(snakes, food):
    """
    Aktualizuje stan gry:
      1. Każdy żywy wąż wybiera ruch i oblicza nową głowę.
      2. Oblicza nowe ciało (jeśli zjadł jedzenie – rośnie, w przeciwnym wypadku ogon znika).
      3. Sprawdza kolizje (ze ścianą, samym sobą, innym wężem).
      4. Aktualizuje ciała węży, zwiększa punkty, usuwa martwe węże i generuje nowe jedzenie.
    """
    # --- Krok 1: Każdy żywy wąż wybiera ruch ---
    for snake in snakes:
        if snake.alive:
            move = snake.choose_move(food, snakes)
            snake.direction = move  # aktualizacja kierunku
            snake.new_head = (snake.body[0][0] + move[0], snake.body[0][1] + move[1])
            if snake.new_head == food:
                snake.growing = True
            else:
                snake.growing = False

    # --- Krok 2: Obliczanie nowych ciał ---
    new_bodies = {}
    for snake in snakes:
        if snake.alive:
            if snake.growing:
                new_body = [snake.new_head] + snake.body  # rośnie – nie usuwa ogona
            else:
                new_body = [snake.new_head] + snake.body[:-1]
            new_bodies[snake] = new_body

    # --- Krok 3: Sprawdzanie kolizji ---
    # Kolizja ze ścianą
    for snake in snakes:
        if snake.alive:
            x, y = snake.new_head
            if not (0 <= x < GRID_WIDTH and 0 <= y < GRID_HEIGHT):
                print(f"{snake.name} uderzył w ścianę!")
                snake.alive = False

    # Kolizja z samym sobą
    for snake in snakes:
        if snake.alive:
            if snake.new_head in new_bodies[snake][1:]:
                print(f"{snake.name} zderzył się ze sobą!")
                snake.alive = False

    # Kolizje między wężami
    for snake in snakes:
        if snake.alive:
            for other in snakes:
                if other != snake and other.alive:
                    if snake.new_head in new_bodies[other]:
                        print(f"{snake.name} zderzył się z {other.name}!")
                        snake.alive = False

    # Kolizja głowa-do-głowy
    if len(snakes) >= 2 and all(s.alive for s in snakes):
        if snakes[0].new_head == snakes[1].new_head:
            print("Kolizja głowa-do-głowy! Oba węże giną!")
            for snake in snakes:
                snake.alive = False

    # --- Krok 4: Aktualizacja ciał i punktów ---
    for snake in snakes:
        if snake.alive:
            snake.body = new_bodies[snake]
            if snake.growing:
                snake.score += 1
        else:
            # Gdy wąż ginie – znika z planszy
            snake.body = []

    # Jeśli któremuś wężowi udało się zjeść jedzenie, generujemy nowe
    ate_food = any(s.alive and s.growing for s in snakes)
    if ate_food:
        food = generate_food(snakes)
    return food

# ----- Inicjalizacja węży i jedzenia -----
snakes = []

# Wąż 1 zaczyna w lewym środkowym sektorze, porusza się w prawo.
snake1 = Snake(body=[(GRID_WIDTH // 4, GRID_HEIGHT // 2)],
               direction=(1, 0),
               color=(0, 255, 0),  # zielony
               name="Wąż 1")

# Wąż 2 zaczyna w prawym środkowym sektorze, porusza się w lewo.
snake2 = Snake(body=[(3 * GRID_WIDTH // 4, GRID_HEIGHT // 2)],
               direction=(-1, 0),
               color=(0, 0, 255),  # niebieski
               name="Wąż 2")

snakes.append(snake1)
snakes.append(snake2)

food = generate_food(snakes)

# ----- Główna pętla gry -----
running = True
while running:
    # Obsługa zdarzeń (np. kliknięcie "zamknij")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Aktualizacja stanu gry
    food = update_game(snakes, food)
    # Rysowanie planszy
    draw_board(snakes, food)
    clock.tick(FPS)

pygame.quit()
sys.exit()
