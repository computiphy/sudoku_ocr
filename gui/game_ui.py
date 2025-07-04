import pygame
import json
import os
import time
import subprocess
import importlib.util
from tkinter import filedialog, Tk, simpledialog
from core.image_to_grid import extract_grid_from_image

# Constants
WIDTH, HEIGHT = 540, 850
GRID_SIZE = 9
CELL_SIZE = WIDTH // GRID_SIZE
FONT = None

subprocess.run(["python", "core/puzzle_generator.py", "easy"])
with open("data/puzzles.json") as f:
    data = json.load(f)
    puzzle = data["puzzle"]
    solution = data["solution"]
user_inputs = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

# Colors
WHITE = (255, 255, 255)
BLUE = (57, 255, 20)
BLACK = (20, 20, 20)
DARK_GRAY = (35, 45, 35)
HIGHLIGHT = (200, 200, 0)
BUTTON_BG = (50, 50, 50)
BUTTON_TEXT = (255, 255, 255)
RED = (200, 50, 50)
GREEN = (50, 200, 50)

algorithm_options = ["backtracking", "dlx"]
current_algorithm = "backtracking"
start_time = time.time()
algorithm_dropdown_visible = False

def import_solver_module(name):
    filepath = os.path.join('core', 'algorithms', f'{name}_solver.py')
    spec = importlib.util.spec_from_file_location(f"{name}_solver", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def draw_grid(screen):
    # Draw the standard alternating background grid
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell_color = DARK_GRAY if (row + col) % 2 == 0 else BLACK
            pygame.draw.rect(screen, cell_color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    # Add blue neon glow for 3x3 subgrids
    glow_color = (0, 175, 75)  # Neon green
    glow_thickness = 1

    for i in range(3):
        for j in range(3):
            x = j * 3 * CELL_SIZE
            y = i * 3 * CELL_SIZE
            rect = pygame.Rect(x, y, 3 * CELL_SIZE, 3 * CELL_SIZE)
            pygame.draw.rect(screen, glow_color, rect, glow_thickness)

    # Redraw main thin grid lines
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, WHITE, (x, 0), (x, WIDTH), 1)
    for y in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, WHITE, (0, y), (WIDTH, y), 1)

    # Bold 3x3 lines for added clarity
    for x in range(0, WIDTH + 1, 3 * CELL_SIZE):
        pygame.draw.line(screen, glow_color, (x, 0), (x, WIDTH), 2)
    for y in range(0, WIDTH + 1, 3 * CELL_SIZE):
        pygame.draw.line(screen, glow_color, (0, y), (WIDTH, y), 2)

def draw_numbers(screen, puzzle, user_inputs):
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            num = puzzle[row][col]
            if num != 0:
                text = FONT.render(str(num), True, WHITE)
                screen.blit(text, (col * CELL_SIZE + 20, row * CELL_SIZE + 15))
            elif user_inputs[row][col] != 0:
                text = FONT.render(str(user_inputs[row][col]), True, BLUE)
                screen.blit(text, (col * CELL_SIZE + 20, row * CELL_SIZE + 15))

def draw_selected_cell(screen, selected):
    if selected:
        row, col = selected
        pygame.draw.rect(screen, HIGHLIGHT, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 3)

def draw_check_result(screen, result):
    text = FONT.render(result, True, GREEN if result == "Correct!" else RED)
    screen.blit(text, (400, WIDTH + 10))

def draw_buttons(screen):
    labels = ["New", "Solve", "Upload", f"Algorithm: {current_algorithm}", "Check"]
    for i, label in enumerate(labels):
        x, y, w, h = 20, 550 + i * 40, 200, 30
        pygame.draw.rect(screen, BUTTON_BG, (x, y, w, h))
        FONT_TXT = pygame.font.SysFont("Arial Narrow", 24)
        text = FONT_TXT.render(label, True, BUTTON_TEXT)
        screen.blit(text, (x + 5, y + 5))
    if algorithm_dropdown_visible:
        for i, algo in enumerate(algorithm_options):
            pygame.draw.rect(screen, BUTTON_BG, (240, 710 + 40 * i, 200, 30))
            text = FONT.render(algo, True, BUTTON_TEXT)
            screen.blit(text, (245, 715 + 40 * i))

def draw_timer(screen):
    elapsed = int(time.time() - start_time)
    minutes = elapsed // 60
    seconds = elapsed % 60
    timer_text = FONT.render(f"Time: {minutes:02}:{seconds:02}", True, WHITE)
    screen.blit(timer_text, (340, 790))

def button_clicked(pos, x, y, w, h):
    return x <= pos[0] <= x + w and y <= pos[1] <= y + h

def load_new_puzzle(difficulty=None):
    global puzzle, solution, user_inputs, check_result, start_time
    root = Tk()
    root.withdraw()
    difficulty = simpledialog.askstring("Difficulty", "Choose difficulty: easy, medium, or hard")
    if difficulty not in ["easy", "medium", "hard"]:
        return
    subprocess.run(["python", "core/puzzle_generator.py", difficulty])
    with open("data/puzzles.json") as f:
        data = json.load(f)
        puzzle = data["puzzle"]
        solution = data["solution"]
    user_inputs = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    check_result = ""
    start_time = time.time()

def upload_puzzle():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            puzzle_grid = extract_grid_from_image(file_path)
            return puzzle_grid, puzzle_grid
        except Exception as e:
            print(f"Upload error: {e}")
    return None, None

def main():
    global FONT, selected_cell, user_inputs, check_result, puzzle, solution, current_algorithm, algorithm_dropdown_visible
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sudoku")
    FONT = pygame.font.SysFont("Bauhaus 93", 28)

    clock = pygame.time.Clock()
    selected_cell = None
    check_result = ""

    running = True
    while running:
        screen.fill(BLACK)
        draw_grid(screen)
        draw_selected_cell(screen, selected_cell)
        draw_numbers(screen, puzzle, user_inputs)
        draw_buttons(screen)
        draw_timer(screen)

        if check_result:
            draw_check_result(screen, check_result)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if y < WIDTH:
                    selected_cell = (y // CELL_SIZE, x // CELL_SIZE)
                elif button_clicked((x, y), 20, 550, 200, 30):
                    load_new_puzzle()
                elif button_clicked((x, y), 20, 590, 200, 30):
                    try:
                        solver_module = import_solver_module(current_algorithm)
                        solved = solver_module.solve(puzzle)
                        for r in range(GRID_SIZE):
                            for c in range(GRID_SIZE):
                                if puzzle[r][c] == 0:
                                    user_inputs[r][c] = solved[r][c]
                    except Exception as e:
                        print(f"Solver error: {e}")
                elif button_clicked((x, y), 20, 630, 200, 30):
                    new_puzzle, new_solution = upload_puzzle()
                    if new_puzzle:
                        puzzle = new_puzzle
                        solution = new_solution
                        user_inputs = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
                        check_result = ""
                        start_time = time.time()
                elif button_clicked((x, y), 20, 670, 200, 30):
                    algorithm_dropdown_visible = not algorithm_dropdown_visible
                elif button_clicked((x, y), 20, 710, 200, 30):
                    is_correct = all(
                        user_inputs[r][c] == solution[r][c]
                        for r in range(GRID_SIZE)
                        for c in range(GRID_SIZE)
                        if puzzle[r][c] == 0
                    )
                    check_result = "Correct!" if is_correct else "Incorrect!"
                elif algorithm_dropdown_visible:
                    for i, algo in enumerate(algorithm_options):
                        if button_clicked((x, y), 240, 710 + 40 * i, 200, 30):
                            current_algorithm = algo
                            algorithm_dropdown_visible = False

            if event.type == pygame.KEYDOWN and selected_cell:
                row, col = selected_cell
                if event.key == pygame.K_UP:
                    selected_cell = ((row - 1) % GRID_SIZE, col)
                elif event.key == pygame.K_DOWN:
                    selected_cell = ((row + 1) % GRID_SIZE, col)
                elif event.key == pygame.K_LEFT:
                    selected_cell = (row, (col - 1) % GRID_SIZE)
                elif event.key == pygame.K_RIGHT:
                    selected_cell = (row, (col + 1) % GRID_SIZE)
                elif puzzle[row][col] == 0:
                    if event.unicode in "123456789":
                        user_inputs[row][col] = int(event.unicode)
                    if event.key in [pygame.K_BACKSPACE, pygame.K_DELETE]:
                        user_inputs[row][col] = 0

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
