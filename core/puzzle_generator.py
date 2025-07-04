import random
import json
import os

GRID_SIZE = 9
BOX_SIZE = 3

class SudokuGenerator:
    def __init__(self):
        self.board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    def is_valid(self, row, col, num):
        for i in range(GRID_SIZE):
            if self.board[row][i] == num or self.board[i][col] == num:
                return False

        start_row, start_col = row - row % BOX_SIZE, col - col % BOX_SIZE
        for i in range(BOX_SIZE):
            for j in range(BOX_SIZE):
                if self.board[start_row + i][start_col + j] == num:
                    return False
        return True

    def fill_board(self):
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.board[row][col] == 0:
                    numbers = list(range(1, GRID_SIZE + 1))
                    random.shuffle(numbers)
                    for num in numbers:
                        if self.is_valid(row, col, num):
                            self.board[row][col] = num
                            if self.fill_board():
                                return True
                            self.board[row][col] = 0
                    return False
        return True

    def remove_cells(self, difficulty='medium'):
        # Easy: 35 filled, Medium: 30, Hard: 25
        clues = {'easy': 35, 'medium': 30, 'hard': 25}
        cells_to_keep = clues.get(difficulty, 30)
        cells_to_remove = GRID_SIZE * GRID_SIZE - cells_to_keep

        attempts = 0
        while attempts < cells_to_remove:
            row = random.randint(0, 8)
            col = random.randint(0, 8)
            if self.board[row][col] != 0:
                backup = self.board[row][col]
                self.board[row][col] = 0
                attempts += 1

    def generate_puzzle(self, difficulty='medium'):
        self.board = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.fill_board()
        solution = [row[:] for row in self.board]
        self.remove_cells(difficulty)
        puzzle = [row[:] for row in self.board]
        return puzzle, solution

    def save_to_json(self, puzzle, solution, difficulty, file_path='data/puzzles.json'):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data = {
            'difficulty': difficulty,
            'puzzle': puzzle,
            'solution': solution
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    generator = SudokuGenerator()
    puzzle, solution = generator.generate_puzzle('medium')
    generator.save_to_json(puzzle, solution, 'medium')
    print("Puzzle and solution saved to JSON.")
