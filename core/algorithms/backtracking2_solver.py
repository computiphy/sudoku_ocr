from core.utils import time_the_func

@time_the_func
def solve(puzzle):
    import copy
    board = copy.deepcopy(puzzle)

    def is_valid(board, row, col, num):
        for i in range(9):
            if board[row][i] == num or board[i][col] == num:
                return False
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == num:
                    return False
        return True

    def backtrack_with_valid_numbers():
        all_num = list(range(1,10))
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    req_list = list(filter(lambda x: x not in board[row], all_num))
                    for num in req_list:
                        if is_valid(board, row, col, num):
                            board[row][col] = num
                            if backtrack_with_valid_numbers():
                                return True
                            board[row][col] = 0
                    return False
        return True

    backtrack_with_valid_numbers()
    return board
