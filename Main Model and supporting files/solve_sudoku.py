def isempty(a):
    return a == 0

def check(board, row, col, num):
    # Checking if the number is not in the row
    for i in range(len(board)):
        if num == board[row][i]:
            return False
    # Checking if the number is not in the column
    for i in range(len(board)):
        if num == board[i][col]:
            return False
    # Checking if the number is not in the subgrid
    region_size = int(len(board) ** 0.5)
    row_start = (row // region_size) * region_size
    col_start = (col // region_size) * region_size
    for i in range(row_start, row_start + region_size):
        for j in range(col_start, col_start + region_size):
            if num == board[i][j]:
                return False
    return True

# Modified find_empty_box function by using MRV heuristic
def find_empty_box(board, possible_values):
    min_values = float('inf')
    min_box = None
    for i in range(len(board)):
        for j in range(len(board[0])):
            if isempty(board[i][j]):
                if len(possible_values[i][j]) < min_values:
                    min_values = len(possible_values[i][j])
                    min_box = (i, j)
    return min_box

# Our recursive backtracking algorithm
def back_track(board, possible_values):
    empty_box = find_empty_box(board, possible_values)
    # If there are no empty boxes then the sudoku is solved.
    if not empty_box:
        return True
    r, c = empty_box
    # Recursively checking which numbers can be fit in the empty boxes.
    for i in possible_values[r][c]:
        if check(board, r, c, i):
            board[r][c] = i
            if back_track(board, possible_values):
                return True
            board[r][c] = 0
    return False

def solve_sudoku_with_backtrack(initial_board):
    # Initialize the possible values for each square
    possible_values = [[[i for i in range(1, len(initial_board) + 1)] for _ in range(len(initial_board))] for _ in range(len(initial_board))]
    
    # Update the possible values when a new value is assigned
    for r in range(len(initial_board)):
        for c in range(len(initial_board[0])):
            if initial_board[r][c] != 0:
                possible_values[r][c] = []
                num = initial_board[r][c]
                for i in range(len(initial_board)):
                    if num in possible_values[r][i]:
                        possible_values[r][i].remove(num)
                    if num in possible_values[i][c]:
                        possible_values[i][c].remove(num)
                region_size = int(len(initial_board) ** 0.5)
                row_start = (r // region_size) * region_size
                col_start = (c // region_size) * region_size
                for i in range(row_start, row_start + region_size):
                    for j in range(col_start, col_start + region_size):
                        if num in possible_values[i][j]:
                            possible_values[i][j].remove(num)
    
    # Solve the Sudoku using backtracking
    back_track(initial_board, possible_values)
    
    return initial_board
