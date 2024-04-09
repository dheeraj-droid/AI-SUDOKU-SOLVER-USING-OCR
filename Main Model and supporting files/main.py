# Disable tensorflow warning messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from solve_sudoku import *
from img_txt import *
import subprocess
import sys
# Read Sudoku puzzle from file
def read_sudoku_from_file(file_path):
    sudoku = []
    with open(file_path, 'r') as file:
        for line in file:
            row = [int(num) for num in line.strip().split()]
            sudoku.append(row)
    return sudoku

subprocess.run(['python', 'img_txt.py',sys.argv[1],sys.argv[2]])

# Read 'cell_locs.txt' file
with open('cell_positions.txt', 'r') as file:
    cell_locs_lines = file.readlines()

# Parse tuples from 'cell_locs.txt'
cell_locs_tuples = []
for line in cell_locs_lines:
    # Split the line by parentheses to separate tuples
    tuples_as_strings = line.strip().split(') (')
    
    # Iterate over each tuple string and parse it
    for tuple_str in tuples_as_strings:
        # Add parentheses back to the tuple string
        tuple_str = tuple_str.strip()
        if not tuple_str.startswith('('):
            tuple_str = '(' + tuple_str
        if not tuple_str.endswith(')'):
            tuple_str = tuple_str + ')'
        
        # Split the tuple string by comma
        numbers_as_strings = tuple_str[1:-1].split(',')
        
        # Convert each number string to integer and create a tuple
        cell_locs_tuples.append(tuple(int(num_str.strip()) for num_str in numbers_as_strings))

# Now, cell_locs_tuples contains the parsed tuples from 'cell_locs.txt'


# Read 'color_puzzle.txt' file
with open('color_puzzle.txt', 'r') as file:
    color_puzzle_lines = file.readlines()

# Parse tuples from 'color_puzzle.txt'
color_puzzle_tuples = []
for line in color_puzzle_lines:
    # Remove square brackets and split the line by spaces
    numbers_as_strings = line.strip().replace('[', '').replace(']', '').split()
    
    # Combine numbers into tuples of three
    for i in range(0, len(numbers_as_strings), 3):
        # Convert each group of three numbers to integers and create a tuple
        numbers = [int(num_str) for num_str in numbers_as_strings[i:i+3]]
        color_puzzle_tuples.append(list(numbers))

# Now, color_puzzle_tuples contains the parsed tuples from 'color_puzzle.txt'

file_path = 'inp.txt'  # Change this to the path of your Sudoku file
initial_board = read_sudoku_from_file(file_path)
check_board = read_sudoku_from_file(file_path)
count =0 
os.remove(file_path)
# Solve the Sudoku puzzle
solved_sudoku = solve_sudoku_with_backtrack(initial_board)

for i in range(len(initial_board)):
    for j in range(len(initial_board)):
        if(check_board[i][j] == solved_sudoku[i][j]):
            count +=1



if(count == len(initial_board)**2):
    print("Either the input sudoku is incorrect or the  model gave an incorrect sudoku.\n")
else:
    print("Sudoku is Solved")
    for row in solved_sudoku:
        print(row)
    display_solutions_on_image(cell_positions, color_puzzle, solved_sudoku)

os.remove('cell_positions.txt')
os.remove('color_puzzle.txt')