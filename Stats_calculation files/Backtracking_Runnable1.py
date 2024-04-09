import sys
import time
import numpy as np
import matplotlib.pyplot as plt

def isempty(a):
    if a == 0:
        return True
    return False

def check(board,row,col,num):
    #Checking if the number is not in the row
    for i in range(9):
        if num==board[row][i]:
            return False
    #Checking if the number is not in the column
    for i in range(9):
        if num == board[i][col]:
            return False
    #Checking if the number is not in the 3*3 subgrid
    row1 = 3 * (row // 3)
    col1 = 3 * (col // 3)
    for i in range(row1, row1 + 3):
        for j in range(col1, col1 + 3):
            if num == board[i][j]:
                return False
    return True

def find_empty_box(board):
    for i in range(9):
        for j in range(9):
            if isempty(board[i][j]):
                return (i,j)
    return None

def back_track(board):
    global IterationsCount
    IterationsCount += 1
    empty_box = find_empty_box(board)
    if not empty_box:
        return True
    r, c = empty_box
    for i in range(1, 10):
        if check(board, r, c, i):
            board[r][c] = i
            if back_track(board):
                return True
            board[r][c] = 0
    return False

InputBoardCounts = [] 
BacktrackingTimes = [] 
BacktrackingIterationss = []

NumberOfSudokuBoardInputs = int(sys.argv[1])
BacktrackingTime = []
BacktrackingIterations = []
InputBoardCount = []

file_name = "MyFile.txt"

file_handle = open(file_name, "r")
DifficultyLevels = ['easy','medium','hard','extreme']

for level in DifficultyLevels:
    for i in range(NumberOfSudokuBoardInputs):
        board=[]
        IterationsCount = 0
        CountZeros = 0
        for i in range(9):
            list1 = []
            line1 = file_handle.readline()
            for i in range(0,len(line1),2):
                if int(line1[i]) == 0:
                   CountZeros += 1
                list1.append(int(line1[i]))
            board.append(list1)   
 
        start = time.time()                                                      
        back_track(board)
        end = time.time()
   
        InputBoardCount.append(81-CountZeros)
        BacktrackingTime.append(end-start)
        BacktrackingIterations.append(IterationsCount)
      
    DifficultyLevel = level

    print("The mean Run Time Taken for ",NumberOfSudokuBoardInputs,"Sudoku boards Of Difficulty Level ",DifficultyLevel,"Is: ",np.mean(BacktrackingTime))
    print("The mean Backtracking Iteration Count Taken for",NumberOfSudokuBoardInputs,"Sudoku boards Of Difficulty Level ",DifficultyLevel,"Is: ",np.mean(BacktrackingIterations))
    print("The mean Inputs That are in the Board Given ",NumberOfSudokuBoardInputs,"Sudoku boards Of Difficulty Level ",DifficultyLevel,"Is: ",np.mean(InputBoardCount))

    InputBoardCounts.extend(InputBoardCount)
    BacktrackingTimes.extend(BacktrackingTime)
    BacktrackingIterationss.extend(BacktrackingIterations)

    InputBoardCount = []
    BacktrackingTime = []
    BacktrackingIterations = []

file_handle.close()
    
# Creating subplots
plt.figure(figsize=(10, 6))

# Subplot 1
plt.subplot(3, 1, 1)
plt.hist(np.array(InputBoardCounts), label="NumberOfSudokuInputs")
plt.title("Number of Sudoku Inputs")
plt.legend()

# Subplot 2
plt.subplot(3, 1, 2)
plt.hist(np.array(BacktrackingTimes), label="Backtracking-Run-Time")
plt.title("Backtracking Run Time")
plt.legend()

# Subplot 3
plt.subplot(3, 1, 3)
plt.hist(np.array(BacktrackingIterationss), label="Backtracking-Iterations-Count")
plt.title("Backtracking Iterations Count")
plt.legend()

# Adjust layout
plt.tight_layout()
plt.legend()

plt.show()


 
