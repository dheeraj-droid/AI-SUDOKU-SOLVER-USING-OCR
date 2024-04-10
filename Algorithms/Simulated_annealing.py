import time
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

print()
print()
print()


max_stagnation_iterations = 5000
max_restarts = 100 
initial_temp1 = 1.0
initial_temp = 1.0
cooling_rate = 0.001
stopping_temp = 0.0001
'''
#It is Just my Reference Code to know like to know like the probability of Finding a Bad Neighbour
count = 0
while initial_temp1 > stopping_temp:
      initial_temp1 = initial_temp1 * (1-cooling_rate)
      count = count + 1

input = int(sys.argv[2])
max_stagnation_iterations = count/input
'''         
print("Maximum Stagnation Iterations are",max_stagnation_iterations)
print("Maximum Restarts are",max_restarts)
print("Temperatures and Cooling Ranges are",initial_temp,stopping_temp,cooling_rate)
print("Maximum Stagnation Count ",max_stagnation_iterations)

RunTimes = []
CostValuess = []
ReasonForRestart = []
Restartss = []
Iterationss = []

def costfunction(array):
    cost = 0
    for i in range(size):
        row_unique_count = len(set(array[i]))  
        col_unique_count = len(set(array[:, i]))  
        cost += (size - row_unique_count) + (size - col_unique_count)
    for i in range(0, size, 3):
        for j in range(0, size, 3):
            block = array[i:i+3, j:j+3]  
            block_unique_count = len(set(block.flatten()))  
            cost += (size - block_unique_count)
    return cost


def generate_neighbor(sudoku, initial_sudoku):
    NeighbourRunTime = 0
    while True:
        block_row = 3 * random.randint(0, 2)
        block_col = 3 * random.randint(0, 2)
        cell1_row = block_row + random.randint(0, 2)
        cell1_col = block_col + random.randint(0, 2)
        cell2_row = block_row + random.randint(0, 2)
        cell2_col = block_col + random.randint(0, 2)
        if ((cell1_row != cell2_row or cell1_col != cell2_col) and initial_sudoku[cell1_row][cell1_col] == 0 and initial_sudoku[cell2_row][cell2_col] == 0):
            sudoku[cell1_row][cell1_col], sudoku[cell2_row][cell2_col] = sudoku[cell2_row][cell2_col],sudoku[cell1_row][cell1_col]
            break;
    return sudoku


def generate_initial_solution(sudoku):
    blocks = np.sqrt(size).astype(int)
    initial_solution = np.copy(sudoku)
    for i in range(0, size, blocks):
        for j in range(0, size, blocks):
            block = initial_solution[i:i + blocks, j:j + blocks]
            missing_values = set(range(1, size + 1)) - set(block.flatten())
            missing_values = list(missing_values)
            random.shuffle(missing_values)
            for k in range(blocks):
                for l in range(blocks):
                    if block[k, l] == 0:
                        block[k, l] = missing_values.pop()
    return initial_solution


def simulated_annealing(sudoku, initial_temp, cooling_rate, stopping_temp, max_stagnation_iterations, max_restarts):
    best_solution = None
    best_cost = float('inf')
    stagnation_counter = 0
    restarts = 0
    iterations = 0
    temperature = initial_temp
    current_solution = generate_initial_solution(sudoku)
    current_cost = costfunction(current_solution)
    #costs = []  
    #temperatures = []  
    #iterations_list = []
    while restarts < max_restarts:
        if best_cost == 0:
            break
        while temperature > stopping_temp:
            neighbor = generate_neighbor(np.copy(current_solution), sudoku)
            neighbor_cost = costfunction(neighbor)
            cost_diff = neighbor_cost - current_cost
            if cost_diff < 0:
                current_solution, current_cost = neighbor, neighbor_cost
                stagnation_counter = 0
            else:
                if random.uniform(0, 1) < math.exp(-cost_diff / temperature):
                    current_solution, current_cost = neighbor, neighbor_cost
                stagnation_counter += 1
            if current_cost < best_cost:
                best_solution, best_cost = current_solution, current_cost
            temperature *= (1 - cooling_rate)
            iterations += 1
            if stagnation_counter >= max_stagnation_iterations:
                break
        if stagnation_counter >= max_stagnation_iterations or temperature <= stopping_temp:
            if stagnation_counter >= max_stagnation_iterations :
               ReasonForRestart.append(1)
            else:
               ReasonForRestart.append(-1)
            current_solution = generate_initial_solution(sudoku)
            current_cost = costfunction(current_solution)
            temperature = initial_temp
            stagnation_counter = 0
            restarts += 1
        if restarts >= max_restarts:
            break
    Iterations.append(iterations)
    Restarts.append(restarts)
    return best_solution



if __name__ == "__main__":
  file_name = "MyFile.txt"
  file_handle = open(file_name, "r")
  for level in ["easy","medium","hard","extreme"]:
     RunTime = []
     CostValues = []
     Restarts = []
     Iterations = []
     for i in range(int(sys.argv[1])):
         board=[]  
         for i in range(9):
             list1 = []
             line1 = file_handle.readline()
             for i in range(0,len(line1),2):
                 list1.append(int(line1[i]))
             board.append(list1)
         size=9        
         start = time.time()
         solution = simulated_annealing(board, initial_temp, cooling_rate, stopping_temp, max_stagnation_iterations, max_restarts)     
         end = time.time()

         RunTime.append(end-start)
         CostValues.append(costfunction(solution))

     print("The mean run time of ",sys.argv[1],level,"sudokus is :",np.mean(RunTime))
     print("The mean Cost Value of ",sys.argv[1],level,"sudokus is :",np.mean(CostValues))
     ZeroCostCount = 0
     for i in range(len(CostValues)):
         if CostValues[i] == 0:
            ZeroCostCount += 1
     print("The Accuracy of ",sys.argv[1],level,"sudokus is :",(100*ZeroCostCount)/len(CostValues))
     print("Average Restarts of",sys.argv[1],level,"sudokus is :",np.mean(Restarts))
     print("Average Iterations of",sys.argv[1],level,"sudokus is :",np.mean(Iterations))
     RunTimes.extend(RunTime)
     CostValuess.extend(CostValues)
     Restartss.extend(Restarts)
     Iterationss.extend(Iterations)
  file_handle.close()


fig, axs = plt.subplots(5, 1, figsize=(8, 12))

axs[0].plot(RunTimes, color='blue', label='RunTime')
axs[1].plot(CostValuess, color='red', label='CostValuesOFSolution')
axs[2].plot(Restartss, color='green', label='restarts')
axs[3].hist(ReasonForRestart, color='orange', label='reasonforrestarting')
axs[4].plot(Iterationss, color='purple', label='Iterations')

for ax in axs:
    ax.legend()

plt.tight_layout()

plt.show()
