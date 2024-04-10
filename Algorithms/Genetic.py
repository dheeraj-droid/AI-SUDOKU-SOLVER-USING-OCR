import time
import random
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

print()
print()
print()


RunTimes = []
FitnessValuess = []
Restartss = []

population_size = 10000
mutation_rate = 0.15
generations = 100

print("The Population-Size of Genetic Algorithm is ",population_size)
print("The Mutation_rate is",mutation_rate)
print("The Generations is",generations)
print("The selection rate is",0.5-mutation_rate)


def make_gene(initial=None):
    if initial is None:
        initial = [0] * 9
    gene = list(range(1, 10))
    random.shuffle(gene)
    mapp = {gene[i]: i for i in range(9)}
    for i in range(9):
        if initial[i] != 0 and gene[i] != initial[i]:
            idx = mapp[initial[i]]
            gene[i], gene[idx] = gene[idx], gene[i]
            mapp[gene[i]], mapp[gene[idx]] = i, idx
    return gene

def make_chromosome(initial=None):
    if initial is None:
        initial = [[0] * 9 for _ in range(9)]
    return [make_gene(initial[i]) for i in range(9)]

def make_population(count, initial=None):
    return [make_chromosome(initial) for _ in range(count)]

def get_fitness(chromosome):
    fitness = 0
    seen = [set() for _ in range(27)]  # Rows, columns, and squares
    for i in range(9):  # Rows and columns check
        for j in range(9):
            num_row, num_col, num_square = i, j + 9, (i // 3) * 3 + (j // 3) + 18
            num = chromosome[i][j]
            if num in seen[num_row] or num in seen[num_col] or num in seen[num_square]:
                fitness -= 1
            seen[num_row].add(num)
            seen[num_col].add(num)
            seen[num_square].add(num)
    return fitness

def crossover(ch1, ch2):
    point = random.randint(1, 8)
    new_ch1 = ch1[:point] + ch2[point:]
    new_ch2 = ch2[:point] + ch1[point:]
    return new_ch1, new_ch2

def mutation(chromosome, pm):
    for i in range(9):
        if random.random() < pm:
            idx1, idx2 = random.sample(range(9), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

def genetic_algorithm(initial, population_size, mutation_rate, generations):
    population = make_population(population_size, initial)
    restarts = 0
    for generation in range(generations):
        restarts += 1
        population = sorted(population, key=get_fitness, reverse=True)
        for i in range(population_size):
            if get_fitness(population[0]) == 0:
               return population[0]
        next_generation = population[:2]
        while len(next_generation) < population_size:
            parents = random.sample(population[:3500], 4)  
            child1, child2 = crossover(parents[0], parents[1])
            child3, child4 = crossover(parents[2], parents[3])
            child1 = mutation(child1, mutation_rate)
            child2 = mutation(child2, mutation_rate)
            child3 = mutation(child3, mutation_rate)
            child4 = mutation(child4, mutation_rate)
            next_generation += [child1, child2, child3, child4]
        population = next_generation
    global Restarts
    Restarts.append(restarts)
    return population[0]  # Return the best solution found

if __name__ == "__main__":
   file_name = "MyFile.txt"
   file_handle = open(file_name,"r")

   for level in ["easy","medium","hard","extreme"]:
       RunTime = []
       FitnessValues = []
       Restarts = []
       for i in range(int(sys.argv[1])):
           board=[]
           for i in range(9):
               list1 = []
               line1 = file_handle.readline()
               for i in range(0,len(line1),2):
                   list1.append(int(line1[i]))
                   board.append(list1)
           start  = time.time()
           solution = genetic_algorithm(board,population_size,mutation_rate,generations)
           end = time.time()

           RunTime.append(end-start)
           FitnessValues.append(get_fitness(solution))
    
       print("The mean run time of ",sys.argv[1],level,"sudokus is :",np.mean(RunTime))
       print("The mean Fitness Value of",sys.argv[1],level,"sudokus is :",np.mean(FitnessValues))
       ZeroCostCount = 0
       for i in range(len(FitnessValues)):
           if FitnessValues[i] == 0:
              ZeroCostCount += 1
       print("The Accuracy of ",sys.argv[1],level,"sudokus is :",(100*ZeroCostCount)/len(FitnessValues))
       print("The Average Restarts of",sys.argv[1],level,"sudokus is :",np.mean(Restarts))
       RunTimes.extend(RunTime)
       FitnessValuess.extend(FitnessValues)
       Restartss.extend(Restarts)
   file_handle.close()

   fig,axs = plt.subplots(3,1,figsize=(8,12))

   axs[0].plot(RunTimes,color='blue',label='RunTime')
   axs[1].plot(FitnessValuess,color='purple',label='FitnessValues')
   axs[2].plot(Restartss,color='green',label='Restarts')

   for ax in axs:
       ax.legend()

   plt.tight_layout()

   plt.show()
    

