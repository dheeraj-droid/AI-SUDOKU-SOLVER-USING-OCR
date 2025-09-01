import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import subprocess
import numpy as np
acc= 0 


for i in range(1, 161): 

    count = 0
    image_path = f'./images/sudoku{i}.jpg'
    model_path = './model/model-1.h5'  

    print(f"Processing image: {image_path}, model: {model_path}")

    # Run img_txt.py script for current image
    subprocess.run(['python', 'img_txt.py', image_path, model_path])

    print("Script executed successfully")

    # Read input Sudoku grid from generated text file
    sudoku_from_model = []
    with open("inp.txt", 'r') as file:
        for line in file:
            row = [int(num) for num in line.strip().split()]
            sudoku_from_model.append(row)

    # Read expected Sudoku solution from text file
    sudoku_test = []
    with open(f"./data/sudoku{i}.txt", 'r') as file:
        for line in file:
            row = [int(num) for num in line.strip().split()]
            sudoku_test.append(row)

    for j in range(9):
        for k in range(9):
            if(sudoku_from_model[j][k]==sudoku_test[j][k]):
                count +=1
    acc += count/81

    print("Processing completed for Sudoku", i)

accuracy = acc/160
print(f"Accuracy of model = {np.round(accuracy*100,5)}% ")
os.remove("cell_positions.txt")
os.remove("color_puzzle.txt")
os.remove("inp.txt")