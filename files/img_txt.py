# Disable tensorflow warning messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import cv2
import imutils
import copy
import sys
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border

# Extracts sudoku puzzle from an image.
def detect_puzzle(image):
    # Convert image to grayscale and blur it to filter out noise
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 3)

    # Apply adaptive thresholding
    thresholded_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours in the thresholded image and sort them by size in descending order
    contours = cv2.findContours(thresholded_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Initialize a contour that corresponds to the puzzle outline
    puzzle_contour = None
    
    # Loop over the contours
    for contour in contours:
        # Approximate the contour
        perimeter = cv2.arcLength(contour, True)
        approximated_contour = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # If the approximated contour has 4 points, then it is assumed that 
        # this contour is the puzzle outline
        if len(approximated_contour) == 4:
            puzzle_contour = approximated_contour
            break
        
    puzzle_outline = image.copy()
    cv2.drawContours(puzzle_outline, [puzzle_contour], -1, (0, 255, 0), 3)

    # Apply a four point perspective transform to both the original image and 
    # grayscale image to obtain a top-down bird's eye view of the puzzle
    colored_puzzle = four_point_transform(image, puzzle_contour.reshape(4, 2)) 
    grayscale_puzzle = four_point_transform(gray_image, puzzle_contour.reshape(4, 2))  
    return colored_puzzle, grayscale_puzzle



# Generate (x, y) coordinates for each cell of the sudoku board.
def compute_cell_positions(step_x, step_y):
    # Initialize a list to store (x,y) coordinates of each cell location
    cell_positions = []

    # Loop over the grid locations
    for y in range(9):
        # Initialize the current list of cell locations
        row = []

        for x in range(9):
            # Compute the starting and ending (x,y) coordinates of the current cell
            start_x = x * step_x
            start_y = y * step_y
            end_x = (x + 1) * step_x
            end_y = (y + 1) * step_y

            # Add the (x,y) coordinates to the cell locations list
            row.append((start_x, start_y, end_x, end_y))

            # Crop the cell from the grayscale_puzzle transformed image and then extract
            # the digit from the cell
            cell = grayscale_puzzle[start_y:end_y, start_x:end_x]
            digit = identify_digit(cell)

            # Confirm that the digit is not empty
            if digit is not None:
                recognize_digit(digit, x, y)

        # Add the row to the cell positions
        cell_positions.append(row)

    return cell_positions



# Extracts and returns the digit of the passed in cell. If no digit was identified, the function returns None.
def identify_digit(cell):
    # Apply automatic thresholding to the cell and then clear any connected borders 
    # that touch the border of the cell
    thresholded_cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresholded_cell = clear_border(thresholded_cell)
    
    # Find contours in the thresholded cell
    contours = cv2.findContours(thresholded_cell.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    
    # If no contours were found then this is an empty cell
    if len(contours) == 0:
        return None
    
    # Otherwise find the largest contour in the cell and create a mask for the contour
    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresholded_cell.shape, dtype="uint8")
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    
    # Compute the percentage of masked pixels relative to the total area of the image
    (height, width) = thresholded_cell.shape
    percent_filled = cv2.countNonZero(mask) / float(width * height)
    
    # If less than 3% of the mask is filled then we are looking at noise and can safely ignore the contour
    if percent_filled < 0.03:
        return None
    
    # Apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresholded_cell, thresholded_cell, mask=mask)

    return digit



# Classifies the passed in digit using the OCR model then assigns the digit to  corresponding cell of the unsolved sudoku board.
def recognize_digit(digit, x, y):
    global unsolved_board
    
    # Resize the cell to 28x28 pixels and prepare it classification 
    # 28x28 is the size of images in the MNIST dataset
    resized_roi = cv2.resize(digit, (28, 28))
    resized_roi = resized_roi.astype("float") / 255.0
    resized_roi = img_to_array(resized_roi)
    resized_roi = np.expand_dims(resized_roi, axis=0)

    # Classify the digit and update the sudoku board with the prediction
    prediction = model.predict(resized_roi, verbose=0).argmax(axis=1)[0]
    unsolved_board[y, x] = prediction


# This function displays the solutions of the sudoku puzzle on the color puzzle image.
def display_solutions_on_image(cell_positions, color_puzzle, solved_board):
    # Loop over the cell positions and boards
    for (cell_row, unsolved_board_row, solved_board_row) in zip(cell_positions, unsolved_board, solved_board):
        # Loop over individual cells in the row
        for (box, unsolved_digit, solved_digit) in zip(cell_row, unsolved_board_row, solved_board_row):
            if unsolved_digit == 0:
                # Unpack the cell coordinates
                start_x, start_y, end_x, end_y = box

                # Compute the coordinates of where the digit will be drawn on the output puzzle image
                text_x = int((end_x - start_x) * 0.33) + start_x
                text_y = int((end_y - start_y) * -0.2) + end_y

                # Draw the digit on the sudoku puzzle image
                cv2.putText(
                    color_puzzle, str(solved_digit), (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Convert BGR image to RGB for displaying
    color_puzzle_rgb = cv2.cvtColor(color_puzzle, cv2.COLOR_BGR2RGB)
    
    # Write the solved puzzle image to a file
    cv2.imwrite('solved_sudoku.jpg', color_puzzle_rgb)

if len(sys.argv) < 3:
    print("Usage: python img_txt.py <input_image> <model>")
    sys.exit(1)
input_image_path = sys.argv[1]
model_path = sys.argv[2]

# Load model to detect digits
model = load_model(model_path)

# Load the input image
input_image = cv2.imread(input_image_path)
input_image = imutils.resize(input_image, width=600)

# Detect puzzle from the input image
color_puzzle, grayscale_puzzle = detect_puzzle(input_image)

# Initialize sudoku board
unsolved_board = np.zeros((9, 9), dtype='int')

# Sudoku is a 9x9 grid (81 individual cells), location of each cell can be inferred by
# dividing the grayscale_puzzle image into a 9x9 grid
step_x = grayscale_puzzle.shape[1] // 9
step_y = grayscale_puzzle.shape[0] // 9

# Generate cell positions
cell_positions = compute_cell_positions(step_x, step_y)

# Convert board to a list before solving the puzzle
unsolved_board = unsolved_board.tolist()

# Create a deep copy of the unsolved_board to be passed into the solve_puzzle() function
unsolved_board_copy = copy.deepcopy(unsolved_board)

with open('inp.txt', 'w') as file:
    for row in unsolved_board:
        file.write(' '.join(map(str, row)) + '\n')

with open('cell_positions.txt', 'w') as file:
    for row in cell_positions:
        file.write(' '.join(map(str, row)) + '\n')

with open('color_puzzle.txt', 'w') as file:
    for row in color_puzzle:
        file.write(' '.join(map(str, row)) + '\n')