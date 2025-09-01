# app.py

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import imutils
from sudoku_solver import solve_sudoku, print_board # Import your solver logic

# --- Load Your Pre-trained OCR Model ---
# This path is relative to the root of your repository
MODEL_PATH = 'model/model.h5'
model = load_model(MODEL_PATH)


# --- Helper Functions (Copied and adapted from your main.py) ---

def pre_process_image(img):
    """Pre-processes the image for digit recognition."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return thresh

def find_corners_of_largest_polygon(img):
    """Finds the corners of the largest polygon (the Sudoku grid)."""
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]
    
    sums = []
    diffs = []
    for p in polygon:
        sums.append(p[0][0] + p[0][1])
        diffs.append(p[0][0] - p[0][1])
    
    top_left = polygon[np.argmin(sums)].squeeze()
    bottom_right = polygon[np.argmax(sums)].squeeze()
    top_right = polygon[np.argmax(diffs)].squeeze()
    bottom_left = polygon[np.argmin(diffs)].squeeze()
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)

def warp_perspective(img, corners):
    """Warps the perspective of the image to get a top-down view of the grid."""
    side_len = 450
    new_corners = np.array([[0, 0], [side_len - 1, 0], [side_len - 1, side_len - 1], [0, side_len - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(corners, new_corners)
    return cv2.warpPerspective(img, matrix, (side_len, side_len))

def extract_digits_and_solve(warped_img, original_img, corners):
    """Extracts each digit, predicts it, solves the puzzle, and draws the solution."""
    grid = np.zeros((9, 9), dtype=int)
    side = warped_img.shape[0] // 9

    for i in range(9):
        for j in range(9):
            cell = warped_img[i*side:(i+1)*side, j*side:(j+1)*side]
            cell = cell[4:-4, 4:-4] # Shave off a few pixels
            
            if cell.sum() > 3000: # If the cell is not empty
                cell = cv2.resize(cell, (28, 28))
                cell = cell / 255.0
                cell = cell.reshape(1, 28, 28, 1)
                
                prediction = model.predict(cell)
                digit = np.argmax(prediction)
                grid[i, j] = digit

    solved_grid = [row[:] for row in grid]
    if solve_sudoku(solved_grid):
        # Draw the solved numbers back onto the original image
        side_len = 450
        new_corners = np.array([[0, 0], [side_len - 1, 0], [side_len - 1, side_len - 1], [0, side_len - 1]], dtype=np.float32)
        matrix_inv = cv2.getPerspectiveTransform(new_corners, corners)

        for i in range(9):
            for j in range(9):
                if grid[i, j] == 0: # If the cell was originally empty
                    text = str(solved_grid[i][j])
                    # Position the text in the center of the cell
                    text_x = j * side + side // 2
                    text_y = i * side + side // 2
                    pos = np.array([[[text_x, text_y]]], dtype=np.float32)
                    
                    # Transform position back to original image perspective
                    original_pos = cv2.perspectiveTransform(pos, matrix_inv)[0][0]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(original_img, text, (int(original_pos[0]-10), int(original_pos[1]+10)), font, 2, (0, 255, 0), 3)
        return original_img
    else:
        return None


# --- Streamlit App UI ---

st.title("AI Sudoku Solver using OCR 📸")
st.write("Upload an image of a Sudoku puzzle, and the AI will solve it for you.")

uploaded_file = st.file_uploader("Choose a Sudoku image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)

    st.image(original_image, channels="BGR", caption="Uploaded Puzzle")
    
    if st.button("Solve Puzzle"):
        with st.spinner("Processing image and solving..."):
            try:
                # Main processing pipeline
                processed_image = pre_process_image(original_image)
                corners = find_corners_of_largest_polygon(processed_image)
                
                # Check if a grid was found
                if corners is None or len(corners) != 4:
                     st.error("Could not find a Sudoku grid in the image. Please try another one.")
                else:
                    warped = warp_perspective(original_image.copy(), corners)
                    solved_image = extract_digits_and_solve(warped, original_image.copy(), corners)

                    if solved_image is not None:
                        st.success("Puzzle Solved!")
                        st.image(solved_image, channels="BGR", caption="Solved Puzzle")
                    else:
                        st.error("Could not solve the Sudoku puzzle. The OCR might have misread a digit.")
            except Exception as e:
                st.error(f"An error occurred: {e}. The image might not be a valid Sudoku puzzle.")
