# AI SUDOKU SOLVER USING OCR
##FIRST MAKE SURE YOU HAVE ALL THE REQUIRED LIBRARIES 
  1. keras
  2. computer vision
  3. imutils
  4. matplotplotlib
  5. numpy
  6. sklearn

##HOW TO RUN AND CHECK THE OUTPUT OF THE PROJECT
   1. FIRST OPEN THE FOLDER MAIN MODEL AND SUPPORTING FILES IN THE COMMAND LINE.
   2. RUN THE COMMAND python ./model/train_model.py
   3. This command creates a model that can recognise numbers.
   4. Now the model with name model.h5 will be created in model folder and also contains 4 pre-used models.
   5. Now since the model has been created we need to give input and run the model.
   6. In the command line run the command python main.py </path to input-image> </path to the model>.
   7. After the command is run we will get a image named solved_sudoku in which our solved sudoku is present.

##The stats calculation files are our additional files that we used for testing purposes.The process of using those files is mentioned in howtouse.txt file inside the folder.
##The algorithm folder contains the codes of our 3 algorithms.
