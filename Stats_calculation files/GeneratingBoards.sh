for (( i=0; i<$1; i++ )); do
   python3 sudoku_generator1.py base.txt easy | sed 's/_/0/g' | sed 's/|/,/g' >> MyFile.txt
done
for (( i=0; i<$1; i++ )); do
   python3 sudoku_generator1.py base.txt medium | sed 's/_/0/g' | sed 's/|/,/g' >> MyFile.txt
done
for (( i=0; i<$1; i++ )); do
   python3 sudoku_generator1.py base.txt hard | sed 's/_/0/g' | sed 's/|/,/g' >> MyFile.txt
done
for (( i=0; i<$1; i++ )); do
   python3 sudoku_generator1.py base.txt extreme | sed 's/_/0/g' | sed 's/|/,/g' >> MyFile.txt
done
