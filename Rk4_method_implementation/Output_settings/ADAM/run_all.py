import subprocess as s
import sys

## Epochs for all the files
# epochs = 10

process1 = s.Popen([sys.executable, 'Testing_Batch_Learning.py'])
process2 = s.Popen([sys.executable, 'Testing_Batch_Learning(Perturbed).py'])

# Wait for both processes to complete
process1.wait()
process2.wait()

