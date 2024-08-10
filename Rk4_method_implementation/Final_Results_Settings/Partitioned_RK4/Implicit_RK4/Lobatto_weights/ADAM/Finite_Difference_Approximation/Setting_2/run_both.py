import subprocess as s
import sys

## Epochs for all the files
# epochs = 10

# With Energy Error
process1 = s.Popen([sys.executable, 'With_Energy/Testing_Batch_Learning.py'])
process2 = s.Popen([sys.executable, 'With_Energy/Testing_Batch_Learning(Perturbed).py'])
# Without Energy Error
process3 = s.Popen([sys.executable, 'Without_Energy/Testing_Batch_Learning.py'])
process4 = s.Popen([sys.executable, 'Without_Energy/Testing_Batch_Learning(Perturbed).py'])

# Wait for both processes to complete
process1.wait()
process2.wait()
process3.wait()
process4.wait()

