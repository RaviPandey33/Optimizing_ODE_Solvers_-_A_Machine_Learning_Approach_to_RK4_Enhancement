Setting Details :
_______________________________________________________________
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
1. Batch Learning (Y/N) : Yes (Using Complete sequence as a batch, i.e. just 1 batch)
        Batch Size : 100
        Halton Sequence : 1
2. Initial weights (A1/2, B1/2) : Lobatto series
3. Perturbed : No, Yes, Running both parallely
_______________________________________________________________
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
4. Gradient (Jacfwd/Numeric) : Numeric
5. Loss Function (using abs) : Squared Loss
6. Energy Error Added (Y/N) : No
_______________________________________________________________
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
7. Learning rate : 0.0001
8. RK4 method Step sizes :
        step = 10
        intermediate step (istep) = 10
_______________________________________________________________
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
9. Optimizer : sgd
        Epochs : 1000000
_______________________________________________________________
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
10. Does it converge for :
        10 steps : Yes
        100 steps : Yes
        1000 steps : 
        100000 steps : To check

_______________________________________________________________
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
11. prk file name : Test_prk_for_optimization
_______________________________________________________________
–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
12. Running : Testing_Batch_Learning.py for 50k times
              Perturbed File : 300k times