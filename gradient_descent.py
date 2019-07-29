# Code modified for our purposes from https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1
# Gradient Descent for minimizing x transpose * A * x
import numpy as np
def minimize(A):
    n = A.shape[0]
    init_x = np.ones(n) # The algorithm starts with all values equal to one
    rate = 0.01 # Learning rate
    precision = 0.000001 #This tells us when to stop the algorithm
    previous_stepsize = 1 
    max_iters = 10000 # maximum number of iterations
    iters = 0 #iteration counter
    A_transpose = np.transpose(A)
    A_added = np.add(A_transpose, A)
    df = lambda x: np.matmul(np.transpose(x), A_added)  #Gradient of our function
    
    while previous_stepsize > precision and iters < max_iters:
        prev_x = init_x #Store current x value in prev_x
        init_x = init_x - rate * df(prev_x) #Grad descent
        #print(init_x - prev_x)
        previous_stepsize = max(np.absolute(init_x - prev_x)) #Change in x
        iters = iters+1 #iteration count
    
    print("Iteration",iters,"X value is",init_x, "step size is", previous_stepsize) #Print iterations
    
#print("The local minimum occurs at", cur_x)
minimize(np.array([[10,1],[1,2]]))

#how to check this?