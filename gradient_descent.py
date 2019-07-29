# Code modified for our purposes from https://towardsdatascience.com/implement-gradient-descent-in-python-9b93ed7108d1
# Projected Gradient Descent for minimizing x transpose * A * x
import numpy as np
def minimize(A):
    n = A.shape[0]
    init_x = np.ones(n) # The algorithm starts with all values equal to one
    rate = 0.01 # Learning rate
    precision = 0.000001 #This tells us when to stop the algorithm
    previous_stepsize = 1 
    max_iters = 100000 # maximum number of iterations
    iters = 0 #iteration counter
    A_transpose = np.transpose(A)
    A_added = np.add(A_transpose, A)
    df = lambda x: np.matmul(np.transpose(x), A_added)  #Gradient of our function
    print("Initial Result",np.matmul(np.transpose(init_x),np.matmul(A,init_x)))
    while previous_stepsize > precision and iters < max_iters:
        prev_x = init_x #Store current x value in prev_x
        init_x = init_x - rate * df(prev_x) #Grad descent
        init_x = projection(init_x)
        #print(init_x)
        previous_stepsize = max(np.absolute(init_x - prev_x)) #Change in x
        iters = iters+1 #iteration count
    
    print("Iteration",iters,"X value is",init_x, "Gradient step size is", previous_stepsize) #Print iterations
    print("Minimized Result",np.matmul(np.transpose(init_x),np.matmul(A,init_x)))
    
def projection(vector):
    #to project the input vector onto a unit-length norm ball
    if np.linalg.norm(vector,2) > 1:
        p = vector/np.linalg.norm(vector,2) #check if the number is 2 for sure
        return p

    elif np.linalg.norm(vector,2) <= 1:
        return vector
    
        
minimize(np.array([[1000,10,10],[10,20,12],[10,12,20]]))
minimize(np.array([[1, 2],[2, 3000]]))
minimize(np.array([[1, 2, 3, 4],[2, 3, 3, 2],[1, 5, 2, 6],[10, 2, 3, 5]]))
minimize(np.array([[1,9], [10000, 200]]))

#how to check this?
# If I adjust the learning rate according to something, would it make it a better algorithm? According to what?
