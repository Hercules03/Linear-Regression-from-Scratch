from numpy import *

def error(b,m,points):
    # initialize it at 0
    totalError = 0
    # for every point
    for i in range(len(points)):
        #get x value
        x = points[i,0]
        #get y value
        y = points[i,1]
        #calculate the specific error and add it to the total error
        totalError += (y - (m*x+b))**2
    #get the average
    return totalError / float(len(points))

def GD(points, starting_b, starting_m, learning_rate, num_interations):
    #starting b and m
    b = starting_b
    m = starting_m
    
    #gradient descent
    for i in range(num_interations):
        #update b and m with the new more accurate b and m by performing
        #this gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def step_gradient(b_cuurent, m_current, points, learning_rate):
    #Initializing the starting points of our gradient
    b_gradient = 0
    m_gradient = 0
    
    N = float(len(points))
    for i in range(len(points)):
        x = points[i,0]
        y = points[i,1]
        #direction with respect to b and m
        #computing partial derivatices of our error function
        m_gradient += -(2/N)*x * (y-(m_current*x+b_cuurent))
        b_gradient += -(2/N)*(y-(m_current*x + b_cuurent))
    
    #update our b and m value using this partial derivatives
    new_b = b_cuurent - learning_rate*b_gradient
    new_m = m_current - learning_rate*m_gradient
    
    return [new_b, new_m]
        
def run():
    #Step 1 - Collect our data
    points = genfromtxt("data.csv", delimiter=",")
    
    #Step 2 - define our hyperparameters
    learning_rate = 0.0001 # how fast the data will converge
    #y = mx + b (slope formula)
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    
    #Step 3 - train our model
    print(f"starting gradient descent at b = {initial_b}, m = {initial_m}, error = {error(initial_b, initial_m, points)}")
    [b, m] = GD(points, initial_b, initial_m, learning_rate, num_iterations)
    
    print(f"ending point at b = {b}, m = {m}, error = {error(b, m, points)}, iteration = {num_iterations}")
    

if __name__ == "__main__":  # what we will do in a main function
    run()