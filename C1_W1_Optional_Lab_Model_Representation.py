# x = x_train
# y = y_train
# x_i, y_i= training example of ith datasets
# m = no. of training examples
# w = parameter or weight
#b = parameter bias
# f =function which equals to f_wb = wx+b

# we will use the numpy and matplotlib
#import the function as

import numpy as np
import matplotlib.pyplot as plt

# define the values for traing examples
# np.array used as numpy function to input values within ([])
# x_train and y_train are variables

x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_train = np.array([100, 200, 300, 400, 500])

#print the values using f string as f'
print(f'x_train = {x_train}')
print(f'y_train = {y_train}')

#m is the no. of training example
# shape is used to detrmine the no. of values

print (f'x_train.shape: {x_train.shape}')

#pass it to a variable m
# [0] no. of rows in an array (2d array)
m = x_train.shape [0] 
n = y_train.shape[0]
print(f'no. of training examples = {m} and {n}')

#also can be used length function len(variable)
m = len(x_train)
n = len(y_train)
print(f'no. of training examples = {m} and {n}')

#accessing training datasets x_i and y_i
i = 0 #pyhton first value index starts with 0
x_i = x_train[i]
y_i = y_train[i]
print(f'x{i}, y{i} = ({x_i}, {y_i})')

#plot the data
#use the function as plt and then plotting type
#c means color shows the points as red c=red and marker as x
plt.scatter(x_train, y_train, marker = 'x', c='r')
plt.title = ("hosuing example")
plt.ylabel=('price in 1000s of dollars')
plt.xlabel =('size in square feet')
#plt.show()

#Now define the model 
#f(w,b) = wx+b
#defining w and b depends on the values that you can choose between the points 
#that you descriobed or represented in the datasets
#meaning if your points have 1-10 the choose 2 or 1 or 3
#if your values are in 100s the choose 100. Actually depends on y variable
#x_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
#y_train = np.array([100, 400, 600, 800, 1000])


w = 70
b = 100

#for  𝑥(0) , f_wb = w * x[0] + b
#for  𝑥(1) , f_wb = w * x[1] + b

#lets define a function
#starts with def with function name and arguments.
#how many arguments you want to pass
#ends with colon
def predict_model(x,w,b):
    m = x.shape[0] #takes the x values as array #initilaize a value
    f_wb = np.zeros(m) #np.zeros(n) return 1D array with n entries
    #loop through the values using for loop
    for i in range(m):#use range to go through the no. of training sets
        f_wb[i] = w * x[i]+b
        #for first value of x
        # 100 + 100 = 200 but our first value is 100
        #so it has to be 50 + 50 =100 but does not work
        #lets try other values
    
    return f_wb

#after defining a function call the function
tmp_f_wb = predict_model(x_train, w, b)
# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')
# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')
plt.show()

#Prediction
#Now that we have a model, we can use it to make our original prediction. Let's predict the price of a house with 1200 sqft. Since the units of  𝑥  are in 1000's of sqft,  𝑥  is 1.2.

w = 70                         
b = 100    
x_i = 3.5
cost_3500sqft = w * x_i + b    
print(f"${cost_3500sqft:.0f} thousand dollars")
