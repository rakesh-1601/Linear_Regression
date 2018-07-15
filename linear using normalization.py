#Hello there... myself Rakesh pandey
#Student at NIT jalndhar(First year -CSE branch)
#And this is minor project
#To implement linear regression without using Sklearn
#I have used analytical or normal method(Normalization method) to make my own LR program....
#below is the code for Linear regression....


#Importing all the essential modules !!!
import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

#extracting data from dataset::

diabetes=datasets.load_diabetes()    #loading diabetes dataset...although any regression dataset can be choosed
#x_train=diabetes.data[:-20,1:4]     #selecting training set and number of features (multivariate)
                                     # here 3 features have been choosed and all data except last 20 from each feature

x_train=diabetes.data[:-20,np.newaxis,2]    #selecting training set and number of features (univariate)
y_train=diabetes.target[:-20]   # selecting label or answer for training set...

#x_test=diabetes.data[-20:,1:4]     #selecting predicting data for model(multivariate)
x_test=diabetes.data[-20:,np.newaxis,2]   #selecting predicting data for model(univariate)
y_test=diabetes.target[-20:]

# function to calculate mean square error::
def my_own_error_obvio_squared_mean_error(test_value,predicted_value):
    totalError = 0
    for i in range(0, len(test_value)):
        x = test_value[i]
        y = predicted_value[i]
        totalError += (y - x) ** 2
    print("The error is ",totalError / float(len(test_value)))


#function to fit the model with training dataset:::
def fitmodel(x_train,y_train):
    global coef
    global base
    #Using the normaliztion formulae for predicting parameters
    # Formulae is  (x(transpose)*x)-1*x(tranpose)*y
    # where is x is same as provided by user ....but first column is being filled with one...using below code
    x_train = np.append([[1 for _ in range(0,len(x_train))]], x_train.T,0).T #to make all values of first column of x as 1
    #implementing formulae
    z=np.dot(x_train.T,x_train)
    a=np.linalg.inv(z)
    b=np.dot(a,x_train.T)
    coe=np.dot(b,y_train)# a matrix with 4 parameters out of which 1st is intercept and rest is coeff....
    coef=coe[1:]
    base=coe[0]


#function to predict value
#value is predicted simply by multiplying features with coef(transpose)...
def predict(x_test):
    predictedvalue=[]
    for i in range(len(x_test)):
        predictedvalue.append((np.dot(coef,x_test[i]))+base)
    ans=np.array(predictedvalue)
    return(ans)

#function to print coefficient and intercept
def coeff_and_intercept():
    print("The coefficients are",coef)
    print("The intercept is ",base)


#running the model
fitmodel(x_train,y_train)
ans=predict(x_test)
print("The predicted values are",ans[:,np.newaxis])
coeff_and_intercept()
my_own_error_obvio_squared_mean_error(y_test,ans)


#printing the graph...only for univariate ....and not for multivariate!!!!
plt.scatter(x_test,y_test,color='black')
plt.plot(x_test,ans,color='blue',linewidth=3)

plt.xticks(())
plt.yticks(())
plt.show()