import numpy as np
import time
from sklearn.datasets import make_regression
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import copy
import math
from numpy import linalg as LA
start_time = time.time()

data_temp,target = load_boston(return_X_y=True)


def SVR_RBF(trn_data = data_temp,trn_target = target,tst_data = data_temp,tst_target = target,kernel = "rbf" ,test = True,gamma = 0.1, C = 1e6):

    #prepare all data
    data_temp = np.array(trn_data)
    train_data = np.array(trn_data)
    train_target = np.array(trn_target)



    number_of_samples = len(trn_target)
    number_of_features = len(trn_data[0])

    #shoudl be size =(number_of_sample,number_of_features)
    train_data = train_data.reshape((len(train_target),number_of_features))
    #shoudl be size =(number_of_sample,1)
    train_target = train_target.reshape((len(train_target),1))

    #training SVM


    clf = SVR(kernel=kernel,C=C, gamma=gamma)

    train_target = np.ravel(train_target)
    clf.fit(train_data,train_target)

    #needed to be writen to file!!
    #print(clf.dual_coef_)
    #print(clf.support_vectors_)
    #print(clf.intercept_)
    if(test == True):#if we want to test at the same time test == true

        test_data = np.array(tst_data)
        test_target = np.array(tst_target)
        no_to_test = 0
        data_test = train_data[no_to_test]#data for tesing self implemented dec function
        y1 =0
        kernel_value = []
        #Counting decision function

        for i in range(len(clf.dual_coef_[0])):
            math.exp((1))
            if kernel == "rbf":
                y1 += clf.dual_coef_[0,i]*(math.exp(-gamma*math.pow(float(LA.norm(clf.support_vectors_[i]-data_test)),2)))#!!!!DESICION FUNCTION FOR RBF KERNEL!!!!
            #kernel_value.append(math.exp(-gamma*math.pow(float(LA.norm(clf.support_vectors_[i]-data_test)),2)))
            elif kernel =="linear":
                y1 += clf.dual_coef_[0, i] * sum((clf.support_vectors_[i] * data_test))
            else:
                print "Wrong kernel!"
        y1 += clf.intercept_ # wartosc wyliczona ze wzrou
        y2 = clf.predict(test_data) # wartosc
        y3 = test_target #rzeczywista wartosc
        print('\n y1 ' + str(y1) + '\n y2 ' + str(y2) + '\n y3 ' + str(y3))  # y1,y2 powinny byc takie same

    return clf.dual_coef_, clf.intercept_, clf.support_vectors_, gamma

execute_time = time.time()- start_time
print("Czas: " + str(execute_time)+' sec')
#plt.show()

if __name__ == "__main__":

    SVR_RBF(data_temp,target)
    print('Done')








