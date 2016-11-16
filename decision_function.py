'''

Counting desicion function based on rbf kernel

'''
import  math
from numpy import linalg as LA

def decision_function(input_data,dual_coef_,intercept_,support_vectors_ ,gamma,kernel = "rbf"):
    y1 = 0
    for i in range(len(dual_coef_[0])):
        if kernel == "rbf":
            y1 += dual_coef_[0, i] * (math.exp(-gamma * math.pow(float(LA.norm(support_vectors_[i] - input_data)), 2)))
        elif kernel == "linear":
            y1 += dual_coef_[0, i] * sum((support_vectors_[i] * input_data))
    y1 += intercept_
    return(y1)

if __name__ == "__main__":
    decision_function()