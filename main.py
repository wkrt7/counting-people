from texture_features import texture_features_calculation
from segmentacja import segmentacja
import os
import cv2
import time
from SVR_own_RBF import SVR_RBF
from get_terget import get_target
import scipy.io
from decision_function import decision_function

# neded paths
mat_path = "vidf1_33_000_count_roi_mainwalkway.mat" # for _000_
directory_train = "C:/Users/Praca/Desktop/Praca_inzynierska/Model programowy/Gotowe/ver1/data_train"
directory_test = "C:/Users/Praca/Desktop/Praca_inzynierska/Model programowy/Gotowe/ver1/data_test"


def main(train_path=directory_train, test_path=directory_test,
         mat_path=mat_path, kernel="rbf",gamma=0.1,C=1e6):
    print("Kernel type: "+ str(kernel))
    # files opening:
    f = open("wyniki.txt","w")

    all_time_start = time.time()
    # load ground truth from file
    mat = scipy.io.loadmat(mat_path)

    i = 0
    train_data = []
    test_data = []
    # set targets
    train_target = [12,13,16,15,12,13,14,18,21,22,20,20]
    temp = get_target(mat)
    test_target = temp[:37]

    # prepare tet_data
    for filename in os.listdir(train_path):
        if filename.endswith(".png"):
            start = time.time()
            print(filename)
            path = (os.path.join(train_path, filename))
            img_2_feat = segmentacja(path)
            temp = texture_features_calculation(img_2_feat)
            train_data.append(temp)
            print("time: " + str(time.time() - start))
        else:
            continue
    # train SVM
    dual_coef_, intercept_, support_vectors_,gamma =SVR_RBF(train_data, train_target,
                                                            kernel=kernel,test=False,
                                                            gamma=gamma,C=C)

    #prepare train_data and testing
    for filename in os.listdir(test_path):
        if filename.endswith(".png"):
            start = time.time()
            print(filename)
            path = (os.path.join(test_path, filename))
            img_2_feat = segmentacja(path)
            temp = texture_features_calculation(img_2_feat)
            # predict value:
            resultat = decision_function(temp,dual_coef_,intercept_,support_vectors_,gamma=gamma,kernel=kernel)

            print("Real value = " + str(test_target[i]) + " calculated value = " + str(resultat))
            f.write("Real value = " + str(test_target[i]) + " calculated value = " + str(resultat)+"\n")
            test_data.append(temp)
            i += 1
            print("time: " + str(time.time() - start))
        else:
            continue

    f.close()
    print("time: " + str(time.time() - all_time_start))

if __name__== "__main__":
    main(kernel= "linear")