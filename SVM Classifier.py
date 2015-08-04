"""
    use an SVM to classify content by feature / tag
    
    feature present - 1
    feature not present - 0
"""

from Process import process

# data goes in in format "url | content string"
# data gets split into 2 sets, training and testing sets
# url_train, url_test will hold the urls
# txt_features_train, txt_features_test are the content strings, the features
# features_train, features_test hold the actual features (in bag of words format)
# labels_train_all, labels_test_all hold the labels for each tag
# dict is teh dictionary of number to word (in bag of words), so dict[N] is the actual Nth word in our bag of words
list_of_data = ['Data/mariott-data.txt', 'Data/IHG-data1.txt', 'Data/IHG-data2.txt', 'Data/IHG-data3.txt', 'Data/hyatt-data.txt']
url_train, url_test, txt_features_train, txt_features_test, features_train, features_test, labels_train_all, labels_test_all, dict = process(list_of_data)

# list of tags to run the model for
tags_str = ["is_food_beverage", "is_price_discount", "is_loyalty_points", "is_brand_awareness", "is_family", "is_romance", "is_resort", "is_loyalty_point_redemption", "is_spa", "is_local_culture"]

# run model for every feature/tag
print "number of training data points: ", len(labels_train_all)
print "number of testing data points: ", len(labels_test_all)

import numpy as np
for i in range(len(labels_train_all[0])):
    # pick out the correct labels from the labels_all matrix
    labels_train = np.array(labels_train_all)[:,i]
    labels_test = np.array(labels_test_all)[:,i]

    # SVM model
    from sklearn.svm import SVC
    print tags_str[i], " number of training labels where tag = 1: ",  sum(labels_train)
    print tags_str[i], " number of testing labels where tag = 1: ",  sum(labels_test)
    clf = SVC(kernel="rbf", C=10000.0)
    clf.fit(features_train, labels_train)

    # get the test predictions and calculate accuracy
    pred_test = clf.predict(features_test)
    from sklearn.metrics import accuracy_score
    acc_test = accuracy_score(pred_test, labels_test)
    print tags_str[i], " testing accuracy: ", acc_test

    # get redictions on the training data set and get training accuracy (should be very high)
    pred_train = clf.predict(features_train)
    acc_train = accuracy_score(pred_train, labels_train)
    print tags_str[i], " training accuracy: ", acc_train

    # put data for output in a data frame and write out to excel
    from pandas import DataFrame
    df_train = DataFrame({'URL': url_train, 'Content': txt_features_train, 'Actual Tag': labels_train, 'Predicted Tag': pred_train})
    df_test = DataFrame({'URL': url_test, 'Content': txt_features_test, 'Actual Tag': labels_test, 'Predicted Tag': pred_test})

    from pandas import ExcelWriter
    writer = ExcelWriter('Output/SVM Results-' + tags_str[i] + '.xlsx')
    df_train.to_excel(writer,'train', columns=['URL', 'Content', 'Actual Tag', 'Predicted Tag'])
    df_test.to_excel(writer,'test', columns=['URL', 'Content', 'Actual Tag', 'Predicted Tag'])
    writer.save()


#########################################################


