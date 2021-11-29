from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

# from plot_Boundary import plot_decision_boundary

def k_nearest_pipeline(): #potentially functionality to choose what data to use, and thus pipeline to form (mfcc vs dft/pca)
    pipeline_object = make_pipeline(StandardScaler(), KNeighborsClassifier())
    knn_param_grid = [{'kneighborsclassifier__n_neighbors': [5, 10, 15, 20],
                       'kneighborsclassifier__weights': ['distance', 'uniform'],
                       'kneighborsclassifier__p': [1, 2, 3], #1 is manhattan, 2 is euclidian, and arbitrary or 3 is minkowski dist
                       }]
    return pipeline_object, knn_param_grid

def svm_pipeline():
    pipeline_obj = make_pipeline(StandardScaler(), SVC(random_state=10))
    svm_param_grid = [{'svc__C': [.01, .1, 1, 10],
                       'svc__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                       'svc__gamma': ['scale', 'auto', .01, .1, 1, 10],
                       }]
    return pipeline_obj, svm_param_grid

def logistic_pipeline():
    pipeline_obj = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=10))
    lr_param_grid = [{'logisticregression__C': [.001, .01, .1, 1],
                      'logisticregression__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                      'logisticregression__penalty': ['l1', 'l2', 'none']}]
    return pipeline_obj, lr_param_grid

def pca_k_nearest_pipeline(variance):
    pipeline_object = make_pipeline(StandardScaler(), PCA(n_components=variance), KNeighborsClassifier())
    knn_param_grid = [{'kneighborsclassifier__n_neighbors': [5, 10, 15, 20],
                       'kneighborsclassifier__weights': ['distance', 'uniform'],
                       'kneighborsclassifier__p': [1, 2, 3], #1 is manhattan, 2 is euclidian, and arbitrary or 3 is minkowski dist
                       }]
    return pipeline_object, knn_param_grid

def pca_svm_pipeline(variance):
    pipeline_obj = make_pipeline(StandardScaler(), PCA(n_components=variance), SVC(random_state=10))
    svm_param_grid = [{'svc__C': [.01, .1, 1, 10],
                       'svc__kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                       'svc__gamma': ['scale', 'auto', .01, .1, 1, 10],
                       }]
    return pipeline_obj, svm_param_grid

def pca_logistic_pipeline(variance):
    pipeline_obj = make_pipeline(StandardScaler(), PCA(n_components=variance), LogisticRegression(max_iter=1000, random_state=10))
    lr_param_grid = [{'logisticregression__C': [.001, .01, .1, 1],
                      'logisticregression__solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
                      'logisticregression__penalty': ['l1', 'l2', 'none']}]
    return pipeline_obj, lr_param_grid

def ml_pipeline(pipeline_choice, X, Y, variance=.75):
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, shuffle=True, random_state = 10, stratify=Y)

    if pipeline_choice == 0:
        pipeline_obj, param_grid = k_nearest_pipeline()
    elif pipeline_choice == 1:
        pipeline_obj, param_grid = svm_pipeline()
    elif pipeline_choice == 2:
        pipeline_obj, param_grid = logistic_pipeline()
    elif pipeline_choice == 3:
        pipeline_obj, param_grid = pca_k_nearest_pipeline(variance)
    elif pipeline_choice == 4:
        pipeline_obj, param_grid = pca_svm_pipeline(variance)
    elif pipeline_choice == 5:
        pipeline_obj, param_grid = pca_logistic_pipeline(variance)
    else:
        print('no classifier for this entry')

    grid_search_obj = GridSearchCV(estimator=pipeline_obj,
                  param_grid=param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=3, verbose=0)

    scores = cross_val_score(grid_search_obj, X_train, Y_train, scoring='accuracy',cv=5, verbose=0)
    

    best_params_model = grid_search_obj.fit(X_train, Y_train) #best params from model will give hyperparameters
    print("Mean Accuracy for pipeline: {:f}".format(np.mean(scores)))
    print("Stdev of Accuracy for pipeline: {:f}".format(np.std(scores)))
    return best_params_model, X_train, X_test, Y_train, Y_test
