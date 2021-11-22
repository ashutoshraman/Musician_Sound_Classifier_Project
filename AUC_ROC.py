#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV


class AUC_ROC():
    
    def AUC_ROC(y_test, y_score):
        def plot_roc_curve(fpr, tpr, label=None): 
            plt.plot(fpr, tpr, c='green', linewidth=4, label=label) 
            plt.plot([0, 1], [0, 1], 'k--') 
            plt.axis([0, 1, 0, 1])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
        
        fpr, tpr, thresholds = roc_curve(y_test, y_score[:,1], pos_label=1)
        auc = roc_auc_score(y_test, y_score[:,1])
        
        plot_roc_curve(fpr, tpr)
        plt.show()
        return(auc)
