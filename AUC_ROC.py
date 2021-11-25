# in case this breaks later on, this was a cell with '#%%' here 
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



    
def AUC_ROC(y_test, y_score, artist):
    def plot_roc_curve(fpr, tpr, label=artist): 
        plt.plot(fpr, tpr, linewidth=4, label=label) 
        # plt.fill_between(fpr, tpr, 0,  #AUC: area under the curve
        #      facecolor="orange", # The fill color
        #      color='green',       # The outline color
        #      alpha=0.2)          # Transparency of the fill
        
    
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
    auc = roc_auc_score(y_test, y_score)
    
    plot_roc_curve(fpr, tpr)
    return auc
