# %%
import joblib
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import plot_confusion_matrix, roc_curve, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

class deploy():
    
    def deploy(model, X_test_real, Y_test_real):
        
        # save the model to disk
        filename = 'finalized_model.sav'
        joblib.dump(model, filename)
        
        # some time later...
        
        # load the model from disk
        loaded_model = joblib.load(filename)
        accuracy = loaded_model.score(X_test_real, Y_test_real)
        print(f'accuracy of this model is {accuracy}')
        
        # not sure if we need the following? - use knn/else to find auc
        knn = KNeighborsClassifier(n_neighbors=5)
        y_score = cross_val_predict(knn, X_test_real,Y_test_real,cv=10, method='predict_proba')
        auc = roc_auc_score(y_test, y_score[:,1])
        print(f'AUC of this model is {auc}')

        return(accuracy, auc)