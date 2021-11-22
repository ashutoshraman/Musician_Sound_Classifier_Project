# import MFCC
from librosa.core import audio
import numpy as np
from AudioFeatureExtractor import AudioFeatureExtractor
from classifiers import ml_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import seaborn as sns
import pickle
from AUC_ROC import AUC_ROC


# Feature extraction.  
#Number of segments to split signal
nsegments=1
#number of cepstral coefficients
num_mfcc=2
# Number of samples to use for testing
t_size = 0.5
classes = ['ed_sheeran', 'drake', 'taylor_swift', 'linkin_park', 'justin_bieber']

if __name__ == '__main__':
    directory = "./project_2_data/*"
    audioProcessor = AudioFeatureExtractor(directory)
    audioFeatures = audioProcessor.constructMFCCFeatures(nsegments, num_mfcc)
    print(audioFeatures)
    print(np.unique(audioFeatures["Target"]))

    def plot_mfcc(dataframe):
        plt.figure()
        sns.scatterplot(x=dataframe['MFCC 0'], y=dataframe['MFCC 1'], hue=dataframe['Target'], style=dataframe['Target'], legend='auto')
        plt.title('First 2 MFCCs for Audio Clips of Artists')
        plt.xlabel('MFCC 1')
        plt.ylabel('MFCC 2')
        plt.show()
    plot_mfcc(audioFeatures)

    best_params, X_train, X_test, Y_train, Y_test = ml_pipeline(0, audioFeatures.iloc[:, 1:], audioFeatures['Target'])
    print(best_params.best_score_)

    # accuracy and evaluation of model
    train_score = best_params.score(X_train, Y_train) #why is this not same as best_score_ above??
    test_score = best_params.score(X_test, Y_test)
    print(train_score, test_score)

    Y_pred = best_params.predict(X_test)
    cm = confusion_matrix(Y_test, Y_pred)
    # print("Confusion Matrix: \n")
    # print(cm)

    titles_options = [("Confusion matrix, without normalization", None),
                    ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(best_params, X_test, Y_test,
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
        disp.ax_.set_title(title)

        # print(title)
        # print(disp.confusion_matrix)

        plt.show()

    print(classification_report(Y_test, Y_pred))

    plt.figure(2)
    for i in classes:
        new_Y_test = (Y_test == i)
        new_Y_pred = (Y_pred == i)
        auc = AUC_ROC(new_Y_test, new_Y_pred, i)
        print('The AUC for {} is {}'.format(i, auc))
        pass
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve with One vs. All for Each Artist')
    plt.legend()
    plt.show()


    # save model for later deployment
    filename = 'finalized_model.sav'
    pickle.dump(best_params, open(filename, 'wb'))
    #joblib.dump(model, filename)