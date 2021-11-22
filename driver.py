# import MFCC
from librosa.core import audio
import numpy as np
from AudioFeatureExtractor import AudioFeatureExtractor
from classifiers import ml_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix
import seaborn as sns


# Feature extraction.  
#Number of segments to split signal
nsegments=1
#number of cepstral coefficients, shall we ask the user to input?
num_mfcc=2
# Number of samples to use for testing
t_size = 0.5

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