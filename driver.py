# import MFCC
from librosa.core import audio
import numpy as np
from AudioFeatureExtractor import AudioFeatureExtractor
from classifiers import ml_pipeline
import matplotlib.pyplot as plt
import seaborn as sns


# Feature extraction.  
#Number of segments to split signal
nsegments=1
#number of cepstral coefficients
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

best_params, X_train, X_test, Y_train, Y_test = ml_pipeline(2, audioFeatures.iloc[:, 1:], audioFeatures['Target'])
print(best_params.best_score_)