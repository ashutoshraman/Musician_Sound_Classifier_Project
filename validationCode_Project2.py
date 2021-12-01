import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from AudioFeatureExtractor import AudioFeatureExtractor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import plot_confusion_matrix, roc_curve, roc_auc_score
import pickle


# -----MODIFY PREPROCESS FUNCTION TO PERFORM NECESSARY FEATURE EXTRACTION------------------------
def preProcess(directory, num_mfcc): # ----MODIFY PREPROCESSING INPUT PARAMETERS AS NEEDED-------
    # Feature extraction step
    # This is my own class. You can have yours here.
    audioProcessor = AudioFeatureExtractor(directory)
    print(audioProcessor.audio_files)
    audioFeatures = audioProcessor.constructMFCCFeatures(1, num_mfcc, validation=True) #ADD VALIDATION = TRUE 
    X = audioFeatures.iloc[:,1:]  
    Y_filename = audioFeatures["Target"]
    print(audioFeatures)

    return X, Y_filename


# -------------------do NOT modify  this function--------------------------------------------------
def validationPredict(estimator, X, Y_filename, Y_validationDict, artists): 

    num_samples = len(Y_filename)
    Y_validation = np.zeros((num_samples,), dtype=object)
    for i in range(num_samples):
        #print(Y_validationDict[Y_filename[i]])
        Y_validation[i] = Y_validationDict[Y_filename[i]]

    # Predict labels
    #Y_pred = estimator.predict(X)
    score = estimator.score(X, Y_validation)

    # Validation performance
    print("Model Accuracy: {:f}".format(score))

    # Plot confusion matrix
    titles_options = [("Confusion matrix, without normalization", None), ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(estimator, X, Y_validation,
                                 display_labels=artists,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
        disp.ax_.set_title(title)

    plt.show()

    return score


# ---do NOT modify------Directory for validation data (provided by Instructor)----------------
directory = "./validation_data/"


# ----MODIFY PREPROCESSING PARAMETERS HERE: (num coefficients, num PCs, etc.)-----------------
num_mfcc=10 #number of cepstral coefficients

# ----MODIFY PREPROCESSING INPUT PARAMETERS AS  NEEDED----------------------------------------
X, Y_filename = preProcess(directory, num_mfcc) #preprocess




#--------------do NOT modify below this line---------------------------------------------------
# Load the model from disk (current directory) using pickle
filename = 'finalized_model.sav'
estimator = pickle.load(open(filename, 'rb'))

#Validation set labels (provided by Instructor)
Y_validationDict =  {'./validation_data/sound_00.wav' : 'drake', 
    './validation_data/sound_01.wav': 'drake',
    './validation_data/sound_02.wav': 'ed_sheeran',
    './validation_data/sound_03.wav': 'ed_sheeran', 
    './validation_data/sound_04.wav': 'justin_bieber', 
    './validation_data/sound_05.wav': 'justin_bieber', 
    './validation_data/sound_06.wav': 'linkin_park',
    './validation_data/sound_07.wav': 'linkin_park',
    './validation_data/sound_08.wav': 'taylor_swift', 
    './validation_data/sound_09.wav': 'taylor_swift'}

# Class labels
artists = ['drake', 'ed_sheeran', 'justin_bieber', 'linkin_park', 'taylor_swift']

# Get validation data accuracy and confusion matrix
accuracy = validationPredict(estimator, X, Y_filename, Y_validationDict, artists)







