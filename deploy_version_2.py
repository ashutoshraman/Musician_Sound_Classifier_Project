import pickle
from AudioFeatureExtractor import AudioFeatureExtractor
from driver import num_mfcc, nsegments

directory = input('Please enter directory name and do not put slash at end: ')

audioProcessor = AudioFeatureExtractor(directory)
audioFeatures = audioProcessor.constructMFCCFeatures(nsegments, num_mfcc)

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))


#loaded_model = joblib.load(filename)

print(loaded_model.score(audioFeatures.iloc[:, 1:], audioFeatures['Target'])) #this is accuracy maybe?
#add AUC score

