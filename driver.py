import MFCC
import numpy as np
from AudioFeatureExtractor import AudioFeatureExtractor


# Feature extraction.  
#Number of segments to split signal
nsegments=1
#number of cepstral coefficients
num_mfcc=20
# Number of samples to use for testing
t_size = 0.5

directory = "./project_2_data/ed_sheeran"
audioProcessor = AudioFeatureExtractor(directory)
audioFeatures = audioProcessor.constructMFCCFeatures(nsegments, num_mfcc)
#print(audioFeatures)
print(np.unique(audioFeatures["Target"]))

print(audioProcessor.audio_files)
