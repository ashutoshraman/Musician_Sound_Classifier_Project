import numpy as np
from numpy.fft import fft, ifft
from matplotlib import pyplot as plt
import pandas as pd
import librosa  as lb
import librosa.display as lbd
from glob import glob

class AudioFeatureExtractor():
    def __init__(self,data_dir):
        self.audio_files = glob(data_dir+'/*.wav')
    
    def getTargetLabel(self, file):
        if 'ed_sheeran' in file:
            target = 'ed_sheeran'
        elif 'drake' in file:
            target = 'drake'
        elif 'taylor_swift' in file:
            target = 'taylor_swift'
        elif 'linkin_park' in file:
            target = 'linkin_park'
        elif 'justin_bieber' in file:
            target = 'justin_bieber'
        else:
            target ="does not exist"
            print("Target Not Found")

        return target

    def splitSignal(self, audioSignal, nsegments):
        signalLength = len(audioSignal)
        segmentLength = int(np.ceil(signalLength/nsegments))
        audioSegments = []
        for i in range(nsegments):
            first = i*segmentLength
            last = first  + segmentLength - 1
            if last > (signalLength-1):
                last  = signalLength-1
            audioSegments.append(audioSignal[first:last])
    
        return audioSegments
    
    def constructMFCCFeatures(self, nsegments=10, num_mfcc=20):
        column_labels=["Target"]
        for q in range(num_mfcc):
            column_labels.append("MFCC "+str(q))
        data = []

        for file in self.audio_files:
            audio_data, Fs = lb.load(file,sr=44100) #check sampling rate of sound signals
            segments = self.splitSignal(audio_data, nsegments)
            target = self.getTargetLabel(file)
            print(target)

            for j in range(nsegments):
                #D = lb.stft(data)
                D= lb.feature.mfcc(segments[j],Fs, n_mfcc=num_mfcc)
                s_db = np.mean(lb.amplitude_to_db(np.abs(D),ref = np.max),1)
                data_entry = [target] + s_db.tolist()
                data.append(data_entry)
        self.mfcc_data_frame = pd.DataFrame(data, columns = column_labels)
        return self.mfcc_data_frame
       

# plt.figure()
# lbd.specshow(s_db, x_axis='time', y_axis='log', sr=Fs, fmin=4, fmax=8)
# plt.colorbar(format="%+2.f dB")