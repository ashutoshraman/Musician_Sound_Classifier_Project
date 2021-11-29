import numpy as np
# from numpy.fft import fft, ifft
from matplotlib import pyplot as plt
import pandas as pd
import librosa  as lb
import librosa.display as lbd
from glob import glob
from scipy.fft import fft

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
            n = len(audio_data) # do this to make sure all songs are around 8 to 10s long
            nfft = int(2**(np.ceil(np.log2(n))))
            
            if nfft == 524288: #delete anything that doesn't fit 8 to 10 sec long
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

    def performFFT(self, sr=44100, freq_cutoff = 1000):
        data = []

        for file in self.audio_files:
            audio_data, sr = lb.load(file, sr=sr) 
            n = len(audio_data)
            #Find next power of 2 that is larger than the signal length
            #then perform FFT
            nfft = int(2**(np.ceil(np.log2(n))))

            if nfft == 524288: #bad to hard code but no other choice, many ed sheeran files are not to format and mess everything up
                signal_fft = fft(audio_data,n=nfft,norm='ortho')
                #Return one-sided FFT.
                half_signal=int(np.ceil(nfft/2))
                signal_fft=signal_fft[0:half_signal+1]
                freqs = lb.fft_frequencies(sr,nfft)
                #Report frequencies below cutoff
                cutoff = np.where(freqs < freq_cutoff)
                cut_freqs = freqs[cutoff]
                cut_signal = np.abs(signal_fft[cutoff])
                # print(cut_signal.shape)
                target = self.getTargetLabel(file)
                print(target)

                data_entry = [target] + cut_signal.tolist()
                data.append(data_entry)

        column_labels= ['Target']
        for q in range(len(cut_signal)):
            column_labels.append('Freq ' + str(q))

        self.dft_data_frame = pd.DataFrame(data, columns = column_labels)
        return self.dft_data_frame

       

# plt.figure()
# lbd.specshow(s_db, x_axis='time', y_axis='log', sr=Fs, fmin=4, fmax=8)
# plt.colorbar(format="%+2.f dB")