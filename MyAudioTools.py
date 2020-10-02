from pydub import AudioSegment
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import librosa

def compress_log(values, gamma=100):
    '''values is a np array, 
    gamma is the compression constant'''
    return np.log(np.ones(values.shape) + gamma*values)

class MyAudioSegment(AudioSegment):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.stft = None
    
    def is_mono(self):
        return self.channels == 1
    
    def samples(self):
        '''returns left channel samples'''
        if self.is_mono():
            return self.get_array_of_samples()
        else:
            print("Warning! Audio has two channels!")
            return self.get_array_of_samples()[::2]

    def get_timeline(self,offset=0):
        N = int(self.frame_rate * self.duration_seconds)
        return np.linspace(offset,offset+self.duration_seconds,N)

    def plot_waveform(self,offset=0):
        '''Time amplitude plot for channel left=0 , right=1'''
        plt.xlabel('Time [seconds]')
        plt.ylabel('Amplitude')
        plt.plot(
                self.get_timeline(offset),
                self.samples()
        )

    def compute_fft(self, window=None):
        if window == 'boxcar':
            window = signal.windows.boxcar(len(self.monosamples()))
        elif window == 'hann':
            window = signal.windows.hann(len(self.monosamples()))
        window_samples = [w * x for (w,x) in zip(window,self.monosamples())]
        return np.fft.rfft(window_samples)

    def compute_stft(self, w_length, hop_factor=0.5):
        "hop factor is hop size/window size"
        self.stft_w_length = w_length
        self.stft = signal.stft(
                                self.samples(),
                                fs=self.frame_rate,
                                nperseg=int(self.frame_rate*w_length),
                                noverlap=int(self.frame_rate*w_length*(1-hop_factor)),
        )
        return self.stft 
    
    def compute_chroma(self, spectrogram=None, window_size=None, hop_size=None):
        if window_size == None:
            N = 2048
        else:
            N = int(window_size*self.frame_rate)
        if hop_size == None:
            H = 512
        else:
            H = int(hop_size*self.frame_rate)
        return librosa.feature.chroma_stft(
                y=np.array(self.monosamples()).astype(np.float),
                sr=self.frame_rate,
                S=spectrogram,
                n_fft=N,
                hop_length=H
        )
    
    def log_freq_spectrogram(self, w_length=None, hop_factor=None):
        if hop_factor == None:
            hop_factor = 0.5
        if self.stft is None:
            stft = self.compute_stft(w_length, hop_factor)[2].T
        else:
            stft = self.stft[2].T
        N, K = stft.shape
        
        def phys_freq(coef):
            return coef/self.stft_w_length
        
        def center_freq(p):
            return 440*2**((p-69)/12)
        P_upper = [center_freq(p+0.5) for p in range(128)]

        log_freq = np.zeros((N,128))
        chroma = np.zeros((N,12))
        
        for time_frame in range(N):
            # Sum up squared magnitudes of coefficients belonging to pitch class
            for fourier_index in range(K):
                for pitch in range(128):
                    if P_upper[pitch] > phys_freq(fourier_index):
                        break          
                log_freq[time_frame][pitch] += np.abs(stft[time_frame][fourier_index])**2
                chroma[time_frame][pitch%12] += np.abs(stft[time_frame][fourier_index])**2
        
        self.log_freq = log_freq
        self.chroma = chroma
        return log_freq
    
    def plot_lf_spectrogram(self):
        ax = plt.axes()
        lf = self.log_freq
        N, P = lf.shape
        ax.pcolormesh(np.linspace(0, self.duration_seconds, N),
                        np.arange(P), 
                        lf.T, 
                        vmin=0, shading='auto')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MIDI pitch")
        ax.set_ylim([50,120])
        return ax
    
    def plot_chroma(self, gamma=None):
        ax = plt.axes()
        chroma = self.chroma
        if gamma is not None:    
            chroma = np.log(np.ones(chroma.shape) + gamma*chroma)
        N, C = chroma.shape
        ax.pcolormesh(np.linspace(0, self.duration_seconds, N),
                        np.arange(C), 
                        chroma.T, 
                        vmin=0, shading='auto')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Note")
        return ax

    def compute_mel(self, w_length, hop_size, nfft):
        return librosa.feature.melspectrogram(
                                                y=np.array(self.monosamples()).astype(np.float),
                                                sr=self.frame_rate,
                                                n_fft=nfft, 
                                                hop_length=math.floor(self.frame_rate*hop_size), 
                                                win_length=math.floor(self.frame_rate*w_length),
                                                n_mels=128
        )
    
    def compute_mfcc(self, n_mfcc, win_length, hop_length, nfft):
        return librosa.feature.mfcc(
                                    y=np.array(self.monosamples()).astype(np.float),
                                    sr=self.frame_rate,
                                    n_mfcc=20,
                                    win_length=math.floor(self.frame_rate*win_length),
                                    hop_length=math.floor(self.frame_rate*hop_length),
                                    n_fft=nfft
        )

    def count_crossings(self):
        count = 0
        samples = self.monosamples()
        for i in range(0, len(self)):
            if samples[i]*samples[i+1] < 0:
                count += 1
        return count

    def spectral_centroid(self):
        assert len(self.monosamples()) % 2 == 0
        fs = self.frame_rate
        N = len(self.monosamples())
        fourier = np.abs(np.fft.rfft(self.monosamples()))
        den = sum([fs*k*fourier[k]/N for k in range(N//2+1)])
        num = sum([fourier[k] for k in range(N//2+1)])
        return den/num