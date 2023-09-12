import numpy as np
import math



def pass_band(spectrum, fre_min, fre_max,fs):
    freqs = np.fft.fftfreq(len(spectrum), d = 1/fs)
    spectrum[np.where(freqs < fre_min)] = 0
    spectrum[np.where(freqs > fre_max)] = 0
    return spectrum

def cut_array(array, L):
    result = []
    overlap = int(L * 0.5)  # Tính overlap dựa trên 50% của L

    start = 0
    end = L

    while end <= len(array):
        result.append(array[start:end])
        start = start + L - overlap
        end = start + L
    return result

# Voice activity detection function
def Vad(raw_data, num_frame):
    frame_data = np.array(np.array_split(raw_data, num_frame))
    # frame_data = np.copy(frame_data)
    W_function = []
    for i in range(num_frame):
        
        raw_frame = np.array([frame_data[i]])
        sign_array = np.zeros(len(raw_frame))
        array = raw_frame ** 2
        power = np.sum(array)/len(raw_frame)
        zero_cross = 0
        sign_array = np.sign(raw_frame).T
        print(sign_array.shape)
        sign_array[np.argwhere(sign_array == 0)] = 1
        for j in range(1, sign_array.shape[1]):
            zero_cross = zero_cross + abs(sign_array[j] - sign_array[j - 1])/2
        zero_cross /= len(sign_array)
        W_function =np.append(W_function,int(power * (1 - zero_cross)*1000))
    mean_W = np.mean(W_function)
    var_W = np.var(W_function)
    alpha = 0.3 * math.pow(var_W, -0.92)
    trigger = mean_W + alpha * var_W
    W_function = W_function - trigger
    W_function = np.sign(W_function)
    frame_data[np.argwhere(W_function >= 0)] = 1
    frame_data[np.argwhere(W_function < 0)] = 0
    extractor = frame_data.flatten()

    #them code xu ly ca doan am thanh co duoc xu ly hay khong (voice chien 60 phan tram thi la 1)
    check_voice = 1
 
    return check_voice, extractor

def music(CovMat,arr_size, Angles, num_sources):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    eig_val,eig_vector = np.linalg.eig(CovMat)
    # tri rieng va vector rieng can duoc xep tu lon den be 
    eig_val = np.abs(eig_val)
    idx_min = np.argsort(eig_val)[:(arr_size - num_sources)]
    idx_min = np.flip(idx_min)
    m = arr_size
    Qn  = eig_vector[:,idx_min]
    # array = np.linspace(0, m -1 , num= m)

    # av = np.array([1, 1/np.exp(elements_distance * np.sin(Angles)), 1/np.exp(2 * elements_distance * np.sin(Angles))]).T
    av = np.array([1, 1, 1]).T
    av = av / np.abs(av)
    pspectrum = 1/((np.abs(av.transpose() @ Qn @ Qn.conj().transpose() @ av)))
    pspectrum = np.log10(pspectrum)

    return pspectrum
