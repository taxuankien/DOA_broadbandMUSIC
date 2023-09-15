import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import math

radius = 1

def recv_data(desire_len, server):
# Nhận dữ liệu cho đến khi nhận đủ kích thước đã xác định
    data = b''
    while len(data) < desire_len:
        packet = server.recv(desire_len - len(data))
        if not packet:
            break
        data += packet
        
    return data

def pass_band(spectrum, fre_min, fre_max,fs):
    freqs = np.fft.fftfreq(len(spectrum), d= 1/fs)
    spectrum[np.argwhere(freqs < fre_min)] = 0
    spectrum[np.argwhere(freqs > fre_max)] = 0
    return spectrum

def cut_array(array, Length, ovlap):
    result = []
    overlap = int(Length * ovlap)  # Tính overlap dựa trên 50% của L
    start = 0
    end = Length

    while end <= len(array):
        result.append(array[start:end])
        start = start + Length - overlap
        end = start + Length
    
    return result

def Ws(array):
    sign_array = np.sign(array)
    array = array ** 2
    power = np.sum(array)/len(array)
    sign_array[np.argwhere(sign_array == 0)] = 1
    zero_cross = 0
    for j in range(1, len(sign_array)):
        zero_cross = zero_cross + abs(sign_array[j] - sign_array[j - 1])/2
    zero_cross /= len(sign_array)

    return int(power) * (1 - zero_cross)*1000

def vad(data, frame_size):
    cut_data = cut_array(data, frame_size, 0)
    Ws_func = []
    # tinh Short time energy va Zero crossing rate voi tung frame
    for i,array in enumerate(cut_data):
        Ws_func = np.append(Ws_func, Ws(array))
    mean_Ws = np.mean(Ws_func)
    var_Ws = np.var(Ws_func)
    alpha = 0.3 * math.pow(var_Ws, -0.92)
    return mean_Ws + alpha * var_Ws

def music(CovMat,arr_size, Angles, num_sources):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    eig_val,eig_vector = np.linalg.eig(CovMat)
    # tri rien va vector rieng can duoc xep tu lon den be 
    eig_val = np.abs(eig_val)
    idx_min = np.argsort(eig_val)[:(arr_size - num_sources)]
    idx_min = np.flip(idx_min)
    m = arr_size
    Qn  = eig_vector[:,idx_min]
    # array = np.linspace(0, m -1 , num= m)

    av = np.array([1/radius, 1/np.sqrt((radius * np.sin(Angles)+0.06)**2 + (radius * np.cos(Angles))**2 ), 1/np.sqrt((radius * np.sin(Angles)+0.12)**2 + (radius * np.cos(Angles))**2 )]).T
    # av = np.array([1, 1, 1]).T
    av = av / np.abs(av)
    pspectrum = 1/((np.abs(av.transpose() @ Qn @ Qn.conj().transpose() @ av)))
    # pspectrum = np.log10(pspectrum)

    return pspectrum