import wave
import numpy as np
import matplotlib.pyplot as plt
from   scipy.signal import hilbert
from    scipy.fft import ifft,fft
import math
import time

elements_distance = 0.06
sound_velo = 344
file_name = 'vol_up0809_p60_1.wav'
num_sources = 1
num_elements = 3

FRAME_SAM = 2048

def pass_band(spectrum, fre_min, fre_max,fs):
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
    check_voice = 1

    # plt.plot(extractor)
    # plt.title('MUSIC (DOA )')
    # plt.legend(['P_music','Estimated DoAs'])

    # plt.show()    
    return check_voice, extractor


def music(CovMat,arr_size, Angles):
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
    # array = np.linspace(0, m -1 , num= m
    # av = np.array([1, 1/(1+ elements_distance * np.sin(Angles)), 1/(1 + 2 * elements_distance * np.sin(Angles))]).T
    av = np.array([1, 1, 1]).T
    av = av / np.linalg.norm(av)
    pspectrum = 1/((np.abs(av.transpose() @ Qn @ Qn.conj().transpose() @ av)))
    # pspectrum = np.log(pspectrum)

    return pspectrum

with wave.open(file_name, 'rb') as wav_file:
    # Lấy thông tin về file WAV
    frame_rate = wav_file.getframerate()
    num_frames = wav_file.getnframes()
    num_channels = wav_file.getnchannels()

    # Đọc dữ liệu từ các loa và chuyển đổi thành mảng NumPy
    wav_data = wav_file.readframes(num_frames)
    wav_array = np.frombuffer(wav_data, dtype=np.int16)
spectrum = np.zeros(361)
angles = np.linspace(-np.pi/2, np.pi/2, 361)
split_data = np.array_split(wav_array,1 )
for m, mang in enumerate(split_data):
    channel_1_raw = mang[0::3]
    channel_2_raw = mang[1::3]
    channel_3_raw = mang[2::3]
    print(channel_1_raw.shape)
    check_voice, extract_array = Vad(channel_1_raw, 40)
    idx = np.nonzero(extract_array)[0][0]
    idx_end = np.nonzero(extract_array)[0][-1]
    plt.plot(channel_1_raw )
    plt.plot(extract_array * 100)
    plt.plot(idx,extract_array[idx] *100, 'x')
    plt.plot(idx_end,extract_array[idx_end] * 100, 'x')
    plt.title('MUSIC (DOA )')
    plt.legend(['P_music','Estimated DoAs', 'start', 'end'])

    # channel_1_raw = channel_1_raw * extract_array
    # channel_2_raw = channel_2_raw * extract_array
    # channel_3_raw = channel_3_raw * extract_array
    plt.grid()
    plt.show()
    if(check_voice > 0):
        
        print("VADed")
        idx = np.nonzero(extract_array)[0][0]
        idx_end = np.nonzero(extract_array)[0][-1]
        print(idx)
        channel_1 = cut_array(channel_1_raw[idx :idx_end ], FRAME_SAM)
        channel_2 = cut_array(channel_2_raw[idx : idx_end], FRAME_SAM)
        channel_3 = cut_array(channel_3_raw [idx : idx_end], FRAME_SAM)
        extractor = cut_array(extract_array[idx : idx_end], FRAME_SAM)
        for k, mang in enumerate(extractor):
            if(sum(extractor[k]) >= 3/4 * FRAME_SAM):
                channel_1_fft = fft(channel_1[k] * np.hamming(FRAME_SAM) )
                print(channel_1[k])
                channel_2_fft = fft(channel_2[k] * np.hamming(FRAME_SAM))
                channel_3_fft = fft(channel_3[k] * np.hamming(FRAME_SAM))
                
                # plt.plot(channel_1[k] )
                
                # plt.title('MUSIC (DOA )')
                # plt.legend(['P_music','Estimated DoAs'])


                # plt.show()
                spec = []
                # su dung PSCM voi tung goc
                length = len(channel_2_fft)
                freqs = np.fft.fftfreq(length, d= 1/ frame_rate)
                
                array = np.linspace(0, length - 1, length)

                start = time.time()
                mic1_theta = pass_band(channel_1_fft, 150, 8000, frame_rate)
                mic2_theta = pass_band(channel_2_fft, 150, 8000, frame_rate)
                mic3_theta = pass_band(channel_3_fft, 150, 8000, frame_rate)
                for i in range(0, len(angles)):
                    freq = array * frame_rate / length
                    
                    
                    
                    mic1_data = mic1_theta
                    mic1_data = ifft(mic1_data)
                    # mic1 = ifft(mic1_theta)
                    loop_time = time.time()
                    
                    time1 = time.time()
                    mic2_data = mic2_theta * np.exp(1j * 2 * np.pi * freq *( elements_distance * np.sin(angles[i])/sound_velo))
                    time2 = time.time()
                    
                    mic2_data = ifft(mic2_data)
                    time3 = time.time()
                    # mic2 = ifft(mic2_theta)

                    
                    mic3_data = mic3_theta * np.exp(1j * 2 * np.pi * freq * 2* elements_distance * np.sin(angles[i])/sound_velo)
                    mic3_data = ifft(mic3_data)
                    
                    # mic3 = ifft(mic3_theta)
                    data = np.array([mic1_data, mic2_data, mic3_data])
                    # data_matrix = np.array([mic1_theta, mic2_theta, mic3_theta])
                    # R_theta = 0j
                    # for j in range(0, length//2 + 1):
                    #     freq = j * frame_rate // length
                    #     T_theta = np.diag([1, np.exp(1j * 2* np.pi * freq * elements_distance * np.sin(angles[i])/sound_velo) 
                    #                         , np.exp(2* 1j * 2* np.pi * freq * elements_distance * np.sin(angles[i])/sound_velo)])
                        
                    #     y_k = np.array([[data_matrix[0][j], data_matrix[1][j], data_matrix[2][j]]]).T
                        
                    #     CSDM_k = y_k @ y_k.conj().transpose()
                    #     Rk_theta = T_theta @ CSDM_k @ T_theta.conj().transpose()
                    #     R_theta = R_theta + Rk_theta
                    # eig_val, eig_vec = np.linalg.eig(R_theta)
                    # eig_max = max(np.abs(eig_val))

                    # data_mean = np.array([np.mean(data, axis= 1)]).T
                    # data = data - data_mean
                    CovMat =  data  @ data.conj().transpose()/(length - 1)
                    
                    spec_i = music(CovMat, num_elements, angles[i])
                    spec = np.append(spec, spec_i)
                
                # eig_val, eig_vec = np.linalg.eig(CovMat)
                # eig_max = max(np.abs(eig_val))
                # spec = np.append(spec, eig_max)
                
                    # print("time for fft: " + str(time1 - loop_time))
                    # print("time for cov: " + str(time2 - time1))
                    # print("time for music:" + str(time3 - time2))
                

                spectrum += np.log(spec)
                doa = np.argmax(spectrum)
                Angles_dgree = angles*180/np.pi
                plt.plot(Angles_dgree,spectrum )
                plt.plot(Angles_dgree[doa],spectrum[doa],'x')
                plt.title('MUSIC (DOA )')
                plt.legend(['P_music','Estimated DoAs'])


                print('MUSIC DoAs:',Angles_dgree[doa],'\n')
                plt.grid()
                # plt.show()
            else:
                print("done") 
                
        print("time for loop: " + str( time.time() - start))
