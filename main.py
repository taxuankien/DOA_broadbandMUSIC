import numpy as np
import matplotlib.pyplot as plt
from    scipy.fft import ifft,fft
from   scipy.signal import hilbert
import socket
import scipy.signal._peak_finding as ss_pf
from broadbandMUSIC import *
import time


sound_velo = 343
elements_distance = 0.06
num_sources = 1
num_elements = 3
frame_rate = 16000
CHUNK = 176400
SAMPLE_WIDTH = 2 #BYTES 

HOST = '192.168.4.1' # dia chi IP cua may tinh
PORT0 = 80          # cong ket noi

FRAME_SIZE = 2048
server_socket0 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address_0 = (HOST, PORT0)
print("connecting to ports ")

def recv_data(desire_len):
# Nhận dữ liệu cho đến khi nhận đủ kích thước đã xác định
    data = b''
    while len(data) < desire_len:
        packet = server_socket0.recv(desire_len - len(data))
        if not packet:
            break
        data += packet
        
    return data


result = server_socket0.connect(server_address_0)

while(result):
    
    data0 = recv_data(CHUNK)
    # print(len(data0))
    wav_array = np.frombuffer(data0, dtype= np.int16)
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
        # plt.plot(channel_1_raw )
        # plt.plot(extract_array * 100)
        # plt.plot(idx,extract_array[idx] *100, 'x')
        # plt.plot(idx_end,extract_array[idx_end] * 100, 'x')
        # plt.title('MUSIC (DOA )')
        # plt.legend(['P_music','Estimated DoAs', 'start', 'end'])

        # plt.grid()
        # plt.show()
        if(check_voice > 0):
            
            print("VADed")
            idx = np.nonzero(extract_array)[0][0]
            idx_end = np.nonzero(extract_array)[0][-1]
            print(idx)
            channel_1 = cut_array(channel_1_raw[idx :idx_end ], FRAME_SIZE)
            channel_2 = cut_array(channel_2_raw[idx : idx_end], FRAME_SIZE)
            channel_3 = cut_array(channel_3_raw [idx : idx_end], FRAME_SIZE)
            extractor = cut_array(extract_array[idx : idx_end], FRAME_SIZE)
            for k, mang in enumerate(extractor):
                if(sum(extractor[k]) >= 3/4 * FRAME_SIZE):
                    channel_1_fft = fft(channel_1[k] * np.hamming(FRAME_SIZE) )
                    print(channel_1[k])
                    channel_2_fft = fft(channel_2[k] * np.hamming(FRAME_SIZE))
                    channel_3_fft = fft(channel_3[k] * np.hamming(FRAME_SIZE))

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
                        loop_time = time.time()
                        
                        mic2_data = mic2_theta * np.exp(1j * 2 * np.pi * freq *( elements_distance * np.sin(angles[i])/sound_velo))
                        mic2_data = ifft(mic2_data)
                        
                        mic3_data = mic3_theta * np.exp(1j * 2 * np.pi * freq * 2* elements_distance * np.sin(angles[i])/sound_velo)
                        mic3_data = ifft(mic3_data)
                        
                        data = np.array([mic1_data, mic2_data, mic3_data])
                       

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
            print("time for loop: " + str( time.time() - start))
print('Disconnected')