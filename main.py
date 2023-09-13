import numpy as np
import matplotlib as plt
from scipy.fft import fft, ifft
import time
import socket
from broadbandMUSIC import *

elements_distance = 0.06
sound_velo =    344
num_sources =   1
num_elements =  3

samp_rate =     44100 
FRAME_SIZE =    2205
CHUNK =         3*2205*2 #Bytes 3 channel 2 bytes

HOST = "192.168.4.1"
PORT = 80

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = (HOST, PORT)
print("Connecting to port")
if(not server_socket.connect(server_address)):
    print("Failed to connecting")
else:
    print("Connected! Ready to receive data.")
    # coi 10 frame dau la nhieu de tinh tri so quyet dinh mau la noise hay voice
    dec_data = np.frombuffer(recv_data(CHUNK * 10, server_socket), dtype= np.int16)

    channel_1 = dec_data[0::3]
    channel_2 = dec_data[1::3]
    channel_3 = dec_data[2::3]
    # ham tinh tri so quyet dinh
    threshold = vad(channel_1, FRAME_SIZE)
    freqs = np.fft.fftfreq(FRAME_SIZE, d= 1/ samp_rate)
    angles = np.linspace(-np.pi/2, np.pi/2, 361)
    array = np.linspace(0, FRAME_SIZE - 1, FRAME_SIZE)
    while(True):
        count_frame = 0
        check = 0
        start_time = time.time()
        spectrum = np.zeros(len(angles))
        while (count_frame < 20):
            count_frame += 1 
            data = recv_data(CHUNK, server_socket)
            channel_1 = data[0::3]
            channel_2 = data[1::3]
            channel_3 = data[2::3]
            
            Ws_func = Ws(channel_1)
            if(Ws_func >= threshold):
                check = 1
                channel_1_fft = fft(channel_1)
                channel_2_fft = fft(channel_2)
                channel_3_fft = fft(channel_3)
                length = len(channel_1_fft)
                spec = []
                # su dung PSCM voi tung goc
                mic1_theta = pass_band(channel_1_fft, 150, 8000, samp_rate)
                mic2_theta = pass_band(channel_2_fft, 150, 8000, samp_rate)
                mic3_theta = pass_band(channel_3_fft, 150, 8000, samp_rate)
                for i in range(0, len(angles)):
                    freq = array * samp_rate / length
                        
                    mic1_j = mic1_theta
                    mic1_data = ifft(mic1_j)

                    mic2_j = mic2_theta * np.exp(1j * 2 * np.pi * freq *( elements_distance * np.sin(angles[i])/sound_velo))
                    mic2_data = ifft(mic2_j)

                    mic3_j = mic3_theta * np.exp(1j * 2 * np.pi * freq * 2* elements_distance * np.sin(angles[i])/sound_velo)
                    mic3_data = ifft(mic3_j)

                    data = np.array([mic1_data, mic2_data, mic3_data])
                    CovMat = data @ data.conj().transpose()
                    spec_i = music(CovMat, num_elements, angles[i], num_sources)
                    spec = np.append(spec, spec_i)
                spectrum += spec

            loop_time = time.time() - start_time
            if(loop_time >= 0.9 and count_frame < 18):
                clear_data = recv_data((20 - count_frame)* CHUNK, server_socket)
                break
        if check == 1:
            doa = np.argmax(spectrum)
            Angles_dgree = angles*180/np.pi
            # plt.plot(Angles_dgree,spectrum )
            # plt.plot(Angles_dgree[doa],spectrum[doa],'x')
            # plt.title('MUSIC (DOA )')
            # plt.legend(['P_music','Estimated DoAs'])
            print('MUSIC DoAs:',Angles_dgree[doa],'\n')
            # plt.grid()
            # plt.show()
        else:
            print("waiting...")
                
    


