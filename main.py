import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import time
import socket
from broadbandMUSIC import *
import wave

elements_distance = 0.064
sound_velo =    348
num_sources =   1
num_elements =  4

samp_rate =     16000
FRAME_SIZE =    800
SAMP_WIDTH =    2   #bytes
CHUNK =         4*FRAME_SIZE #Bytes 4 channel 2 bytes

HOST = "192.168.4.1"       #dia chi IP cua esp32
PORT = 80
log_file_name = 'log.txt'



server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_address = (HOST, PORT)
print("Connecting to port")
try:
    server_socket.connect(server_address)
    print("Connected! Ready to receive data.")
    t = time.time()
    
    dec_data = np.frombuffer(recv_data(CHUNK* SAMP_WIDTH , server_socket), dtype= np.int16)

    # ham tinh tri so quyet dinh
    threshold = 15
    
    freqs = np.fft.fftfreq(FRAME_SIZE * 2, d= 1/ samp_rate)
    angles = np.linspace(-np.pi/2, np.pi/2, 181)
    array = np.linspace(0, FRAME_SIZE * 2 - 1, FRAME_SIZE * 2)
    data = np.zeros(CHUNK * 2)
    data[0: CHUNK] = dec_data
    Ws_func_1st = 0
    # print(threshold)
    spectrum = np.zeros(len(angles))
    Angles_dgree = angles*180/np.pi

    plt.ion()

    fig, ax = plt.subplots()
    line, = ax.plot(Angles_dgree, spectrum)

    plt.ylim(0, 150)
    plt.grid()
    plt.xlabel('degree')
    plt.ylabel('spec')

    
    while(True):
        count_frame = 1
        check = 0
        start_time = time.time()
        spectrum = np.zeros(len(angles))
        wf = open(log_file_name,'a')
        while (count_frame < 10):
            
            frame_data = recv_data(CHUNK*SAMP_WIDTH, server_socket)
            # print("received bytes:", len(frame_data))
            # print(len(data[CHUNK:]))
            data[CHUNK:] = np.frombuffer(frame_data, dtype= np.int16)
            channel_1 = data[0::4].astype(np.int32)
            channel_2 = data[1::4].astype(np.int32)
            channel_3 = data[2::4].astype(np.int32)
            channel_4 = data[3::4].astype(np.int32)
            
            Ws_func_2nd = snr(channel_1[FRAME_SIZE :])
            
            if(Ws_func_1st >= threshold and Ws_func_2nd >= threshold):
                # print(threshold)
                print(str(Ws_func_2nd), end = ' ')
                wf.write('snr: ' + str(Ws_func_2nd) + ' ')
                count_frame += 1 
                check = 1
                # print("voice frame:", count_frame)
                length = len(channel_1)
                channel_1_fft = fft(channel_1 * np.hamming(length))
                channel_2_fft = fft(channel_2 * np.hamming(length))
                channel_3_fft = fft(channel_3 * np.hamming(length))
                channel_4_fft = fft(channel_4 * np.hamming(length))
                spec = []
                # loai bo cac tan so khong can thiet
                mic1_theta = pass_band(channel_1_fft, 150, 2500, samp_rate)
                mic2_theta = pass_band(channel_2_fft, 150, 2500, samp_rate)
                mic3_theta = pass_band(channel_3_fft, 150, 2500, samp_rate)
                mic4_theta = pass_band(channel_4_fft, 150, 2500, samp_rate)
                # su dung PSCM voi tung goc
                for i in range(0, len(angles)):
                    freq = array * samp_rate / length
                        
                    mic1_j = mic1_theta
                    # mic1_data = ifft(mic1_j)

                    mic2_j = mic2_theta * np.exp(1j * 2 * np.pi * freq *( elements_distance * np.sin(angles[i])/sound_velo))
                    # mic2_data = ifft(mic2_j)

                    mic3_j = mic3_theta * np.exp(1j * 2 * np.pi * freq * 2* elements_distance * np.sin(angles[i])/sound_velo)
                    # mic3_data = ifft(mic3_j)
                    mic4_j = mic4_theta * np.exp(1j * 2 * np.pi * freq * 3* elements_distance * np.sin(angles[i])/sound_velo)

                    matrix_data = np.array([mic1_j, mic2_j, mic3_j, mic4_j])
                    CovMat = matrix_data @ matrix_data.conj().transpose()
                    spec_i = music(CovMat, num_elements, angles[i], num_sources)
                    spec = np.append(spec, spec_i)
                spectrum += spec
            # ngat khoi vong lap va xoa hang doi khi thoi gian xu ly qua lau
            loop_time = time.time() - start_time
            
            if(loop_time > 0.25 and count_frame < 10):
                clear_data = recv_data((10 - count_frame)* CHUNK * SAMP_WIDTH, server_socket)
                data[CHUNK:] = np.frombuffer(clear_data, dtype= np.int16)[-CHUNK:]
                break
            Ws_func_1st = Ws_func_2nd        
            data[:CHUNK] = data[CHUNK:]

        
        if check == 1:
            spectrum /= count_frame
        
            doa = np.argmax(spectrum)
            print('MUSIC DoAs:',str(Angles_dgree[doa]), end = '\n')
            wf.write('MUSIC DoAs:' + str(Angles_dgree[doa]) + '\n')
            line.set_ydata(spectrum)
            
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            # loop_time = time.time() - start_time

            # print("time:", loop_time)
        else:
            print("waiting...", end= '\r')

except socket.error as err:
    print ("socket creation failed with error %s" %(err))        
    


