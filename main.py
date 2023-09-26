import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import time
import socket
from broadbandMUSIC import *

elements_distance = 0.064
sound_velo =    346
num_sources =   1
num_elements =  4

samp_rate =     48000
FRAME_SIZE =    2400
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
    # t = time.time()
    # nguong cong suat quyet dinh nguon am duoc xu ly
    threshold = 15
    
    freqs = np.fft.fftfreq(FRAME_SIZE * 2, d= 1/ samp_rate)
    angles = np.linspace(-np.pi/2, np.pi/2, 181)
    array = np.linspace(0, FRAME_SIZE * 2 - 1, FRAME_SIZE * 2)
    data = np.zeros(CHUNK * 2)
    
    Ws_func_1st = 0
    # print(threshold)
    spectrum = np.zeros(len(angles))
    Angles_dgree = np.round(angles*180/np.pi).astype(np.int8)

    plt.ion()

    fig, ax = plt.subplots()
    line, = ax.plot(Angles_dgree, spectrum)

    plt.ylim(0, 150)
    plt.grid(which='major', axis='x', linestyle='-', linewidth=0.5)
    plt.grid(which='minor', axis='x', linestyle=':', linewidth=0.5)
    plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(5))
    plt.xlabel('degree')
    plt.ylabel('spec')

    dec_data = np.frombuffer(recv_data(CHUNK* SAMP_WIDTH , server_socket), dtype= np.int16)
    data[0: CHUNK] = dec_data

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
                channel_1_fft = fft(channel_1)
                channel_2_fft = fft(channel_2)
                channel_3_fft = fft(channel_3)
                channel_4_fft = fft(channel_4)
                spec = []
                # loai bo cac tan so khong can thiet
                channel_1 = ifft(pass_band(channel_1_fft, 150, 2500, samp_rate))
                channel_2 = ifft(pass_band(channel_2_fft, 150, 2500, samp_rate))
                channel_3 = ifft(pass_band(channel_3_fft, 150, 2500, samp_rate))
                channel_4 = ifft(pass_band(channel_4_fft, 150, 2500, samp_rate))
                # su dung PSCM voi tung goc
                for i in range(0, len(angles)):
                    
                    mic1_data = channel_1
                    mic2_data = arr_shift(channel_2, int(np.round(-1 * elements_distance * np.sin(angles[i])/sound_velo * samp_rate)))
                    mic3_data = arr_shift(channel_3, int(np.round(-2 * elements_distance * np.sin(angles[i])/sound_velo * samp_rate)))
                    mic4_data = arr_shift(channel_4, int(np.round(-3 * elements_distance * np.sin(angles[i])/sound_velo * samp_rate)))

                    n = int(abs(np.round(-3 * elements_distance * np.sin(angles[i])/sound_velo * samp_rate)))

                    matrix_data = np.array([mic1_data[n: (length - n)], mic2_data[n: (length - n)], mic3_data[n: (length - n)], mic4_data[n: (length - n)]])
                    CovMat = matrix_data @ matrix_data.conj().transpose()
                    spec_i = music(CovMat, num_elements, angles[i], num_sources)
                    spec = np.append(spec, spec_i)
                spectrum += spec
            # ngat khoi vong lap va xoa hang doi khi thoi gian xu ly qua lau
            loop_time = time.time() - start_time
            
            if(loop_time > 0.4 and count_frame < 10):
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
    


