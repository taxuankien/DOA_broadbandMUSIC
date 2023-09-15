import wave
import numpy as np
import matplotlib.pyplot as plt
from   scipy.signal import hilbert
from    scipy.fft import ifft,fft


elements_distance = 0.06
sound_velo = 343
file_name = 'DOA_project/hello_p40_0.wav'
num_sources = 1
num_elements = 3
radius = 1
def pass_band(spectrum, fre_min, fre_max,fs):
    spectrum[np.where(freqs < fre_min)] = 0
    spectrum[np.where(freqs > fre_max)] = 0
    return spectrum

def music(CovMat,arr_size, Angles):
    # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # array holds the positions of antenna elements
    # Angles are the grid of directions in the azimuth angular domain
    eig_val,eig_vector = np.linalg.eig(CovMat)
    # tri rien va vector rieng can duoc xep tu lon den be 
    # eig_val = np.abs(eig_val)
    idx_min = np.argsort(eig_val)[:(arr_size - num_sources)]
    idx_min = np.flip(idx_min)
    m = arr_size
    Qn  = eig_vector[:,idx_min]
    # array = np.linspace(0, m -1 , num= m)

    av = np.array([1/radius, 1/np.sqrt((radius * np.sin(Angles)+0.06)**2 + (radius * np.cos(Angles))**2 ), 1/np.sqrt((radius * np.sin(Angles)+0.12)**2 + (radius * np.cos(Angles))**2 )]).T
    # av = np.array([1, 1, 1]).T
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
    channel_1 = wav_array[0:-1:3]
    channel_2 = wav_array[1:-1:3]
    channel_3 = wav_array[2::3]


channel_1_fft = fft(channel_1)
channel_2_fft = fft(channel_2)
channel_3_fft = fft(channel_3)

freqs = np.fft.fftfreq(len(channel_1), 1/frame_rate)
plt.subplot(2,1,1)
plt.plot(freqs[0:len(channel_1)//2], np.abs(channel_1_fft[:len(channel_1)//2]))
plt.plot(freqs[0:len(channel_1)//2], np.abs(channel_2_fft[:len(channel_1)//2]))
plt.plot(freqs[0:len(channel_1)//2], np.abs(channel_3_fft[:len(channel_1)//2]))
# plt.plot(channel_1)
# plt.plot(channel_2)
# plt.plot(channel_3)
plt.legend(['chan0', 'chan1', 'chan2'])
plt.grid()

channel_1_fft = fft(channel_1)
channel_2_fft = fft(channel_2)
channel_3_fft = fft(channel_3) 

spec = []
# su dung PSCM voi tung goc
length = len(channel_2_fft)
freqs = np.fft.fftfreq(length, d= 1/ frame_rate)
angles = np.linspace(-np.pi/2, np.pi/2, 361)
array = np.linspace(0, length - 1, length)
freq = array * frame_rate / length
mic1_theta = pass_band(channel_1_fft, 150, 8000, frame_rate)
mic2_theta = pass_band(channel_2_fft, 150, 8000, frame_rate)
mic3_theta = pass_band(channel_3_fft, 150, 8000, frame_rate)
for i in range(0, len(angles)):
    # time = array/ frame_rate
    
    mic1_j = mic1_theta
    mic1_data = ifft(mic1_j)
    # mic1 = ifft(mic1_theta)

    mic2_j = mic2_theta * np.exp(1j * 2 * np.pi * freq *( elements_distance * np.sin(angles[i])/sound_velo))
    mic2_data = ifft(mic2_j)
    
    # mic2 = ifft(mic2_theta)

    mic3_j = mic3_theta * np.exp(1j * 2 * np.pi * freq * 2* elements_distance * np.sin(angles[i])/sound_velo)
    mic3_data = ifft(mic3_j)
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
    CovMat = data @ data.conj().transpose()/length
    spec_i = music(CovMat, num_elements, angles[i])
    spec = np.append(spec, spec_i)

    # eig_val, eig_vec = np.linalg.eig(CovMat)
    # eig_max = max(np.abs(eig_val))
    # spec = np.append(spec, eig_max)

doa = np.argmax(spec)
Angles_dgree = angles*180/np.pi
plt.subplot(2,1,2)
plt.plot(Angles_dgree,spec)
plt.plot(Angles_dgree[doa],spec[doa],'x')
plt.title('MUSIC (DOA )')
plt.legend(['P_music','Estimated DoAs'])

print('MUSIC DoAs:',Angles_dgree[doa])
plt.grid()
plt.show()
