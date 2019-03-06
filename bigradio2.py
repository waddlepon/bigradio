import pyaudio
import numpy as np
import math
import multiprocessing
import functools
import time

cos_columns = {}

def calculate_cos(k, n, t):
    global cos_columns
    cos_column = np.array(list(map(lambda x: np.cos((n + k)*x), t)))
    cos_columns[(n, k)] = cos_column

def calculate_cos_column(n, length):
    t = np.linspace(0, np.pi, length)
    with multiprocessing.Pool(10) as p:
        p.map(functools.partial(calculate_cos, n=n, t=t), range(20, 2000))

def compute_integral(signals, n, k):
    if (n, k) not in cos_columns:
        t = np.linspace(0, np.pi, len(signals))
        cos_column = np.array(list(map(lambda x: np.cos((n + k)*x), t)))
        cos_columns[(n, k)] = cos_column
    else:
        cos_column = cos_columns[(n, k)]

    integral = np.trapz(signals*cos_column,dx=np.pi/(len(signals)-1))*(4/np.pi)
    return integral

def find_coefficients(signals, n):
    coefficients = []
    for i in range(20, 2000):
        ans = round(compute_integral(signals, n, i*2*np.pi))
        if ans != 0:
            coefficients.append((ans, i))
            if len(coefficients) >= 3:
                coefficients = tuple(coefficients)
                return coefficients

def play_sine_waves(coes, duration):
    p = pyaudio.PyAudio()
    fs = 44100
    volume = 0.5
    
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
    for c in coes:
        samples = (c[0][0]*np.sin(2*np.pi*np.arange(fs*duration)*c[0][1]/fs) + c[1][0]*np.sin(2*np.pi*np.arange(fs*duration)*c[1][1]/fs) + c[2][0]+np.sin(2*np.pi*np.arange(fs*duration)*c[2][1]/fs)).astype(np.float32)
        stream.write(volume*samples)
    stream.stop_stream()
    stream.close()
    p.terminate()

def big_calc(n):
    data = np.transpose(np.genfromtxt('am_radio_extension.csv', delimiter=','))

    start_time = time.time()
    calculate_cos_column(n, len(data[0]))
    print('Calculating cosines took: ' + str(time.time() - start_time) + ' seconds')

    all_coefficients = []

    start_time = time.time()
    with multiprocessing.Pool(10) as p:
        all_coefficients = np.array(list(p.map(functools.partial(find_coefficients, n=n), data)))
    print('Calculating coefficients took: ' + str(time.time() - start_time) + ' seconds')
    return all_coefficients

def main():
    freq = float(input('frequency: '))
    n = freq*(2*np.pi)
    interval = float(input('interval: '))

    all_coefficients = []
    try:
        all_coefficients = np.load('big_coefficients_' + str(freq) + '.npy')
    except:
        all_coefficients = big_calc(n)
        np.save('big_coefficients_' + str(freq) + '.npy', all_coefficients)
    
    print('Sound time!!')
    play_sine_waves(all_coefficients, interval)

if __name__ == '__main__':
    main()
