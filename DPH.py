import numpy as np
import scipy.fftpack
import scipy.io.wavfile
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time


def get_cepstral_coefficients_DPH(signal, num_coefficients):
    """
    Вычисляет кепстральные коэффициенты для заданного сигнала.

    Аргументы:
    signal -- временной сигнал
    num_coefficients -- количество коэффициентов кепстра

    Возвращает:
    Массив кепстральных коэффициентов
    """

    # Применяем оконную функцию Ханна к сигналу
    windowed_signal = signal * np.hanning(len(signal))

    # Вычисляем дискретное преобразование Хартли (DHT)
    dht = scipy.fftpack.dct(windowed_signal, type=2, norm='ortho')

    # Вычисляем логарифмический спектр
    log_spectrum = np.log(np.abs(dht))

    # Вычисляем обратное DHT для логарифмического спектра
    cepstral_coefficients = scipy.fftpack.idct(log_spectrum, type=2, norm='ortho')

    # Возвращаем первые num_coefficients коэффициентов кепстра
    return cepstral_coefficients[:num_coefficients]


# Загружаем звуковой файл
sample_rate, signal_mono = wavfile.read("voice1.wav")

# Вычисляем кепстральные коэффициенты
start = time.time()
num_coefficients = 10
cepstral_coefficients = get_cepstral_coefficients_DPH(signal_mono, num_coefficients)
end = time.time()

# Вычисляем спектр сигнала
spectrum = np.abs(np.fft.fft(signal_mono))

# Создаем ось времени для графиков
time_axis = np.arange(0, len(signal_mono)) / sample_rate

# Получаем время выполнения функции ДПФ
print("Получение кепстральных коэффициентов через ДПХ заняло:",
      (end-start) * 10**3, "ms")


# Строим график спектра сигнала
plt.subplot(211)
plt.plot(time_axis, signal_mono)
plt.title('Signal Spectrum DPH')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.grid()

# Строим график кепстральных коэффициентов
plt.subplot(212)
plt.plot(np.arange(num_coefficients), cepstral_coefficients)
plt.title('Cepstral Coefficients DPH')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.grid()

plt.tight_layout()
plt.show()
