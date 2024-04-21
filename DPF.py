import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time


def get_cepstral_coefficients_DPF(signal, num_coefficients):
    """
    Вычисляет кепстральные коэффициенты для заданного звукового сигнала.

    Аргументы:
    signal -- временной сигнал
    num_coefficients -- количество коэффициентов кепстра

    Возвращает:
    Массив кепстральных коэффициентов
    """

    # Шаг 1: Применяем оконную функцию Ханна к сигналу
    window = np.hanning(len(signal))
    windowed_signal = np.copy(signal) * window

    # Шаг 2: Выполнить ДПФ для получения спектра сигнала
    spectrum = np.fft.fft(windowed_signal)

    # Шаг 3: Вычислить логарифм от модуля спектра сигнала
    log_spectrum = np.log(np.abs(spectrum))

    # Шаг 4: Выполнить обратное ДПФ для получения кепстральных коэффициентов
    cepstrum = np.fft.ifft(log_spectrum)

    # Шаг 5: Выбрать первые num_coefficients коэффициентов
    cepstral_coefficients = np.real(cepstrum)[:num_coefficients]

    # Возвращаем первые num_coefficients коэффициентов кепстра
    return cepstral_coefficients[:num_coefficients]


# Загружаем звуковой файл
sample_rate, signal_mono = wavfile.read("voice1.wav")

# Вычисляем кепстральные коэффициенты
start = time.time()
num_coefficients = 10
cepstral_coefficients = get_cepstral_coefficients_DPF(signal_mono, num_coefficients)
end = time.time()

# Вычисляем спектр сигнала
spectrum = np.abs(np.fft.fft(signal_mono))

# Получаем время выполнения функции ДПФ
print("Получение кепстральных коэффициентов через ДПФ заняло:",
      (end-start) * 10**3, "ms")


# Создаем ось времени для графиков
time_axis = np.arange(0, len(signal_mono)) / sample_rate

# Строим график спектра сигнала
plt.subplot(211)
plt.plot(time_axis, signal_mono)
plt.title('Signal Spectrum DPF')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.grid()

# Строим график кепстральных коэффициентов
plt.subplot(212)
plt.plot(np.arange(num_coefficients), cepstral_coefficients)
plt.title('Cepstral Coefficients DPF')
plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.grid()

plt.tight_layout()
plt.show()
