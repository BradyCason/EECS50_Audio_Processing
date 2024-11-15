import librosa
import numpy as np
from scipy.signal import butter, filtfilt, lfilter
import soundfile as sf
import matplotlib.pyplot as plt

def low_pass_filter(data, cutoff, sr, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    print("Low Pass Transfer Function (H(z)):")
    print("Numerator (b):", b)
    print("Denominator (a):", a)
    y_filtered = filtfilt(b, a, data)
    return y_filtered

def high_pass_filter(data, cutoff, sr, order=5):
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    print("High Pass Transfer Function (H(z)):")
    print("Numerator (b):", b)
    print("Denominator (a):", a)
    y_filtered = filtfilt(b, a, data)
    return y_filtered

def band_pass_filter(data, low_cutoff, high_cutoff, sr, order=5):
    nyquist = 0.5 * sr
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    print("High Pass Transfer Function (H(z)):")
    print("Numerator (b):", b)
    print("Denominator (a):", a)
    y_filtered = filtfilt(b, a, data)
    return y_filtered

def plot_frequency_spectrum(y, sr, title="Frequency Spectrum"):
    # Compute the Fast Fourier Transform (FFT)
    freqs = np.fft.rfftfreq(len(y), d=1/sr)
    fft_magnitude = np.abs(np.fft.rfft(y))
    
    # Plot the frequency spectrum
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_magnitude, color='blue')
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, sr // 2)  # Show only up to Nyquist frequency
    plt.show()

def main():
    filename = librosa.example('nutcracker')
    y, sr = librosa.load(filename, sr=None)

    # Set the cutoff frequency (example: 3000 Hz)
    cutoff_frequency = 1500
    low_frequency = 1000
    high_frequency = 2000

    # Apply low-pass filter
    # y_filtered = low_pass_filter(y, cutoff_frequency, sr)
    # y_filtered = high_pass_filter(y, cutoff_frequency, sr)
    y_filtered = band_pass_filter(y, low_frequency, high_frequency, sr)

    # Save the filtered audio
    sf.write('filtered_audio.wav', y_filtered, sr)

    plot_frequency_spectrum(y, sr, title="Original Audio Frequency Spectrum")
    plot_frequency_spectrum(y_filtered, sr, title="Filtered Audio Frequency Spectrum")

if __name__ == "__main__":
    main()