#!/usr/bin/env python

"""Tools to encode and decode a signal.
Look at main for a usage example.
"""

import configparser
import binascii
import numpy as np
from scipy import signal
from scipy import fftpack
from scipy.io.wavfile import write
import matplotlib.pyplot as plt
from scipy.io import loadmat


def hammify(bits: 'numpy.array'):
    # NOTE: This only works for 8-bit bytes
    bits = list(bits)
    n = 8
    bytes = [bits[i*n:(i+1)*n] for i in range((len(bits)+n-1) // n)]
    if len(bytes[-1]) != 8:
        bytes = bytes[:-1]
    parity_positions = [0, 1, 3, 7]
    pos1 = [1, 3, 5, 7, 9, 11]
    pos2 = [2, 3, 6, 7, 10, 11]
    pos3 = [4, 5, 6, 7, 12]
    pos4 = [8, 9, 10, 11, 12]
    for byte in bytes:
        byte.insert(0, ' ')
        byte.insert(1, ' ')
        byte.insert(3, ' ')
        byte.insert(7, ' ')
        par1freq = [byte[i-1] for i in pos1].count(1)
        par2freq = [byte[i-1] for i in pos2].count(1)
        par3freq = [byte[i-1] for i in pos3].count(1)
        par4freq = [byte[i-1] for i in pos4].count(1)
        byte[0] = 0 if par1freq % 2 == 0 else 1
        byte[1] = 0 if par2freq % 2 == 0 else 1
        byte[3] = 0 if par3freq % 2 == 0 else 1
        byte[7] = 0 if par4freq % 2 == 0 else 1
    return np.array(bytes).ravel()


def dehammify(x, code_length=12):
    x = np.reshape(x, (-1, code_length))
    return np.array([message_from_code(fix_parity(y)) for y in x]).ravel()


def fix_parity(x):
    parity_bits = np.int(np.log2(np.size(x)))+1
    parity_fails = np.zeros(parity_bits)
    for i in np.arange(parity_bits):
        cons_check = 2**i
        parity_pos = np.int(2**i-1)
        sum = 0
        for check_pos in np.arange(parity_pos, np.size(x), cons_check*2):
            for bit_pos in np.arange(check_pos, min(check_pos+cons_check, np.size(x)), dtype=np.int32):
                sum = sum + x[bit_pos]
        if sum % 2 != 0:
            parity_fails[i] = 1

    num_errors = np.sum(parity_fails)
    if num_errors == 0:
        return x
    bit_to_flip = np.int(
        np.sum(np.power(2, np.arange(parity_bits))*parity_fails)-1)
    if bit_to_flip < np.size(x):
        x[bit_to_flip] = np.abs(x[bit_to_flip] - 1)
    return x


def message_from_code(x):
    parity_bits = np.int(np.log2(np.size(x)))+1
    mask = np.ones(np.size(x), dtype=bool)
    mask[np.power(2, np.arange(0, parity_bits))-1] = False
    return(x[mask])


def compute_phase(received_signal):
    """Compute the phase of the channel using the FT.

    Args:
        received_signal: The complex signal after it has been
            multiplied by cosine and sine.
        sampling_period: The sampling period in seconds.

    Returns:
        The estimated phase over time in the form of f * t + theta.
    """
    normalized_output = received_signal / received_signal.std()
    quarted_normalized_output = normalized_output**4
    fft = fftpack.fft(quarted_normalized_output)
    fftfreqs = fftpack.fftfreq(quarted_normalized_output.size) * 2 * np.pi
    # plt.plot(fftfreqs, np.abs(fft))
    # plt.show()

    # Take the second argmax because the highest one will be at 0 Hz.
    peak_location = np.abs(fft).argsort()[-1]
    peak_freq = fftfreqs[peak_location] * \
        np.pi / 2 / received_signal.size
    peak_angle = (np.angle(fft[peak_location])) / 4
    print(fftfreqs[peak_location])

    phase = peak_freq * np.arange(received_signal.size) + peak_angle
    return phase


def greycode(num):
    """Convert complex numbers to greycode.

    In clockwise order from the first quadrant, the codes are: 00, 01,
    11, and 10.

    Args:
        num: The complex number or array of complex numbers to
            greycode.

    Returns:
        An array of the number or numbers converted to graycode.
    """
    digit1 = num.real > 0
    digit0 = num.imag > 0
    return np.dstack([digit1, digit0]).ravel()


def modulate_signal(symbols, carrier_frequency, sampling_rate):
    """Modulate the signal.

    Args:
        symbols (array): The sequence of imaginary numbers to encode.
        carrier_frequency (float): The carrier frequency of the system.
        sampling_rate (float): The sampling rate of the system.

    Returns:
        The QPSK modulated signal.
    """
    times = np.arange(symbols.size) / sampling_rate
    trigarg = 2 * np.pi * carrier_frequency * times
    coeff = np.sqrt(2)
    in_phase = symbols.real * coeff * np.cos(trigarg)
    out_of_phase = symbols.imag * -coeff * np.sin(trigarg)
    return in_phase + out_of_phase


def demodulate_signal(received_signal, carrier_frequency, sampling_rate,
                      width):
    """Demodulate the signal.

    This does not correct the phase.
    To recover the signal completely, this output must be multiplied
    by exp(1j * computed_phase)

    Args:
        received_signal (array): The signal to demodulate.
        carrier_frequency (float): The carrier frequency of the signal.
        sampling_rate (float): The sampling rate.
        width: The cutoff frequency for the lowpass filter.

    Returns:
        The demodulated signal.
    """

    times = np.arange(received_signal.size) / sampling_rate
    trigarg = 2 * np.pi * carrier_frequency * times
    coeff = np.sqrt(2)
    sos = signal.butter(1, width, output='sos', fs=sampling_rate)
    reals = signal.sosfilt(
        sos, received_signal * coeff * np.cos(trigarg))
    imaginaries = signal.sosfilt(
        sos, received_signal * -coeff * np.sin(trigarg))
    return reals + imaginaries * 1j


def string_to_bits(message):
    bit_repr = bin(int(binascii.hexlify(bytearray(message, 'utf-8')), 16))
    bit_array = np.array(list(bit_repr))[2:].astype(bool)
    return np.concatenate([[0], bit_array])


def bits_to_string(message):
    reshaped_bytes = message.reshape(-1, 8)
    ords = reshaped_bytes.dot(2**np.arange(reshaped_bytes.shape[1])[::-1])
    letters = [str(chr(num)) for num in ords]
    return ''.join(letters)


def rotate_pair(pair):
    if np.all(pair == np.array([True, True])):
        return np.array([True, False])
    elif np.all(pair == np.array([True, False])):
        return np.array([False, False])
    elif np.all(pair == np.array([False, False])):
        return np.array([False, True])
    elif np.all(pair == np.array([False, True])):
        return np.array([True, True])


def rotate_message(message):
    rotated_message = []
    for pair in message.reshape(-1, 2):
        rotated_message.extend(rotate_pair(pair))
    return np.array(rotated_message)


def rotate_message_count(message, n):
    if n <= 0:
        return message
    else:
        return rotate_message_count(rotate_message(message), n - 1)


def find_best_rotation(predicted_message, actual_message):
    rotated_messages = np.stack([rotate_message_count(
        predicted_message, count) for count in range(4)])
    accuracies = (rotated_messages == actual_message).mean(1)
    return accuracies.argmax()


def main_sim():
    """Use the functions to simulate sending and decoding a signal."""

    config = configparser.RawConfigParser()
    config.read('system.cfg')

    carrier_frequency = config.getfloat('Values', 'carrier_frequency')
    sampling_rate = config.getfloat('Values', 'sampling_rate')
    width = config.getfloat('Values', 'width')
    symbol_period = config.getint('Values', 'symbol_period')

    message = np.load(config.get('Files', 'message'))
    message = hammify(message)
    impulse_response = np.load(config.get('Files', 'impulse_response'))
    white_header = np.load(config.get('Files', 'noise_header'))[:1000]

    # Convert bits into imaginary numbers
    keyed_message = message.copy()
    keyed_message[keyed_message == 0] = -1
    symbols = keyed_message[::2] + keyed_message[1::2] * 1j

    # Upsample the symbol for QPSK
    upsampled_symbols = symbols.repeat(symbol_period)

    # Transmit the signal
    transmitted_signal = modulate_signal(
        upsampled_symbols, carrier_frequency, sampling_rate)

    signal_with_header = np.concatenate([white_header, transmitted_signal])
    write(config.get('Files', 'signal_wav'),
          int(sampling_rate), signal_with_header)

    # Send the transmitted signal through a channel
    received_signal = signal.convolve(
        transmitted_signal, impulse_response)[:transmitted_signal.size]

    # Estimate the phase of the channel.
    estimated_phase = compute_phase(received_signal)

    # Recover the signal
    decoded_signal = demodulate_signal(received_signal,
                                       carrier_frequency, sampling_rate, width)

    corrected_signal = (decoded_signal * np.exp(1j * estimated_phase))
    # corrected_signal = decoded_signal

    # Sample the signal to recover the original metrics
    unrotated_predicted_message = greycode(
        corrected_signal.reshape(-1, symbol_period).mean(1))

    best_rotation = find_best_rotation(unrotated_predicted_message, message)

    predicted_message = rotate_message_count(
        unrotated_predicted_message, best_rotation)

    # Get metrics
    print('Accuracy:', (predicted_message == message).mean())
    # print('Accuracy:', accuracy1)


def main():
    """Use the functions to simulate sending and decoding a signal."""

    config = configparser.RawConfigParser()
    config.read('system.cfg')

    carrier_frequency = config.getfloat('Values', 'carrier_frequency')
    sampling_rate = config.getfloat('Values', 'sampling_rate')
    width = config.getfloat('Values', 'width')
    symbol_period = config.getint('Values', 'symbol_period')

    message = np.load(config.get('Files', 'message'))

    # received_signal = np.load(config.get('Files', 'received_signal'))[
    #     :hammify(message).size * symbol_period // 2]
    received_signal = loadmat('rx.mat')['signal'].ravel()[:hammify(message).size * symbol_period // 2]

    estimated_phase = compute_phase(received_signal)

    decoded_signal = demodulate_signal(received_signal,
                                       carrier_frequency, sampling_rate, width)

    corrected_signal = decoded_signal * np.exp(1j * estimated_phase)
    # averaged = corrected_signal.reshape(-1, symbol_period).mean(1)
    averaged = corrected_signal.reshape(-1, symbol_period)[:, 15:25].mean(1)

    # plt.plot(received_signal)
    # plt.show()

    # corr = signal.correlate(averaged, )
    # plt.plot(corr)
    plt.plot(averaged.real, averaged.imag, '.')
    ax = plt.gca()
    ax.set_xlabel('real part')
    ax.set_ylabel('imaginary part')
    ax.set_title('Constellation')
    # plt.show()
    plt.savefig('constellation.png')
    plt.close()
    # plt.plot(decoded_signal.real)
    # plt.show()

    unrotated_predicted_message = greycode(
        corrected_signal.reshape(-1, symbol_period).mean(1))
    print(received_signal.size)
    best_rotation = find_best_rotation(
        unrotated_predicted_message, hammify(message))
    predicted_message = rotate_message_count(
        unrotated_predicted_message, best_rotation)
    complex_message = message[::2] + message[1::2] * 1j
    corr = signal.correlate(averaged, complex_message)
    print('Accuracy:', (dehammify(predicted_message) == message).mean())
    print('Accuracy:', (predicted_message == hammify(message)).mean())
    print('Actual Message is:', bits_to_string((message)))
    print('Received Message :', bits_to_string(dehammify(predicted_message)))
    ax = plt.gca()
    ax.set_title('Error Concentration')
    plt.plot(predicted_message == hammify(message))
    plt.savefig('errs.png')


if __name__ == '__main__':
    main()
