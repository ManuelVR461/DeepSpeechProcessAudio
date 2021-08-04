import sys
import pyaudio
import numpy as np
import audioop
import math
import collections
import wave
import librosa
import librosa.display
import matplotlib.pyplot as plt


class AudioData(object):
    def __init__(self, frame_data, sample_rate, sample_width, channels):
        self.frame_data = frame_data
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = int(sample_width)

class Microphone(object):
    def __init__(self):
        audio = pyaudio.PyAudio()
        try:
            self.FORMAT = pyaudio.paInt16
            self.SAMPLE_WIDTH = pyaudio.get_sample_size(self.FORMAT)
            device_info = audio.get_default_input_device_info()
            sample_rate = int(device_info["defaultSampleRate"])

        finally:
            audio.terminate()

        #Ajustes de Audio
        self.device_index = None #puede cambiarse al numero de dispositivo activo
        self.CHANNELS = 1
        self.SAMPLE_RATE = sample_rate
        self.CHUNK = 2048
        print([self.CHANNELS, self.FORMAT, self.SAMPLE_WIDTH,
              self.SAMPLE_RATE, self.CHUNK])

        #Ajustes de Ambiente
        self.energy_threshold = 4000
        self.dynamic_energy_threshold = True
        self.dynamic_energy_adjustment_damping = 0.15
        self.dynamic_energy_ratio = 1.5
        self.pause_threshold = 0.8 #Tiempo minimo considerado para la pausa
        self.phrase_threshold = 0.3 #Tiempo minimo de una frase
        self.non_speaking_duration = 0.5 #Tiempo minimo para considerar seguir grabando
        self.audiofile = 'DeepTrain/xample/audio2.wav'
        self.audio = None
        self.stream = None

    def __enter__(self):
        assert self.stream is None, "El Stream Esta Vacio o no se pudo inicializar"
        self.audio = pyaudio.PyAudio()
        try:
            self.stream = Microphone.MicrophoneStream(
                self.audio.open(
                    input_device_index=self.device_index,
                    channels=self.CHANNELS,
                    format=self.FORMAT,
                    rate=self.SAMPLE_RATE,
                    frames_per_buffer=self.CHUNK,
                    input=True
                )
            )
        except Exception:
            self.audio.terminate()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stream.close()
        finally:
            self.stream = None
            self.audio.terminate()

    def adjust_noise(self, source, duration=0.5):
        seconds_per_buffer = (source.CHUNK + 0.0) / source.SAMPLE_RATE
        elapsed_time = 0
        while True:
            elapsed_time += seconds_per_buffer
            if elapsed_time > duration:
                break
            buffer = source.stream.read(source.CHUNK)
            energy = audioop.rms(buffer, source.SAMPLE_WIDTH)
            damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer
            target_energy = energy * self.dynamic_energy_ratio
            self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)

    class MicrophoneStream(object):
        def __init__(self, pyaudio_stream):
            self.pyaudio_stream = pyaudio_stream

        def read(self, size):
            return self.pyaudio_stream.read(size, exception_on_overflow=False)

        def close(self):
            try:
                if not self.pyaudio_stream.is_stopped():
                    self.pyaudio_stream.stop_stream()
            finally:
                self.pyaudio_stream.close()

    def listen(self, source):
        seconds_per_buffer = float(source.CHUNK) / source.SAMPLE_RATE
        pause_buffer_count = int(math.ceil(self.pause_threshold / seconds_per_buffer)) # number of buffers of non-speaking audio during a phrase, before the phrase should be considered complete
        phrase_buffer_count = int(math.ceil(self.phrase_threshold / seconds_per_buffer))  # minimum number of buffers of speaking audio before we consider the speaking audio a phrase
        non_speaking_buffer_count = int(math.ceil(self.non_speaking_duration / seconds_per_buffer))  # maximum number of buffers of non-speaking audio to retain before and after a phrase

        elapsed_time = 0  # number of seconds of audio read
        buffer = b""  # an empty buffer means that the stream has ended and there is no data left to read

        while True:
            frames = collections.deque()

            # store audio input until the phrase starts
            while True:
                elapsed_time += seconds_per_buffer

                buffer = source.stream.read(source.CHUNK)
                if len(buffer) == 0:
                    break
                frames.append(buffer)

                # ensure we only keep the needed amount of non-speaking buffers
                if len(frames) > non_speaking_buffer_count:
                    frames.popleft()

                # detect whether speaking has started on audio input
                energy = audioop.rms(buffer, source.SAMPLE_WIDTH)  # energy of the audio signal
                if energy > self.energy_threshold: break

                # dynamically adjust the energy threshold using asymmetric weighted average
                if self.dynamic_energy_threshold:
                    # account for different chunk sizes and rates
                    damping = self.dynamic_energy_adjustment_damping ** seconds_per_buffer
                    target_energy = energy * self.dynamic_energy_ratio
                    self.energy_threshold = self.energy_threshold * damping + target_energy * (1 - damping)

            # read audio input until the phrase ends
            pause_count, phrase_count = 0, 0
            while True:
                # handle phrase being too long by cutting off the audio
                elapsed_time += seconds_per_buffer

                buffer = source.stream.read(source.CHUNK)
                if len(buffer) == 0:
                    break  # reached end of the stream
                frames.append(buffer)
                phrase_count += 1

                # check if speaking has stopped for longer than the pause threshold on the audio input
                # unit energy of the audio signal within the buffer
                energy = audioop.rms(buffer, source.SAMPLE_WIDTH)
                if energy > self.energy_threshold:
                    pause_count = 0
                else:
                    pause_count += 1
                if pause_count > pause_buffer_count:  # end of the phrase
                    break

            # check how long the detected phrase is, and retry listening if the phrase is too short
            phrase_count -= pause_count  # exclude the buffers for the pause before the phrase
            if phrase_count >= phrase_buffer_count or len(buffer) == 0:
                break  # phrase is long enough or we've reached the end of the stream, so stop listening

        # obtain frame data
        for i in range(pause_count - non_speaking_buffer_count): frames.pop()  # remove extra non-speaking frames at the end
        frame_data = b"".join(frames)

        return AudioData(frame_data, source.SAMPLE_RATE, source.SAMPLE_WIDTH, source.CHANNELS)

    def save(self, source):
        try:
            wavefile = wave.open(self.audiofile, 'wb')
            wavefile.setnchannels(source.channels)
            wavefile.setsampwidth(source.sample_width)
            wavefile.setframerate(source.sample_rate)
            wavefile.writeframes(source.frame_data)
            wavefile.close()
            print("Saved")
            return
        finally:
            pass

    def histogramAudio(self):
        out, sr = librosa.load(self.audiofile)

        #Duracion de un sample
        sample_duration = 1/sr
        print(f"Duracion del sample rate= {sample_duration:.6f} Segundos")

        #Duracion de la senal de audio en segundos
        duration = sample_duration * len(out)
        print(f"Duracion del audio = {duration:.2f} Segundos")

        #Visualizar  la forma de Onda
        plt.figure(figsize=(15, 17))
        plt.subplot(3, 1, 1)
        librosa.display.waveplot(out, alpha=0.5)
        plt.title("Audio Escuchado")
        plt.ylim((-1, 1))
        plt.show()

    def MFFCsAudio(self):
        out, sr = librosa.load(self.audiofile)
        mfccs = librosa.feature.mfcc(out, sr, n_mfcc=13, n_fft=2048,hop_length=512)
        mfccs.shape
        plt.figure(figsize=(25, 10))
        librosa.display.specshow(mfccs,x_axis="time",sr=sr)
        plt.colorbar(format="%+2.f")
        plt.show()


def main():
    #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
    escuchar = True
    mic = Microphone()
    while escuchar:
        with mic as source:
            mic.adjust_noise(source)
            audio = mic.listen(source)
            mic.save(audio)
            # mic.histogramAudio()
            mic.MFFCsAudio()
            escuchar = False


if __name__ == "__main__":
    main()
