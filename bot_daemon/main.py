import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

def main():
    model = "british.onnx"
    voice = PiperVoice.load(model)
    stream = sd.OutputStream(samplerate=voice.config.sample_rate, channels=1, dtype='int16')
    stream.start()

    while True:
        try:
            new_string = input()
            for audio_bytes in voice.synthesize_stream_raw(new_string):
                int_data = np.frombuffer(audio_bytes, dtype=np.int16)
                stream.write(int_data)
        except:
            break
    stream.stop()
    stream.close()


if __name__ == "__main__":
    main()
