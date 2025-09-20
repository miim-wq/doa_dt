import math
import queue
import threading
import warnings

import numpy as np
import pyaudio
import librosa
import tensorflow as tf

from gcc_phat import gcc_phat
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ----------------------------
# Array / geometry parameters
# ----------------------------
SOUND_SPEED = 343.2                # m/s
MIC_DISTANCE_4 = 0.08127           # meters between opposite mics in your 4-mic array
MAX_TDOA_4 = MIC_DISTANCE_4 / float(SOUND_SPEED)

# ----------------------------
# Audio / model parameters
# ----------------------------
CAPTURE_SR = 16000                 # mic capture rate (kept at 16 kHz for stable DOA)
CHANNELS = 4                       # 4-mic array
CHUNK = CAPTURE_SR // 4            # 0.25s per chunk (4000 samples per channel total interleaved)
SEGMENT_SEC = 0.5                  # run one classification every 0.5s
SEGMENT_SAMPLES = int(CAPTURE_SR  SEGMENT_SEC)

MODEL_SR = 22050                   # MFCC / model sample rate (matches your librosa setup)
N_MFCC = 64                       # MFCCs for the classifier
CONF_THRESHOLD = 0.60              # print confidence

TFLITE_PATH = "/home/pi/model.tflite"
TRAINING_LABELS = ["Environment", "Mavic", "Race", "UnknownDrone", "X5"]

# ============================================================
# MicArray: same logic as your original, cleaned/PEP8 styled
# ============================================================
class MicArray:
    def __init__(self, rate=CAPTURE_SR, channels=CHANNELS, chunk_size=CHUNK):
        self.p = pyaudio.PyAudio()
        self.queue = queue.Queue()
        self.quit_event = threading.Event()
        self.channels = channels
        self.sample_rate = rate
        self.chunk_size = chunk_size

        # Pick the first input device that exactly matches the required channels
        device_index = None
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            name = dev["name"]
            max_in = dev["maxInputChannels"]
            max_out = dev["maxOutputChannels"]
            print(i, name, max_in, max_out)
            if max_in == self.channels:
                print(f"Use {name}")
                device_index = i
                break

        if device_index is None:
            raise RuntimeError(
                f"Cannot find input device with {self.channels} channel(s)."
            )

        self.stream = self.p.open(
            input=True,
            start=False,
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=int(self.sample_rate),
            frames_per_buffer=int(self.chunk_size),
            stream_callback=self._callback,
            input_device_index=device_index,
        )

    def _callback(self, in_data, frame_count, time_info, status):
        self.queue.put(in_data)
        return None, pyaudio.paContinue

    def start(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.stream.start_stream()

    def read_chunks(self):
        self.quit_event.clear()
        while not self.quit_event.is_set():
            frames = self.queue.get()
            if not frames:
                break
            frames = np.frombuffer(frames, dtype=np.int16)
            yield frames

    def stop(self):
        self.quit_event.set()
        self.stream.stop_stream()
        self.queue.put(b"")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        self.stream.close()
        self.p.terminate()
        # re-raise exceptions if any
        if exc:
            return False

    # --- DOA estimation (unchanged logic) ---
    def get_direction(self, buf_int16):
        # buf_int16 is interleaved int16 for 4 channels
        MIC_GROUP_N = 2
        MIC_GROUP = [[0, 2], [1, 3]]

        tau = [0.0]  MIC_GROUP_N
        theta = [0.0] * MIC_GROUP_N

        # Use s

lices for

opposite mic pairs
        for i, pair in enumerate(MIC_GROUP):
            tau[i], _ = gcc_phat(
                buf_int16[pair[0]::4],
                buf_int16[pair[1]::4],
                fs=self.sample_rate,
                max_tau=MAX_TDOA_4,
                interp=1,
            )
            theta[i] = math.asin(tau[i] / MAX_TDOA_4) * 180.0 / math.pi

        if abs(theta[0]) < abs(theta[1]):
            if theta[1] > 0:
                best = (theta[0] + 360) % 360
            else:
                best = (180 - theta[0])
        else:
            if theta[0] < 0:
                best = (theta[1] + 360) % 360
            else:
                best = (180 - theta[1])

        # Your post-rotation adjustments
        best = (best + 90 + 180) % 360
        best = (-best + 120) % 360
        return best


# ============================================================
# Classifier setup (TFLite + label encoder)
# ============================================================
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
INPUT_DETAILS = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()

le = LabelEncoder()
le.fit(TRAINING_LABELS)

def classify_segment(mono_float, capture_sr=CAPTURE_SR):
    """
    mono_float: 1-D float32 audio in [-1, 1] at CAPTURE_SR
    Returns: (label:str, confidence:float)
    """
    # Resample to model/sample rate for MFCCs
    if capture_sr != MODEL_SR:
        y = librosa.resample(mono_float, orig_sr=capture_sr, target_sr=MODEL_SR)
        sr = MODEL_SR
    else:
        y = mono_float
        sr = capture_sr

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feature_scaled = np.mean(mfcc.T, axis=0).astype(np.float32)

    # Prepare and run inference
    x = np.expand_dims(feature_scaled, axis=0).astype(np.float32)
    interpreter.set_tensor(INPUT_DETAILS[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(OUTPUT_DETAILS[0]["index"])

    idx = int(np.argmax(output, axis=-1))
    conf = float(np.max(output))
    label = le.inverse_transform([idx])[0]
    return label, conf


# ============================================================
# Main: read chunks, compute DOA every chunk,
#       accumulate 0.5s for classification, print both.
# ============================================================
def main():
    # For building 0.5s segments from 0.25s chunks
    mono_buffer = np.zeros(0, dtype=np.float32)
    samples_needed = SEGMENT_SAMPLES

    print("Starting 4-mic localization + classification. Press Ctrl+C to stop.\n")

    try:
        with MicArray(rate=CAPTURE_SR, channels=CHANNELS, chunk_size=CHUNK) as mic:
            for interleaved in mic.read_chunks():
                # ---- DOA (uses int16 interleaved) ----
                direction_deg = mic.get_direction(interleaved)

                # ---- Build mono float for classifier ----
                # shape -> (-1, 4), average channels to mono, normalize to [-1,1]
                try:
                    frame_i16 = interleaved.reshape(-1, CHANNELS)
                except ValueError:
                    # In case of a partial buffer, skip gracefully
                    continue

                mono_i16 = frame_i16.mean(axis=1)  # float64 now
                mono_f32 = (mono_i16 / 32768.0).astype(np.float32)

                # Append and classify when we have >= 0.5s
                mono_buffer = np.concatenate([mono_buffer, mono_f32])
                if mono_buffer.size >= samples_needed:
                    segment = mono_buffer[:samples_needed]
                    mono_buffer = mono_buffer[samples_needed:]

                    label, conf = classify_segment(segment, capture_sr=CAPTURE_SR)
                    print(f"dir={int(round(direction_deg))}°, class={label}, conf={conf:.2f}")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == '__main__':
    main()
