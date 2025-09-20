"""
DOA (UCA-RB-MUSIC) + TFLite classifier example for ReSpeaker 4-Mic Array on Raspberry Pi.

Notes:
- Requires doa_py (e.g. pip install git+https://github.com/zhiim/doa_py.git)
- Designed for 4-mic ReSpeaker (interleaved int16)
"""

import math
import queue
import threading
import warnings
import numpy as np
import pyaudio
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Import MUSIC implementation and array geometry from doa_py

from music_based import uca_rb_music
from arrays import UniformCircularArray


warnings.filterwarnings("ignore")

# ----------------------------
# Array / geometry parameters
# ----------------------------
SOUND_SPEED = 343.2                # m/s
MIC_DISTANCE_4 = 0.08127           # meters between opposite mics in 4-mic arrangement
MAX_TDOA_4 = MIC_DISTANCE_4 / float(SOUND_SPEED)

# ----------------------------
# Audio / model parameters
# ----------------------------
CAPTURE_SR = 16000                 # mic capture rate
CHANNELS = 4                       # 4-mic array
CHUNK = CAPTURE_SR // 4            # 0.25s per chunk (4000 samples)
SEGMENT_SEC = 0.5                  # run classification every 0.5s
SEGMENT_SAMPLES = int(CAPTURE_SR * SEGMENT_SEC)
MODEL_SR = 22050                   # MFCC / model sample rate
N_MFCC = 64                      # MFCCs for the classifier
CONF_THRESHOLD = 0.60              # confidence print threshold
TFLITE_PATH = "/home/pi/model.tflite"
TRAINING_LABELS = ["Environment", "Mavic", "Race", "UnknownDrone", "X5"]

# MUSIC search grids (azimuth and elevation)
AZIMUTH_GRID = np.linspace(0.0, 360.0, 361)       # 0..360 deg by 1 degree
ELEVATION_GRID = np.array([0.0])                  # horizontal plane

# ============================================================
# MicArray: handles audio capture and UCA geometry for MUSIC
# ============================================================
class MicArray:
    def __init__(self, rate=CAPTURE_SR, channels=CHANNELS, chunk_size=CHUNK):
        self.p = pyaudio.PyAudio()
        self.queue = queue.Queue()
        self.quit_event = threading.Event()
        self.channels = channels
        self.sample_rate = rate
        self.chunk_size = chunk_size

        # Find first input device with required input channels
        device_index = None
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            name = dev.get("name", "")
            max_in = int(dev.get("maxInputChannels", 0))
            max_out = int(dev.get("maxOutputChannels", 0))
            print(i, name, max_in, max_out)
            if max_in >= self.channels:
                print(f"Using device {i}: {name} (in-channels={max_in})")
                device_index = i
                break

        if device_index is None:
            raise RuntimeError(f"Cannot find input device with >= {self.channels} channels.")

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

        # Prepare array geometry for MUSIC (UCA)
        radius = MIC_DISTANCE_4 / 2.0
        self._array = UniformCircularArray(m=self.channels, r=radius)

    def _callback(self, in_data, frame_count, time_info, status):
        # push raw bytes to queue
        self.queue.put(in_data)
        return (None, pyaudio.paContinue)

    def start(self):
        # drain queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.stream.start_stream()

    def read_chunks(self):
        """
        Generator: yields numpy int16 arrays of shape (n_samples * channels,)
        (interleaved int16 flattened)
        """
        self.quit_event.clear()
        while not self.quit_event.is_set():
            frames = self.queue.get()
            if not frames:
                break
            arr = np.frombuffer(frames, dtype=np.int16)
            if arr.size == 0:
                continue
            yield arr

    def stop(self):
        self.quit_event.set()
        try:
            self.stream.stop_stream()
        except Exception:
            pass
        # put sentinel
        self.queue.put(b"")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        try:
            self.stream.close()
        except Exception:
            pass
        try:
            self.p.terminate()
        except Exception:
            pass
        if exc_type:
            return False

    # --- DOA estimation using UCA-RB-MUSIC ---
    def get_direction(self, buf_int16):
        """
        buf_int16: 1-D numpy int16 of interleaved samples (n_samples * channels)
        returns azimuth in degrees or None
        """
        try:
            frame_i16 = buf_int16.reshape(-1, self.channels)
        except ValueError:
            # partial/invalid buffer
            return None

        # transpose to shape (channels, n_samples)
        data = frame_i16.T.astype(np.float32)

        # STFT parameters for snapshots
        frame_len = 256
        hop = frame_len // 2
        nfft = 512

        num_samples = data.shape[1]
        if num_samples < frame_len:
            pad = frame_len - num_samples
            data = np.pad(data, ((0, 0), (0, pad)), mode="constant")
            num_samples = data.shape[1]

        # number of frames (snapshots)
        num_frames = 1 + (num_samples - frame_len) // hop
        if num_frames < 2:
            num_frames = 2
            pad_needed = (num_frames - 1) * hop + frame_len - num_samples
            if pad_needed > 0:
                data = np.pad(data, ((0, 0), (0, pad_needed)), mode="constant")
                num_samples = data.shape[1]

        window = np.hanning(frame_len)

        # collect per-channel frame spectra
        frames_list = []  # list of (num_frames, nfft_bins) per channel
        psd_accum = np.zeros(nfft // 2 + 1, dtype=np.float64)

        for ch in range(self.channels):
            ch_data = data[ch]
            frames_ch = []
            for i in range(num_frames):
                start = i * hop
                seg = ch_data[start : start + frame_len]
                if seg.shape[0] < frame_len:
                    seg = np.pad(seg, (0, frame_len - seg.shape[0]), mode="constant")
                seg = seg * window
                spec = np.fft.rfft(seg, n=nfft)  # complex
                frames_ch.append(spec)
            frames_ch = np.stack(frames_ch, axis=0)  # (num_frames, nfft_bins)
            frames_list.append(frames_ch)
            # magnitude-squared averaged across frames
            psd_accum += np.mean(np.abs(frames_ch) ** 2, axis=0)

        # pick dominant bin within a sensible band
        avg_psd = psd_accum / float(self.channels)
        min_freq_hz = 50.0
        max_freq_hz = min(8000.0, self.sample_rate / 2.0)
        freqs = np.fft.rfftfreq(nfft, d=1.0 / self.sample_rate)
        valid_bins = np.where((freqs >= min_freq_hz) & (freqs <= max_freq_hz))[0]
        if valid_bins.size == 0:
            peak_bin = int(np.argmax(avg_psd))
        else:
            peak_bin = int(valid_bins[np.argmax(avg_psd[valid_bins])])
        signal_freq_hz = float(freqs[peak_bin])

        # build received_data: (num_antennas, num_snapshots)
        num_snapshots = num_frames
        complex_bin_matrix = np.zeros((self.channels, num_snapshots), dtype=np.complex128)
        for ch in range(self.channels):
            frames_ch = frames_list[ch]  # (num_frames, nfft_bins)
            complex_bin_matrix[ch, :] = frames_ch[:, peak_bin]

        received_data = complex_bin_matrix

        num_signal = 1

        try:
            spectrum = uca_rb_music(
                received_data=received_data,
                num_signal=num_signal,
                array=self._array,
                signal_fre=signal_freq_hz,
                azimuth_grids=AZIMUTH_GRID,
                elevation_grids=ELEVATION_GRID,
                unit="deg",
            )
        except Exception as e:
            warnings.warn(f"uca_rb_music failed: {e}", UserWarning)
            return None

        # spectrum shape likely (az_bins, elev_bins)
        if spectrum is None:
            return None

        if spectrum.ndim == 2:
            az_spectrum = np.abs(spectrum[:, 0])
        else:
            az_spectrum = np.abs(spectrum)

        idx = int(np.argmax(az_spectrum))
        azimuth = float(AZIMUTH_GRID[idx])  # degrees

        return azimuth


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
    mono_float: 1-D float32 audio in [-1, 1] at capture_sr
    Returns: (label:str, confidence:float)
    """
    if capture_sr != MODEL_SR:
        y = librosa.resample(mono_float, orig_sr=capture_sr, target_sr=MODEL_SR)
        sr = MODEL_SR
    else:
        y = mono_float
        sr = capture_sr

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    feature_scaled = np.mean(mfcc.T, axis=0).astype(np.float32)

    # Ensure correct input shape expected by the TFLite model
    x = np.expand_dims(feature_scaled, axis=0).astype(np.float32)
    # If model expects a different shape (e.g. (1, N, 1)) user must adapt here

    interpreter.set_tensor(INPUT_DETAILS[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(OUTPUT_DETAILS[0]["index"])
    idx = int(np.argmax(output, axis=-1))
    conf = float(np.max(output))
    # Safe inverse_transform: if idx out of range, fallback to Unknown
    try:
        label = le.inverse_transform([idx])[0]
    except Exception:
        label = "Unknown"
    return label, conf


# ============================================================
# Main loop: read chunks, compute DOA every chunk,
# accumulate 0.5s for classification, print both.
# ============================================================
def main():
    mono_buffer = np.zeros(0, dtype=np.float32)
    samples_needed = SEGMENT_SAMPLES

    print("Starting 4-mic UCA-RB-MUSIC localization + classification. Press Ctrl+C to stop.\n")

    try:
        with MicArray(rate=CAPTURE_SR, channels=CHANNELS, chunk_size=CHUNK) as mic:
            for interleaved in mic.read_chunks():
                # interleaved is np.int16 array
                direction_deg = mic.get_direction(interleaved)
                if direction_deg is None:
                    print("DOA: n/a", end="\r")
                else:
                    # build mono float for classifier
                    try:
                        frame_i16 = interleaved.reshape(-1, CHANNELS)
                    except ValueError:
                        continue
                    mono_i16 = frame_i16.mean(axis=1)  # average channels
                    mono_f32 = (mono_i16 / 32768.0).astype(np.float32)

                    mono_buffer = np.concatenate([mono_buffer, mono_f32])
                    if mono_buffer.size >= samples_needed:
                        segment = mono_buffer[:samples_needed]
                        mono_buffer = mono_buffer[samples_needed:]
                        label, conf = classify_segment(segment, capture_sr=CAPTURE_SR)
                        if conf >= CONF_THRESHOLD:
                            print(f"dir={int(round(direction_deg))}°, class={label}, conf={conf:.2f}")
                        else:
                            print(f"dir={int(round(direction_deg))}°, class={label}, conf={conf:.2f} (low conf)")
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == '__main__':
    main()
