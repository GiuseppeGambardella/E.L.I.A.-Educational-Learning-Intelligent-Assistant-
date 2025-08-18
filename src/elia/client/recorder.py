import io, wave
import sounddevice as sd
import webrtcvad

def record_until_silence(samplerate=16000, frame_ms=20, max_silence_ms=1000, vad_aggressiveness=2):
    """
    Registra dal microfono finché rileva voce e si ferma dopo max_silence_ms di silenzio.
    
    Ritorna:
        wav_bytes (bytes): audio WAV in memoria
        duration (float): durata in secondi
    """
    vad = webrtcvad.Vad(vad_aggressiveness)

    frame_samples = int(samplerate * frame_ms / 1000)
    silence_frames_needed = int(max_silence_ms / frame_ms)

    collected = bytearray()
    silent_count = 0

    with sd.RawInputStream(
        samplerate=samplerate,
        channels=1,
        dtype="int16",
        blocksize=frame_samples
    ) as stream:
        while True:
            data, _ = stream.read(frame_samples)
            b = bytes(data)
            if vad.is_speech(b, samplerate):
                collected.extend(b)
                silent_count = 0
            else:
                silent_count += 1
                if silent_count >= silence_frames_needed:
                    break

    pcm = bytes(collected)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm)

    duration = len(pcm) / (2 * samplerate)
    return bio.getvalue(), duration
