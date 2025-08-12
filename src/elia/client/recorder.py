import io, wave
import sounddevice as sd
import webrtcvad

def record_until_silence(samplerate=16000, frame_ms=30, max_silence_ms=1500):
    """Registra finché c'è voce e si ferma dopo un certo silenzio."""
    
    vad = webrtcvad.Vad(2)  # livello di aggressività del VAD (0-3)
    frame_bytes = int(samplerate * frame_ms / 1000) * 2  # byte per frame (mono int16)
    silence_frames_needed = int(max_silence_ms / frame_ms)

    collected = []     # blocchi audio con voce
    silent_count = 0   # frame consecutivi di silenzio

    print("🎙️ Parla pure… mi fermo quando c'è silenzio")
    with sd.RawInputStream(samplerate=samplerate, channels=1, dtype="int16") as stream:
        while True:
            data, _ = stream.read(frame_bytes // 2)
            b = bytes(data)
            if vad.is_speech(b, samplerate):
                collected.append(b)
                silent_count = 0
            else:
                silent_count += 1
                if silent_count >= silence_frames_needed:
                    break

    # converte in WAV in memoria
    pcm = b"".join(collected)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(pcm)
    return bio.getvalue()
