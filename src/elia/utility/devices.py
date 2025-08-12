from pvrecorder import PvRecorder

devices = PvRecorder.get_available_devices()
for i, d in enumerate(devices):
    print(f"{i}: {d}")