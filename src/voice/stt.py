from pathlib import Path


class MoonshineSTT:
    def __init__(self, model_name: str = "moonshine/base"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from moonshine import MoonshineASR
            self._model = MoonshineASR(model_name=self.model_name)

    def transcribe(self, audio_path: str | Path) -> str:
        self._load()
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return self._model.transcribe(str(audio_path))

    def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        import tempfile
        import soundfile as sf
        import numpy as np

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate)
            return self.transcribe(tmp.name)
