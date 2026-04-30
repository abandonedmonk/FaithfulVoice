from pathlib import Path


class KokoroTTS:
    def __init__(self, model_name: str = "kokoro-82m"):
        self.model_name = model_name
        self._model = None
        self._voicepack = None

    def _load(self):
        if self._model is None:
            from kokoro import Kokoro
            self._model = Kokoro(self.model_name)

    def synthesize(self, text: str, output_path: str | Path, voice: str = "af_heart") -> Path:
        self._load()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.synthesize(text, output_path, voice=voice)
        return output_path

    def synthesize_bytes(self, text: str, voice: str = "af_heart") -> bytes:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = self.synthesize(text, tmp.name, voice=voice)
            return Path(path).read_bytes()
