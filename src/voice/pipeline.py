from dataclasses import dataclass
from pathlib import Path

from src.voice.stt import MoonshineSTT
from src.voice.tts import KokoroTTS


@dataclass
class VoiceResult:
    transcript: str
    answer_text: str
    audio_path: str
    stt_ms: float = 0.0
    tts_ms: float = 0.0
    total_ms: float = 0.0


class VoicePipeline:
    def __init__(
        self,
        stt_model: str = "moonshine/base",
        tts_model: str = "kokoro-82m",
    ):
        self.stt = MoonshineSTT(model_name=stt_model)
        self.tts = KokoroTTS(model_name=tts_model)
        self._rag_fn = None

    def set_rag_fn(self, fn):
        self._rag_fn = fn

    def run(self, audio_path: str | Path, output_dir: str = "output/voice") -> VoiceResult:
        import time

        t0 = time.perf_counter()
        transcript = self.stt.transcribe(audio_path)
        stt_ms = (time.perf_counter() - t0) * 1000

        if self._rag_fn is None:
            raise RuntimeError("Call set_rag_fn() before run()")
        answer_text = self._rag_fn(transcript)

        t1 = time.perf_counter()
        output_path = Path(output_dir) / "response.wav"
        self.tts.synthesize(answer_text, output_path)
        tts_ms = (time.perf_counter() - t1) * 1000

        total_ms = (time.perf_counter() - t0) * 1000
        return VoiceResult(
            transcript=transcript,
            answer_text=answer_text,
            audio_path=str(output_path),
            stt_ms=stt_ms,
            tts_ms=tts_ms,
            total_ms=total_ms,
        )
