import asyncio
import io
import json
import logging
import os
import re
import sys
import threading
import wave

import numpy as np
import openai
import pyaudio
import pygame
import pyttsx3
import sounddevice as sd
import webrtcvad  # Added for accurate voice activity detection
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient


# Ensure sounddevice is initialized with native audio API
def select_native_audio():
    """
    Choose the host API index for ALSA (Linux) and log native APIs on Windows/macOS.
    """
    apis = sd.query_hostapis()
    target = None

    for idx, api in enumerate(apis):
        name = api.get("name", "").lower()

        if sys.platform.startswith("linux") and "alsa" in name:
            target = idx
            print(f"🖥️ Using ALSA (API {idx}: {api['name']})")
            return

        if sys.platform == "win32" and "wasapi" in name:
            target = idx
            print(
                f"🖥️ Detected WASAPI (API {idx}: {api['name']}) – no assignment needed"
            )
            return

        if sys.platform == "darwin" and "core audio" in name:
            target = idx
            print(
                f"🖥️ Detected Core Audio (API {idx}: {api['name']}) – no assignment needed"
            )
            return

    # Fallback
    default_api = apis[sd.default.hostapi]["name"]
    print(
        f"⚠️ Native audio API not found; using default ({sd.default.hostapi}: {default_api})"
    )


# Call right away
select_native_audio()


def credential_path(rel_path: str) -> str:
    """
    Return the absolute path to rel_path, preferring:
     1) An external file next to the EXE (when frozen) or script.
     2) A bundled file inside _MEIPASS (if you ever embed defaults).
    """
    if getattr(sys, "frozen", False):
        # Running as frozen exe
        base_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        base_dir = os.path.dirname(os.path.abspath(__file__))
    external = os.path.join(base_dir, rel_path)
    if os.path.exists(external):
        return external
    # Fallback into bundle (if embedded)
    bundle_dir = getattr(sys, "_MEIPASS", None)
    if bundle_dir:
        bundled = os.path.join(bundle_dir, rel_path)
        if os.path.exists(bundled):
            return bundled
    raise FileNotFoundError(f"Cannot find {rel_path} in {base_dir}")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

TTS_ENGINE = pyttsx3.init()


class VoiceAssistant:
    """Voice assistant with wake-word gating and VAD-based silence detection."""

    def __init__(
        self,
        openai_api_key: str,
        model: str = "o4-mini",
        vad_aggressiveness: int = 2,
        silence_threshold: int = 500,
        silence_duration: float = 0.5,
        mcp_config: dict | None = None,
        notes_dir: str | None = None,
        system_prompt: str | None = None,
    ):
        # Audio parameters
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = int(self.rate * 30 / 1000)  # 30ms per frame
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        # flag to let “stop” commands interrupt speaking (see process_command)
        self._stop_speaking = threading.Event()
        # Wake word
        self.wake_word = "assistant"

        # VAD selection
        if webrtcvad:
            self.vad = webrtcvad.Vad(vad_aggressiveness)
            self.use_webrtcvad = True
            logger.info("Using webrtcvad for voice activity detection")
        else:
            self.vad = None
            self.use_webrtcvad = False
            logger.info("Using amplitude threshold for silence detection")

        # Initialize audio I/O
        self.audio = pyaudio.PyAudio()
        pygame.mixer.init()

        # OpenAI client (Whisper & GPT)
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.model = model

        # MCP agent config
        self.mcp_config = mcp_config
        self.agent = None
        self.system_prompt = system_prompt or (
            "You are a helpful voice assistant that helps user to manage their gmail. Be concise in your responses. Do not add unrelevant information. If you are asked to read an email, fetch 10 latest emails, read the subject and the short summary of the body of the email, then mark it as read."
        )

    def _substitute_env_vars(self, config: dict) -> dict:
        """Recursively substitute environment variable placeholders in config."""
        if isinstance(config, dict):
            result = {}
            for key, value in config.items():
                result[key] = self._substitute_env_vars(value)
            return result
        elif isinstance(config, str):
            # Handle environment variable substitution
            if config.startswith("${") and config.endswith("}"):
                env_var = config[2:-1]  # Remove ${ and }
                return os.getenv(
                    env_var, config
                )  # Return original if env var not found
            return config
        else:
            return config

    async def initialize_mcp(self):
        """Initialize MCP client and agent with proper error handling."""
        # Use provided config or load from file
        if self.mcp_config:
            config = self.mcp_config
        else:
            # New: look for mcp_servers.json next to the EXE/script
            config_path = credential_path("mcp_servers.json")
            logger.debug(f"Loading MCP config from {config_path}")
            with open(config_path, "r") as f:
                config = json.load(f)
            # Replace environment variable placeholders
            config = self._substitute_env_vars(config)

        try:
            # Create MCP client
            self.mcp_client = MCPClient.from_dict(config)

            # Create LLM
            llm = ChatOpenAI(model=self.model)

            # Create agent with memory
            self.agent = MCPAgent(
                llm=llm,
                client=self.mcp_client,
                max_steps=10,
                memory_enabled=True,
                system_prompt=self.system_prompt,
            )
            await self.agent.initialize()

            logger.info("MCP servers initialized successfully")
            return True

        except Exception as e:
            logger.error("Error initializing MCP", exc_info=e)
            return False

    def detect_silence(self, audio_data: bytes) -> bool:
        """Fallback amplitude-based silence detection."""
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        return np.max(np.abs(audio_array)) < self.silence_threshold

    def record_audio(self) -> bytes | None:
        """Record audio. Skip leading silence, then record until end-of-speech."""
        logger.info("Listening for speech...")
        try:
            stream = self.audio.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )
            frames = []
            recording = False
            silence_count = 0

            # Determine number of silence frames
            if self.use_webrtcvad:
                max_silence_frames = int(self.silence_duration * 1000 / 30)
            else:
                max_silence_frames = int(self.rate / self.chunk * self.silence_duration)

            while True:
                data = stream.read(self.chunk, exception_on_overflow=False)
                # Determine if current chunk has speech
                if self.use_webrtcvad:
                    is_speech = self.vad.is_speech(data, sample_rate=self.rate)
                else:
                    is_speech = not self.detect_silence(data)

                if not recording:
                    # Wait for initial speech
                    if is_speech:
                        recording = True
                        frames.append(data)
                        logger.debug("Speech detected, starting recording")
                    else:
                        # Skip leading silence
                        continue
                else:
                    frames.append(data)
                    if is_speech:
                        silence_count = 0
                    else:
                        silence_count += 1
                        if silence_count > max_silence_frames:
                            logger.debug("End of speech detected, stopping recording")
                            break

                # Safety cap: max 30 seconds
                if len(frames) > int(self.rate / self.chunk * 30):
                    logger.debug("Max recording duration reached, stopping")
                    break

            stream.stop_stream()
            stream.close()

            if not frames:
                logger.warning("No speech captured.")
                return None

            logger.debug("Audio captured: %d frames", len(frames))
            return b"".join(frames)
        except Exception as e:
            logger.error("Error recording audio", exc_info=e)
            return None

    def audio_to_text(self, audio_data: bytes) -> str | None:
        """Convert audio bytes to text via Whisper."""
        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(audio_data)
            wav_buffer.seek(0)
            wav_buffer.name = "audio.wav"
            resp = self.openai_client.audio.transcriptions.create(
                model="whisper-1", file=wav_buffer, language="en"
            )
            return resp.text.strip()
        except Exception as e:
            logger.error("Error transcribing audio", exc_info=e)
            return None

    async def text_to_speech(self, text: str) -> bool:
        """Speak text via OpenAI TTS (tts-1, alloy voice), falling back to pyttsx3 if that fails."""
        try:
            # Call OpenAI TTS endpoint synchronously
            resp = self.openai_client.audio.speech.create(
                model="tts-1", voice="shimmer", input=text
            )
            # Extract raw MP3 bytes
            audio_bytes = resp.content

            # Play via pygame
            buf = io.BytesIO(audio_bytes)
            buf.name = "speech.mp3"
            buf.seek(0)
            pygame.mixer.music.load(buf)
            pygame.mixer.music.play()
            # Wait until playback finishes
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            return True

        except Exception as e:
            logger.warning(f"OpenAI TTS failed ({e}), falling back to pyttsx3.")
            TTS_ENGINE.say(text)
            TTS_ENGINE.runAndWait()
            return True

    async def process_command(self, text: str) -> str:
        """Handle built-in commands or forward to MCP agent."""
        logger.info(f"User said: {text}")
        # normalize once: strip punctuation + lowercase
        normalized = re.sub(r"[^\w\s]", "", text).strip().lower()

        # built-in exit
        if normalized in {"exit", "quit", "goodbye"}:
            return "Goodbye!"

        # clear history
        if normalized == "clear":
            if self.agent:
                self.agent.clear_conversation_history()
            return "History cleared."

        # stub for stopping mid-speech (needs wiring in text_to_speech)
        if normalized == "stop":
            self._stop_speaking.set()
            return ""  # no spoken response

        if not self.agent:
            return "Assistant not initialized."

        try:
            result = await self.agent.run(text)
            # never return None
            if not isinstance(result, str) or not result:
                return "Sorry, I didn't catch that."
            return result
        except Exception:
            logger.exception("Error in MCP agent")
            return "Error processing command."

    async def run(self) -> None:
        """Main loop: record until silence, detect wake word in transcript, then process command."""
        logger.info("Starting assistant...")
        # Initialize MCP
        if not await self.initialize_mcp():
            logger.error("MCP init failed, exiting.")
            return
        try:
            while True:
                # 1) Record audio (wake word + command)
                audio = self.record_audio()
                if not audio:
                    continue

                # 2) Transcribe entire chunk
                text = self.audio_to_text(audio)
                if not text:
                    continue
                logger.info(f"Heard: {text}")

                # 3) Check for wake word
                lower = text.lower()
                if self.wake_word not in lower:
                    logger.debug("Wake word not detected, continuing...")
                    continue

                # 4) Extract command after wake word
                cmd = lower.split(self.wake_word, 1)[1].strip()
                if not cmd:
                    logger.debug("No command after wake word, continuing...")
                    continue
                logger.info(f"Command detected: {cmd}")

                # 5) Process and speak
                response = await self.process_command(cmd)
                logger.info(f"Assistant: {response!r}")
                if response:
                    await self.text_to_speech(response)

                # 6) Exit on 'goodbye'
                if response.strip().lower().startswith("goodbye"):
                    break
        finally:
            # Cleanup
            self.audio.terminate()
            pygame.mixer.quit()
            if hasattr(self, "mcp_client") and self.mcp_client:
                await self.mcp_client.close_all_sessions()


async def main():
    """CLI entrypoint: parse args and launch assistant."""
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Voice-enabled AI assistant")
    parser.add_argument(
        "--openai-api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4.1"),
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--vad-aggressiveness",
        type=int,
        default=int(os.getenv("VAD_AGGRESSIVENESS", "3")),
        help="webrtcvad aggressiveness (0-3)",
    )
    parser.add_argument(
        "--silence-threshold",
        type=int,
        default=int(os.getenv("VOICE_SILENCE_THRESHOLD", "500")),
        help="Amplitude threshold for silence",
    )
    parser.add_argument(
        "--silence-duration",
        type=float,
        default=float(os.getenv("VOICE_SILENCE_DURATION", "0.5")),
        help="Silence duration in seconds",
    )
    parser.add_argument(
        "--system-prompt",
        default=os.getenv("ASSISTANT_SYSTEM_PROMPT"),
        help="Custom system prompt",
    )

    args = parser.parse_args()

    if not args.openai_api_key:
        logger.error("OpenAI API key is required")
        logger.error("Set OPENAI_API_KEY environment variable or pass --openai-api-key")
        sys.exit(1)

    # Load MCP servers config
    with open(credential_path("mcp_servers.json"), "r") as f:
        mcp_config = json.load(f)

    assistant = VoiceAssistant(
        openai_api_key=args.openai_api_key,
        model=args.model,
        vad_aggressiveness=args.vad_aggressiveness,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        system_prompt=args.system_prompt,
        mcp_config=mcp_config,
    )
    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main())
