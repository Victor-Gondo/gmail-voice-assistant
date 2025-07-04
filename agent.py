import asyncio
import io
import json
import os
import sys
import tempfile
import wave

import numpy as np
import openai
import pyaudio
import pygame
import pyttsx3
import webrtcvad  # Added for accurate voice activity detection
from elevenlabs import play
from elevenlabs.client import ElevenLabs
from elevenlabs.types.voice_settings import VoiceSettings
from langchain_openai import ChatOpenAI
from mcp_use import MCPAgent, MCPClient

# Logging setup (added previously)
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

TTS_ENGINE = pyttsx3.init()

class VoiceAssistant:
    """Voice assistant with wake-word gating and VAD-based silence detection."""

    def __init__(
        self,
        openai_api_key: str,
        elevenlabs_api_key: str | None = None,
        model: str = "o4-mini",
        elevenlabs_voice_id: str = "ZF6FPAbjXT4488VcRRnw",
        vad_aggressiveness: int = 2,
        silence_threshold: int = 500,
        silence_duration: float = 1.5,
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

        # ElevenLabs TTS
        self.elevenlabs_client = ElevenLabs(api_key=elevenlabs_api_key) if elevenlabs_api_key else None
        self.elevenlabs_voice_id = elevenlabs_voice_id

        # MCP agent config
        self.mcp_config = mcp_config
        self.agent = None
        self.system_prompt = system_prompt or (
            "You are a helpful voice assistant with access to various tools. Be concise in your responses."
        )

        # Notes directory
        self.notes_dir = notes_dir or os.path.join(tempfile.gettempdir(), "voice_assistant_notes")
        os.makedirs(self.notes_dir, exist_ok=True)

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
                return os.getenv(env_var, config)  # Return original if env var not found
            return config
        else:
            return config

    async def initialize_mcp(self):
        """Initialize MCP client and agent with proper error handling."""
        # Use provided config or load from file
        if self.mcp_config:
            config = self.mcp_config
        else:
            # Try to load from mcp_servers.json
            config_file = os.path.join(os.path.dirname(__file__), "mcp_servers.json")
            if os.path.exists(config_file):
                logger.debug("Found mcp_servers.json file")
                with open(config_file) as f:
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

    async def wait_for_wake(self) -> None:
        """Listen continuously until wake word is heard."""
        logger.info(f"Waiting for wake word: '{self.wake_word}'")
        while True:
            audio = self.record_audio()
            if not audio:
                continue
            text = self.audio_to_text(audio)
            if not text:
                continue
            logger.info(f"Heard: {text}")
            if self.wake_word in text.lower():
                logger.info("Wake word detected!")
                return
            logger.debug("Wake word not found, retrying...")

    def audio_to_text(self, audio_data: bytes) -> str | None:
        """Convert audio bytes to text via Whisper."""
        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.audio_format))
                wf.setframerate(self.rate)
                wf.writeframes(audio_data)
            wav_buffer.seek(0)
            wav_buffer.name = 'audio.wav'
            resp = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_buffer,
                language="en"
            )
            return resp.text.strip()
        except Exception as e:
            logger.error("Error transcribing audio", exc_info=e)
            return None

    async def text_to_speech(self, text: str) -> bool:
        """Speak text via ElevenLabs or fallback TTS_ENGINE."""
        if self.elevenlabs_client:
            try:
                audio = self.elevenlabs_client.text_to_speech.convert(
                    text=text,
                    voice_id=self.elevenlabs_voice_id,
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                    optimize_streaming_latency="2",
                    voice_settings=VoiceSettings(speed=1.1),
                )
                play(audio)
                return True
            except Exception:
                logger.warning("ElevenLabs TTS failed, using fallback.")
        TTS_ENGINE.say(text)
        TTS_ENGINE.runAndWait()
        return True

    async def process_command(self, text: str) -> str:
        """Handle built-in commands or forward to MCP agent."""
        logger.info(f"User said: {text}")
        if text.lower() in ["exit", "quit", "goodbye"]:
            return "Goodbye!"
        if text.lower() == "clear":
            if self.agent:
                self.agent.clear_conversation_history()
            return "History cleared."
        if not self.agent:
            return "Assistant not initialized."
        try:
            return await self.agent.run(text)
        except Exception as e:
            logger.error("Error in MCP agent", exc_info=e)
            return "Error processing command."

    async def run(self) -> None:
        """Main loop: wake-word → record → transcribe → respond → speak."""
        logger.info("Starting assistant...")
        if not await self.initialize_mcp():
            logger.error("MCP init failed, exiting.")
            return
        try:
            while True:
                await self.wait_for_wake()
                logger.info("Ready for command.")
                audio = self.record_audio()
                if not audio:
                    continue
                text = self.audio_to_text(audio)
                if not text:
                    continue
                response = await self.process_command(text)
                logger.info(f"Assistant: {response}")
                await self.text_to_speech(response)
                if text.lower() in ["exit", "quit", "goodbye"]:
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.audio.terminate()
            pygame.mixer.quit()
            if hasattr(self, 'mcp_client') and self.mcp_client:
                await self.mcp_client.close_all_sessions()


async def main():
    """CLI entrypoint: parse args and launch assistant."""
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Voice-enabled AI assistant")
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument("--elevenlabs-api-key", default=os.getenv("ELEVENLABS_API_KEY"), help="ElevenLabs API key")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4"), help="OpenAI model to use")
    parser.add_argument("--voice-id", default=os.getenv("ELEVENLABS_VOICE_ID", "ZF6FPAbjXT4488VcRRnw"), help="ElevenLabs voice ID")
    parser.add_argument("--vad-aggressiveness", type=int, default=2, help="webrtcvad aggressiveness (0-3)")
    parser.add_argument("--silence-threshold", type=int, default=int(os.getenv("VOICE_SILENCE_THRESHOLD", "500")), help="Amplitude threshold for silence")
    parser.add_argument("--silence-duration", type=float, default=float(os.getenv("VOICE_SILENCE_DURATION", "1.5")), help="Silence duration in seconds")
    parser.add_argument("--system-prompt", default=os.getenv("ASSISTANT_SYSTEM_PROMPT"), help="Custom system prompt")

    args = parser.parse_args()

    if not args.openai_api_key:
        logger.error("OpenAI API key is required")
        logger.error("Set OPENAI_API_KEY environment variable or pass --openai-api-key")
        sys.exit(1)

    # Load MCP servers config
    with open("mcp_servers.json", "r") as f:
        mcp_config = json.load(f)

    assistant = VoiceAssistant(
        openai_api_key=args.openai_api_key,
        elevenlabs_api_key=args.elevenlabs_api_key,
        model=args.model,
        elevenlabs_voice_id=args.voice_id,
        vad_aggressiveness=args.vad_aggressiveness,
        silence_threshold=args.silence_threshold,
        silence_duration=args.silence_duration,
        system_prompt=args.system_prompt,
        mcp_config=mcp_config,
    )
    await assistant.run()


if __name__ == "__main__":
    asyncio.run(main())
