"""
voice_module.py
Milestone 3 - Voice Interaction Module (STT & TTS)
Uses built-in speech_to_text from streamlit_mic_recorder and browser Web Speech TTS.
"""

import streamlit as st
import streamlit.components.v1 as components

try:
    from streamlit_mic_recorder import speech_to_text
    HAS_SPEECH_TO_TEXT = True
except ImportError:
    HAS_SPEECH_TO_TEXT = False


class VoiceInteractionModule:
    @staticmethod
    def render_speech_to_text_widget() -> str:
        """
        Renders microphone recording button and returns the transcribed text directly.
        """
        st.markdown("##### 🎙️ Voice Input")

        if not HAS_SPEECH_TO_TEXT:
            st.warning("Please install: `pip install streamlit-mic-recorder`")
            return ""

        col1, col2 = st.columns([2, 5])
        with col1:
            transcribed_text = speech_to_text(
                start_prompt="🔴 Start Speaking",
                stop_prompt="⏹️ Stop & Transcribe",
                language="en",
                use_container_width=True,
                key="voice_stt_component"
            )

        if transcribed_text:
            with col2:
                st.success(f"Transcribed: *\"{transcribed_text}\"*")
            return transcribed_text

        return ""

    @staticmethod
    def render_text_to_speech_controls(text_content: str) -> None:
        """
        Renders native playback controls for synthesized voice output.
        """
        cleaned_text = (
            text_content.replace('"', '\\"')
            .replace("'", "\\'")
            .replace("\n", " ")
            .strip()
        )

        tts_html = f"""
        <div style="margin-top: 8px; padding: 6px 10px; background: #f1f3f5; border-radius: 6px; display: inline-flex; align-items: center; gap: 8px; font-family: sans-serif;">
            <span style="font-size: 0.85em; font-weight: bold; color: #495057;">🔊 Read Response:</span>
            <button onclick="readAloud()" style="font-size: 0.8em; padding: 3px 8px; border: 1px solid #ced4da; border-radius: 4px; background: #fff; cursor: pointer;">▶️ Play</button>
            <button onclick="pauseSpeech()" style="font-size: 0.8em; padding: 3px 8px; border: 1px solid #ced4da; border-radius: 4px; background: #fff; cursor: pointer;">⏸️ Pause</button>
            <button onclick="resumeSpeech()" style="font-size: 0.8em; padding: 3px 8px; border: 1px solid #ced4da; border-radius: 4px; background: #fff; cursor: pointer;">⏯️ Resume</button>
            <button onclick="stopSpeech()" style="font-size: 0.8em; padding: 3px 8px; border: 1px solid #ced4da; border-radius: 4px; background: #fff; cursor: pointer;">⏹️ Stop</button>
        </div>

        <script>
            let synth = window.speechSynthesis;
            let utterText = "{cleaned_text}";

            function readAloud() {{
                if (!synth) return;
                synth.cancel();
                let utterance = new SpeechSynthesisUtterance(utterText);
                utterance.rate = 1.0;
                synth.speak(utterance);
            }}

            function pauseSpeech() {{
                if (synth && synth.speaking) synth.pause();
            }}

            function resumeSpeech() {{
                if (synth && synth.paused) synth.resume();
            }}

            function stopSpeech() {{
                if (synth) synth.cancel();
            }}
        </script>
        """
        components.html(tts_html, height=50)