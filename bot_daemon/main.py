import numpy as np
import sounddevice as sd
import speech_recognition as sr
import os
import io
import wave
from piper.voice import PiperVoice

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

class AI:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o")
        self.workflow = StateGraph(state_schema=MessagesState)

        def call_model(state: MessagesState):
            system_prompt = ()
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = self.model.invoke(messages)
            return {"messages": response}
        
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", call_model)
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
        self.config = {"configurable": {"thread_id": "1"}}


    def invoke(self, text: str):
        response = self.app.invoke({"messages": [HumanMessage(content=text)]}, config=self.config)
        return response["messages"][-1].content


class TTS:
    def __init__(self, model_path):    
        self.voice = PiperVoice.load(model_path)
        self.stream = sd.OutputStream(samplerate=self.voice.config.sample_rate, channels=1, dtype='int16')
        self.stream.start()

    def say(self, text):
        for audio_bytes in self.voice.synthesize_stream_raw(text):
            self.stream.write(np.frombuffer(audio_bytes, dtype=np.int16))

    def close(self):
        self.stream.stop()
        self.stream.close()

def find_microphone(mic_name):
    mic_list = sr.Microphone.list_microphone_names()
    print(mic_list)
    for i, name in enumerate(mic_list):
        if mic_name in name:
            return i

    return None

def record_audio(duration=10, samplerate=44100, channels=1, device=None):
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='int16', device=device)
    sd.wait()

    raw_audio_bytes = audio_data.tobytes()
    return sr.AudioData(raw_audio_bytes, samplerate, 2)

def main():
    tts_agent = TTS("en_GB-semaine-medium.onnx")
    ai_agent = AI()

    mic_idx = find_microphone("UAC 1.0 Microphone & HID-Mediak")
    print(mic_idx)
    if mic_idx is None:
        tts_agent.say("Microphone Not Found... Exitting.")
        return

    r = sr.Recognizer()

    tts_agent.say("Started!")
    print("While AI is speaking, you will not be able to respond.")
    while True:            
        buffer = ""
        collected_message = []
        message_started = False
        transcribed_message = ""
        while True:
            audio = record_audio(device=mic_idx)
            try:
            	text = r.recognize_google(audio)
            except:
                tts_agent.say("Sorry, I couldn't understand that. Please try again.")
                continue

            print("Transcription: " + text)

            buffer += " " + text

            if not message_started:
                trigger_message = "start message"
                start_index = buffer.lower().find(trigger_message)

                if start_index != -1:
                    print("Trigger detected. Starting capture.")
                    message_started = True
                    buffer = buffer[start_index + len(trigger_message):]
            if message_started:
                end_message = "stop message"
                end_index = buffer.lower().find(end_message)

                if end_index != -1:
                    collected_message.append(buffer[:end_index].strip())
                    print("End detected. Capture complete.")
                    transcribed_message = " ".join(collected_message).strip()
                    break
                else:
                    collected_message.append(buffer.strip())
                    buffer = ""


        print("You: " + transcribed_message)
        ai_msg = ai_agent.invoke(transcribed_message)
        print("AI: " + ai_msg)
        tts_agent.say(ai_msg)

    tts_agent.close()

if __name__ == "__main__":
    main()
