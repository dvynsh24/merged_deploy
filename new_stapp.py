import os
import streamlit as st
import av
import time
import threading
from streamlit_webrtc import (
    VideoHTMLAttributes,
    webrtc_streamer,
    WebRtcMode,
)
from dotenv import load_dotenv
from twilio.rest import Client
from merged_audio_handle import AudioFrameHandler
from merged_video_handle import VideoFrameHandler

load_dotenv()

alarm_file_path = os.path.join("assets", "audio_files", "wake_up.wav")

st.set_page_config(
    page_title="Doze Off",
    page_icon="💤",
)

st.title("Merged DozeOff and HeadPosn")

WAIT_DOZEOFF_TIME = 2
WAIT_HEADPOSN_TIME = 5

EAR_THRESH = 0.18
LEFT_THRESH = 10
RIGHT_THRESH = 10
DOWN_THRESH = 10
UP_THRESH = 10


thresholds = {
    "EAR_THRESH": EAR_THRESH,
    "LEFT_THRESH": LEFT_THRESH,
    "RIGHT_THRESH": RIGHT_THRESH,
    "DOWN_THRESH": DOWN_THRESH,
    "UP_THRESH": UP_THRESH,
    "WAIT_DOZEOFF_TIME": WAIT_DOZEOFF_TIME,
    "WAIT_HEADPOSN_TIME": WAIT_HEADPOSN_TIME,
}

video_handler = VideoFrameHandler()
audio_handler = AudioFrameHandler(sound_file_path=alarm_file_path)

lock = (
    threading.Lock()
)  # For thread-safe access & to prevent race-condition.

shared_state = {"play_alarm": False}


def video_frame_callback(frame: av.VideoFrame):
    
    frame = frame.to_ndarray(format="bgr24")

    frame, play_alarm = video_handler.process(frame, thresholds)
    with lock:
        shared_state["play_alarm"] = play_alarm
    

    return av.VideoFrame.from_ndarray(frame, format="bgr24")


def audio_frame_callback(frame: av.AudioFrame):
    with lock:
        play_alarm = shared_state["play_alarm"]

    new_frame: av.AudioFrame = audio_handler.process(
        frame, play_sound=play_alarm
    )
    return new_frame


def get_ice_servers():
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers

ctx = webrtc_streamer(
    key="drowsiness-detection",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    audio_frame_callback=audio_frame_callback,
    rtc_configuration={"iceServers": get_ice_servers()},
    media_stream_constraints={
        "video": {"height": {"ideal": 480}},
        "audio": True,
    },
    video_html_attrs=VideoHTMLAttributes(
        autoPlay=True, controls=False, muted=False
    ),
    async_processing=True,
)

# print("started session at", start_session_time)
# print("ended session at", end_session_time)
# print("time spent", end_session_time - start_session_time)