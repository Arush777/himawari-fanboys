"""Streamlit demo UI for the Track 2 video captioning agent.

Run locally:   streamlit run app.py           (reads .env via config.py)
On Streamlit Community Cloud: set the same variables in the app's Secrets store.
"""
import os

import streamlit as st

# Copy Streamlit Cloud secrets into the environment BEFORE importing config, so the
# same config/pipeline code works locally (.env) and on Streamlit Cloud (st.secrets).
try:
    for _key, _value in st.secrets.items():
        os.environ.setdefault(_key, str(_value))
except Exception:
    pass  # no secrets file when running locally — .env covers it

import config
from llm_client import RitsClient
from pipeline import STYLE_GUIDE, captions_from_description, describe_video

EXAMPLE_CLIPS = {
    "Urban autumn boulevard (city traffic)": "https://storage.googleapis.com/amd-hackathon-clips/1860079-uhd_2560_1440_25fps.mp4",
    "Orange kitten in a garden": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
    "Office worker at a computer": "https://storage.googleapis.com/amd-hackathon-clips/3044693-uhd_3840_2160_24fps.mp4",
}

STYLE_LABELS = {
    "formal": "🎩 Formal",
    "sarcastic": "🙄 Sarcastic",
    "humorous_tech": "🤓 Humorous (tech)",
    "humorous_non_tech": "😂 Humorous (non-tech)",
}


@st.cache_resource
def get_client() -> RitsClient:
    return RitsClient(config.VISION_API_KEY, config.VISION_API_ENDPOINT, config.VISION_MODEL_ID)


st.set_page_config(page_title="Video Captioning Agent", page_icon="🎬", layout="centered")
st.title("🎬 Video Captioning Agent")
st.caption(
    "AMD Developer Hackathon — Track 2. Paste any video URL (or pick an example clip), "
    "choose the caption styles, and the agent watches the clip and writes a caption in each tone."
)

source = st.radio("Video source", ["Example clip", "Custom URL"], horizontal=True)
if source == "Example clip":
    choice = st.selectbox("Clip", list(EXAMPLE_CLIPS))
    video_url = EXAMPLE_CLIPS[choice]
else:
    video_url = st.text_input("Direct video URL (mp4)", placeholder="https://...")

styles = st.multiselect(
    "Caption styles",
    options=list(STYLE_GUIDE),
    default=list(STYLE_GUIDE),
    format_func=lambda s: STYLE_LABELS.get(s, s),
)

if video_url:
    st.video(video_url)

if st.button("Generate captions", type="primary", disabled=not (video_url and styles)):
    client = get_client()
    try:
        with st.spinner("Watching the video (downloading + sampling frames + vision model)..."):
            description = describe_video(video_url, client)
        with st.expander("What the agent saw (factual description)"):
            st.write(description)
        with st.spinner("Writing styled captions..."):
            captions = captions_from_description(description, styles, client)
    except Exception as e:
        st.error(f"Captioning failed: {e}")
    else:
        for style in styles:
            st.subheader(STYLE_LABELS.get(style, style))
            st.markdown(f"> {captions.get(style, '') or '_no caption returned_'}")
