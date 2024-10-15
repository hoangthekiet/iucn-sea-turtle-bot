
import streamlit as st
from server.constants.view import ICON_BOT, ICON_USER, ICON_ERROR, ABOUT, DISCLAIMER

st.set_page_config(page_icon="ℹ", layout="wide", page_title="About 🍵")
st.markdown(ABOUT)
st.warning(DISCLAIMER, icon="⚠️")