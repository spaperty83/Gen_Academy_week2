"""Run with: uv run streamlit run app.py"""
import streamlit as st

from park_rag.config import Settings
from park_rag.qa import NOTICE, ParkAssistant, load_active


def set_session_ended(ended):
    st.session_state.clear()
    st.session_state.session_ended = ended


st.set_page_config(page_title="Yellowstone NPS Assistant", page_icon="??")
if "session_ended" not in st.session_state:
    st.session_state.session_ended = False

st.title("Yellowstone NPS Assistant")
if st.session_state.session_ended:
    st.success("Your session has ended.")
    st.write("You can close this browser tab or start a new session.")
    st.button("Start again", key="start_again", on_click=set_session_ended, args=(False,))
    st.stop()

st.sidebar.button(
    "Exit", key="exit_session", icon=":material/logout:",
    on_click=set_session_ended, args=(True,), width="stretch",
)
st.write("Ask about safety, pets, and hiking in a small collection of official NPS pages.")
st.info(NOTICE + " Check official NPS sources for current conditions. Questions are answered independently.")

try:
    settings = Settings.load()
    settings.require_credentials()
    snapshot = load_active(settings)
except ValueError as exc:
    st.warning(str(exc))
    st.stop()

with st.sidebar:
    st.header("Source collection")
    st.caption("Collection time is when this app downloaded a page, not when NPS last updated it.")
    for source in snapshot["sources"]:
        st.link_button(source["title"], source["source_url"])
        st.caption("Collected (UTC): " + source["collected_at"])
    st.caption(f"{snapshot['page_count']} pages ? {snapshot['chunk_count']} passages")

question = st.chat_input("Ask a Yellowstone question", max_chars=2000)
if question:
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching collected NPS documents?"):
                result = ParkAssistant(settings, snapshot).ask(question)
            st.caption(result.status.capitalize())
            st.text(result.text)
            for source in result.sources:
                with st.expander(f"[{source['source_id']}] {source['title']}"):
                    st.link_button("Open supporting NPS page", source["url"])
                    st.caption("Collected (UTC): " + source["collected_at"])
                    st.text(source["text"])
        except Exception:  # noqa: BLE001 -- sanitize provider errors at the user-facing boundary
            st.error("The service could not complete this request. Check your local configuration, "
                     "network access, provider permissions and quota, then retry. "
                     "No answer was generated from unverified information.")
