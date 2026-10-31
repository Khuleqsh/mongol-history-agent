"""Ажиллуулах: pip install -e '.[ui]' && streamlit run app/streamlit_app.py"""
import streamlit as st

from mha.agent import Agent
from mha.feedback import log_feedback

st.set_page_config(page_title="Монголын түүхийн судлаач", page_icon="📜")
st.title("📜 Монголын түүхийн судлаач")


@st.cache_resource
def get_agent() -> Agent:
    return Agent()


agent = get_agent()
level = st.sidebar.selectbox("Тайлбарын түвшин", ["standard", "brief", "student", "researcher"])
verify = st.sidebar.checkbox("Ишлэлийг давхар шалгах (удаан)", value=False)

if "chat" not in st.session_state:
    st.session_state.chat = []

for i, m in enumerate(st.session_state.chat):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            with st.expander("Эх сурвалж"):
                for label, s in m["sources"].items():
                    st.write(f'**[{label}]** {s["title"]} — {s["author"]} {s["year"]} ({s["lang"]})')
            c1, c2, _ = st.columns([1, 1, 8])
            if c1.button("👍", key=f"up{i}"):
                log_feedback(m["question"], m["content"], "good")
            if c2.button("👎", key=f"down{i}"):
                log_feedback(m["question"], m["content"], "bad")

q = st.chat_input("Асуултаа бич (Монгол / English)")
if q:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat]
    st.session_state.chat.append({"role": "user", "content": q})
    with st.spinner("Архивт хайж байна..."):
        r = agent.ask(q, level=level, history=history, verify=verify)
    st.session_state.chat.append(
        {"role": "assistant", "content": r["answer"], "question": q, "sources": r["sources"]}
    )
    st.rerun()
