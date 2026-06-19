import streamlit as st
import requests
from collections import Counter

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AWS RAG Q&A", page_icon="📄")
st.title("📄 AWS Customer Agreement — RAG Q&A")
st.markdown("Powered by **LangChain** + **Groq** + **ChromaDB**")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Settings")
    st.divider()

    page = st.radio("Navigate", ["💬 Chat", "📊 Analytics"], index=0)

    st.divider()

    if st.button("📥 Ingest Document", use_container_width=True):
        with st.spinner("Ingesting..."):
            try:
                response = requests.post(f"{API_URL}/ingest", timeout=60)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "already_ingested":
                        st.info(f"Already ingested ({data['chunks']} chunks)")
                    else:
                        st.success(f"Done — {data['chunks_created']} chunks created")
                else:
                    st.error(f"Ingest failed: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI.")

    st.divider()
    st.caption(f"💬 Messages: {len(st.session_state.get('messages', []))}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# CHAT PAGE
if page == "💬 Chat":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for src in message["sources"]:
                        st.markdown(f"**Page {src['page']}**")
                        st.caption(src["snippet"])
            if message.get("score") is not None:
                st.caption(
                    f"Similarity: `{message['score']}` | Distance: `{message['distance']}`"
                )

    if question := st.chat_input("Ask something about the AWS Customer Agreement..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("▌")

            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=30,
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    message_placeholder.markdown(answer)

                    if data.get("sources"):
                        with st.expander("Sources"):
                            for src in data["sources"]:
                                st.markdown(f"**Page {src['page']}**")
                                st.caption(src["snippet"])

                    st.caption(
                        f"Similarity: `{data.get('similarity_score')}` | Distance: `{data.get('distance')}`"
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": data.get("sources", []),
                            "score": data.get("similarity_score"),
                            "distance": data.get("distance"),
                        }
                    )

                else:
                    err = response.json().get("detail", "Something went wrong.")
                    message_placeholder.error(f"Error {response.status_code}: {err}")

            except requests.exceptions.ConnectionError:
                message_placeholder.error("Cannot connect to FastAPI.")


# ANALYTICS PAGE
elif page == "📊 Analytics":
    st.subheader("📊 Usage Analytics")

    if st.button("🔄 Refresh"):
        st.rerun()

    with st.spinner("Loading analytics..."):
        try:
            response = requests.get(f"{API_URL}/analytics", timeout=10)

            if response.status_code == 200:
                data = response.json()

                st.markdown("### ⚡ Response Latency")
                col1, col2 = st.columns(2)
                col1.metric("Avg Latency", f"{round(data['avg_latency_ms'], 1)} ms")
                col2.metric("P95 Latency", f"{round(data['p95_latency_ms'], 1)} ms")

                st.divider()

                st.markdown("### 🔁 Most Frequently Asked Questions")
                top_questions = data.get("most_frequent_question", [])
                if top_questions:
                    labels = [
                        q["question"][:60] + "..."
                        if len(q["question"]) > 60
                        else q["question"]
                        for q in top_questions
                    ]
                    counts = [q["count"] for q in top_questions]
                    st.bar_chart(
                        {"Question": labels, "Count": counts}, x="Question", y="Count"
                    )
                else:
                    st.info("No query data yet.")

                st.divider()

                st.markdown("### ❌ Queries With No Answer Found")
                no_answer = data.get("no_answer_questions", [])
                if no_answer:
                    counts = Counter(no_answer)
                    na_data = {
                        "Question": [
                            q[:60] + "..." if len(q) > 60 else q for q in counts.keys()
                        ],
                        "Count": list(counts.values()),
                    }
                    st.bar_chart(na_data, x="Question", y="Count")
                else:
                    st.success("No unanswered queries so far.")

            else:
                st.error(f"Analytics endpoint returned {response.status_code}")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI. Make sure it is running on port 8000.")
