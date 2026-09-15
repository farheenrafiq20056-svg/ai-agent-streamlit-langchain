import streamlit as st
from datetime import datetime

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# ---------------------------------------------------
# Simple AI Agent App using Streamlit + LangChain + Groq
# Beginner friendly version - easy to read and edit
#
# Difference from a normal chatbot:
# A chatbot just replies with text.
# An AGENT can decide to USE TOOLS (like searching the
# web or doing math) before giving you the final answer.
# ---------------------------------------------------

st.set_page_config(page_title="My AI Agent", page_icon="🕵️")

st.title("🕵️ My AI Agent")
st.write(
    "This is an AI Agent. Unlike a simple chatbot, it can decide to "
    "search the web or do calculations to answer your question."
)

# ---------------------------
# Step 1: Get API key from user
# ---------------------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter your Groq API Key", type="password")
    st.caption("Get a free key at console.groq.com/keys. It is only used for this session, never stored.")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------
# Step 2: Define the tools the agent is allowed to use
# ---------------------------

# Tool 1: Search the web (free, no API key needed)
search_tool = DuckDuckGoSearchRun()


# Tool 2: Do a math calculation
@tool
def calculator(expression: str) -> str:
    """Use this to calculate a math expression, e.g. '25 * 4 + 10'."""
    try:
        # eval is used here only for simple beginner-level math expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Could not calculate that: {e}"


# Tool 3: Get the current date and time
@tool
def get_current_datetime(_: str = "") -> str:
    """Use this to find out today's date or the current time."""
    return datetime.now().strftime("%A, %d %B %Y - %I:%M %p")


tools = [search_tool, calculator, get_current_datetime]

# ---------------------------
# Step 3: Set up chat history
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Step 4: Show previous messages
# ---------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ---------------------------
# Step 5: Get new user input
# ---------------------------
user_input = st.chat_input("Ask me something the agent can research or calculate...")

if user_input:
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar first.")
    else:
        # Show user's message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Build and run the agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking and using tools if needed..."):
                try:
                    llm = ChatGroq(
                        api_key=api_key,
                        model="openai/gpt-oss-20b",
                        temperature=0.3,
                    )

                    agent = create_agent(
                        model=llm,
                        tools=tools,
                        system_prompt=(
                            "You are a helpful AI agent. Use tools when they "
                            "help you give a more accurate or up-to-date answer."
                        ),
                    )

                    result = agent.invoke(
                        {"messages": [{"role": "user", "content": user_input}]}
                    )
                    bot_reply = result["messages"][-1].content

                    st.write(bot_reply)

                    # Save bot reply to chat history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": bot_reply}
                    )

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
