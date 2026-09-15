# AI Agent (Streamlit + LangChain + Groq)

A beginner-friendly AI Agent built with LangChain and Groq's free LLM API,
deployed on Streamlit.

## What makes this an "agent" and not just a chatbot?
A chatbot only replies with text.
An **agent** can decide, on its own, to use tools before answering:
- 🔎 **Web Search** — looks up current information (DuckDuckGo, no API key needed)
- 🧮 **Calculator** — solves math expressions
- 🕐 **Date/Time** — tells you the current date and time

The agent reads your question, decides which tool (if any) it needs, uses it,
then gives you a final answer.

## How to run locally
1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Run the app:
   ```
   streamlit run app.py
   ```
3. Get a free API key from https://console.groq.com/keys, enter it in the sidebar.

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io and connect your repo.
3. Set `app.py` as the main file.
4. Deploy — no secrets needed since the key is entered in the app itself.

## Example prompts to try
- "What's the latest news about SpaceX?" → uses web search
- "What is 456 * 78 - 100?" → uses the calculator
- "What's today's date?" → uses the date/time tool
- "Hi, how are you?" → answers directly, no tool needed

## How it works
- `create_tool_calling_agent` builds an agent that can call the 3 tools above.
- `AgentExecutor` runs the agent, letting the LLM decide when to call a tool
  and feeding the tool's result back in before it replies.
- `openai/gpt-oss-20b` on Groq is used as the LLM because it supports tool calling.
