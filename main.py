from flask import Flask, render_template, request, jsonify, session, send_file
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings
import requests
import json
import os
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import io
from datetime import date

# ── Research agent imports ────────────────────────────────────


app = Flask(__name__)
app.secret_key = "emre_ai_secret_123"

# ── Your ElevenLabs API key ───────────────────────────────────
ELEVENLABS_API_KEY = "sk_812e420469d5b43b470708397509e2242a9cd05796e32a6b"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3n"

MAX_HISTORY = 20
MAX_SEARCH_RESULTS = 4
MAX_PAGE_CHARS = 3000

SYSTEM_PROMPT = """You are Jarvis, a highly intelligent AI assistant. You:
- Give concise answers unless asked to elaborate
- Admit when you don't know something
- Ask clarifying questions when a request is ambiguous
- Format responses with markdown when helpful
- Remember context from earlier in the conversation
- Be proactive: if you notice a better way to do something, mention it
Today's date is: {date}
"""

RESEARCH_SYSTEM_PROMPT = """You are an expert research assistant with access to web search tools.

Your task:
1. Search the web to gather information on the given topic
2. Fetch relevant pages to get more detail when needed
3. Synthesize everything into a well-structured report

Your final report MUST include:
- Executive Summary
- Key Findings (grouped by theme)
- Important facts and statistics
- Conclusion
- Sources (list URLs used)

Use markdown formatting. Be thorough but concise."""


# ── Helpers ───────────────────────────────────────────────────

def load_history(user_id):
    path = f"histories/{user_id}.json"
    try:
        return json.load(open(path)) if os.path.exists(path) else []
    except (json.JSONDecodeError, IOError):
        return []

def save_history(user_id, history):
    os.makedirs("histories", exist_ok=True)
    with open(f"histories/{user_id}.json", "w") as f:
        json.dump(history, f)


# ── Research agent tools ──────────────────────────────────────

def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_SEARCH_RESULTS))
        if not results:
            return "No results found."
        output = []
        for r in results:
            output.append(f"Title: {r['title']}\nURL: {r['href']}\nSummary: {r['body']}\n")
        return "\n---\n".join(output)
    except Exception as e:
        return f"Search error: {e}"


def fetch_page(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:MAX_PAGE_CHARS] + ("..." if len(text) > MAX_PAGE_CHARS else "")
    except Exception as e:
        return f"Could not fetch page: {e}"


TOOLS = {
    "web_search": {
        "fn": web_search,
        "spec": {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web using DuckDuckGo.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            }
        }
    },
    "fetch_page": {
        "fn": fetch_page,
        "spec": {
            "type": "function",
            "function": {
                "name": "fetch_page",
                "description": "Fetch and read the full text content of a web page by URL.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The full URL to fetch"}
                    },
                    "required": ["url"]
                }
            }
        }
    }
}

TOOL_SPECS = [t["spec"] for t in TOOLS.values()]


def run_research_agent(topic: str) -> str:
    """Agentic loop — searches the web and writes a structured report."""
    messages = [
        {"role": "system", "content": RESEARCH_SYSTEM_PROMPT},
        {"role": "user", "content": f"Research this topic and write a comprehensive report: {topic}"}
    ]

    for _ in range(10):  # max iterations
        payload = {
            "model": MODEL,
            "messages": messages,
            "tools": TOOL_SPECS,
            "stream": False
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        msg = resp.json().get("message", {})
        messages.append(msg)

        tool_calls = msg.get("tool_calls", [])

        if not tool_calls:
            return msg.get("content", "No report generated.")

        for call in tool_calls:
            fn_name = call["function"]["name"]
            fn_args = call["function"].get("arguments", {})
            result = TOOLS[fn_name]["fn"](**fn_args) if fn_name in TOOLS else f"Unknown tool: {fn_name}"
            messages.append({"role": "tool", "content": result})

    return "Research timed out after maximum iterations."


# ── Routes ────────────────────────────────────────────────────

@app.route("/")
def home():
    session["history"] = []
    session.setdefault("user_id", os.urandom(8).hex())
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message", "")
    user_id      = session.get("user_id", "default")
    history      = load_history(user_id)

    history.append({"role": "user", "content": user_message})
    history = history[-MAX_HISTORY:]

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(date=date.today())}
        ] + history,
        "stream": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "repeat_penalty": 1.1}
    }

    try:
        response   = requests.post(OLLAMA_URL, json=payload, timeout=120)
        result     = response.json()
        ai_message = result["message"]["content"]

        history.append({"role": "assistant", "content": ai_message})
        session["history"] = history
        save_history(user_id, history)

        return jsonify({"response": ai_message})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"}), 500


@app.route("/research", methods=["POST"])
def research():
    """New endpoint — runs the full research agent on a topic."""
    topic = request.json.get("topic", "").strip()
    if not topic:
        return jsonify({"error": "No topic provided"}), 400

    try:
        report = run_research_agent(topic)

        # Optionally save the report to disk
        os.makedirs("reports", exist_ok=True)
        safe = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)
        filename = f"reports/{safe[:40].replace(' ','_')}_{date.today()}.md"
        with open(filename, "w") as f:
            f.write(f"# {topic}\n\n{report}")

        return jsonify({"topic": topic, "report": report, "saved_to": filename})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/research/list", methods=["GET"])
def list_reports():
    """Returns a list of all saved research reports."""
    files = []
    if os.path.exists("reports"):
        files = sorted(os.listdir("reports"), reverse=True)
    return jsonify({"reports": files})


# ── Text to Speech ────────────────────────────────────────────

@app.route("/speak", methods=["POST"])
def speak():
    text     = request.json.get("text", "")
    voice_id = request.json.get("voice_id", "EXAVITQu4vr4xnSDxMaL")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=0.5,
                similarity_boost=0.75,
                style=0.5,
                use_speaker_boost=True
            )
        )

        audio_buffer = io.BytesIO()
        for chunk in audio:
            audio_buffer.write(chunk)
        audio_buffer.seek(0)

        return send_file(audio_buffer, mimetype="audio/mpeg", as_attachment=False)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/voices", methods=["GET"])
def get_voices():
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        voices = client.voices.get_all()
        voice_list = [{"id": v.voice_id, "name": v.name} for v in voices.voices]
        return jsonify({"voices": voice_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)