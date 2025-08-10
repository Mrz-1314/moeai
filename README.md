# Unified AI Chat Service

This project provides a simple web application that unifies three AI providers
— OpenAI ChatGPT (GPT‑4o), Google Gemini and xAI Grok — behind a single chat
interface. It demonstrates a minimalistic implementation of a **Mixture of
Experts** routing strategy where the front‑end selects the provider, and the
back‑end dispatches to the appropriate API.

## Features

* Choose between OpenAI, Google Gemini or xAI Grok for your query.
* Maintains conversation history for contextual replies.
* Simple HTML/JavaScript front‑end; FastAPI back‑end.
* CORS enabled for easy integration with other front‑ends.
* Graceful fallback when API keys are missing.

## Getting Started

### 1. Install dependencies

Create a virtual environment (optional but recommended) and install the
requirements:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastapi uvicorn python-dotenv openai google-generativeai \
            anthropic langchain-openai httpx
```

### 2. Configure environment variables

Create a `.env` file in the project root (`ai_moe_site/.env`) with the
following content, replacing the placeholders with your actual API keys:

```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...your_google_generative_ai_key...
XAI_API_KEY=...your_xai_api_key...
```

If an API key is not provided, the server will return a helpful message
indicating that the key is missing.

### 3. Run the server

Use `uvicorn` to run the FastAPI application:

```bash
uvicorn main:app --reload --port 8000
```

The `--reload` flag enables auto‑reloading on code changes. When the server is
running, open your browser and navigate to `http://localhost:8000` to access
the chat interface.

## Usage

1. Select a provider from the drop‑down (ChatGPT, Gemini or Grok).
2. Type your message and press **Send**.
3. The AI's response will appear below. The conversation history is
   preserved so that subsequent messages include context.

## Notes

* **Security**: Do not expose your API keys publicly. Store them in
  environment variables or secret management services in production.
* **Routing**: This demo uses a simple provider selection. For a real
  Mixture‑of‑Experts approach, incorporate a router/gating mechanism (e.g.,
  rules or ML model) to automatically choose the best expert based on the
  prompt.
* **Error handling**: The server will return error messages if the API call
  fails. In production you should implement robust error handling and
  logging.

## License

This project is provided as an educational example. Feel free to modify and
adapt it for your needs.