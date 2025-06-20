## 🛠️ Setup with Poetry

```bash
# 1. Install Poetry (if not already):
curl -sSL https://install.python-poetry.org | python3 -

# 2. Clone & enter project:
git clone https://github.com/yourorg/codebase-assistant.git
cd codebase-assistant

# 3. Install deps:
poetry install

# 4. Configure:
cp config/settings.toml.example config/settings.toml
# → Fill in your OpenAI key, repo URL, etc.

# 5a. Ingest & embed:
poetry run python main.py ingest

# 5b. Chat:
poetry run python main.py chat

# 5c. Streamlit demo:
cd streamlit_app
poetry run streamlit run app.py
