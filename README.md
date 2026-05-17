# MedAI - Medical Literature and Patient Education Assistant

LIVE LINK: https://rag-assistant-j4bwmtgxuottghxnheayqv.streamlit.app/

MedAI is a Streamlit assistant that answers health questions using retrieved medical sources. It combines patient-friendly education sources for common symptom and treatment questions with research literature and drug-label sources for deeper clinical and research questions.

This app is for educational awareness only. It does not diagnose, prescribe, or replace a licensed medical professional.

## What It Uses

| Source | Best for | Cost |
| --- | --- | --- |
| MedlinePlus | Patient-friendly disease, symptom, prevention, and treatment overviews | Free |
| CDC Content Syndication API | Public health pages such as stroke warning signs and prevention topics | Free |
| PubMed | Biomedical research abstracts | Free |
| Europe PMC | Biomedical literature and open-access article metadata | Free |
| OpenFDA Drug Label API | Drug warnings, interactions, contraindications, and dosage label sections | Free |
| Gemini 2.5 Flash | Safe query extraction and grounded answer generation | API key |
| OpenRouter | Optional fallback answer generation | Free tier / API key |

## How It Works

```text
User question
     |
     v
[Safe query extraction and query routing]
     |
     +-- Gemini extracts symptoms/search terms without diagnosing
     +-- Patient education questions -> MedlinePlus + CDC
     +-- Drug questions -> OpenFDA + literature
     +-- Latest/research questions -> PubMed + Europe PMC
     |
     v
[Source ranking and filtering]
     |
     v
[Strict grounded prompt]
     |
     v
[Gemini model, with OpenRouter fallback]
     |
     v
Answer grounded only in retrieved sources
```

## Improvements Made

- Common symptom and treatment questions now prefer MedlinePlus and CDC sources.
- Research questions such as "latest research on type 2 diabetes" still prioritize PubMed and Europe PMC.
- Drug questions such as "Can metformin cause lactic acidosis?" use OpenFDA and drug-specific literature.
- The assistant no longer says "no related data found" while also citing relevant sources.
- OpenFDA query generation is safer and no longer sends the full natural-language question as a drug-label phrase.
- Gemini can extract symptoms and search terms without diagnosing the user.
- Gemini is preferred for grounded answer generation when `GEMINI_API_KEY` is configured.
- OpenRouter 429 rate-limit errors are handled with retry/fallback behavior and a cleaner user message.
 - Common misspellings of drug and medical terms are auto-corrected (e.g. `paracetemol` → `paracetamol`) before searching sources.
 - Europe PMC/network errors are handled gracefully so transient SSL/timeouts from Europe PMC don't break the whole query — the system falls back to other available sources.

## Example Questions

- Symptoms and treatment steps for hypertension
- Latest research on type 2 diabetes
- Can metformin cause lactic acidosis?
- What are warning signs of stroke?

## Running Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a local `secrets.toml` file:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
GEMINI_MODEL = "gemini-2.5-flash"

OPENROUTER_API_KEY = "sk-or-your-openrouter-key"
OPENROUTER_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
OPENROUTER_FALLBACK_MODELS = "meta-llama/llama-3.3-70b-instruct:free,openrouter/free"

NCBI_API_KEY = ""
NCBI_EMAIL = "you@example.com"
NCBI_TOOL = "medical-rag-assistant"
```

Run the app:

```bash
streamlit run app.py
```

Notes:
- Typo correction and query normalization happen automatically in the pipeline; no additional config required.
- If you rely on OpenRouter as a fallback, ensure `OPENROUTER_MODEL` and `OPENROUTER_API_KEY` are set in `secrets.toml` or the sidebar. The project also includes a safe built-in OpenRouter default model used when a configured model fails.

## Project Structure

```text
RAG health/
|-- app.py
|-- rag_pipeline.py
|-- load_secrets.py
|-- requirements.txt
|-- secrets.toml.example
`-- README.md
```

## Notes

- `faiss_index.bin` and `chunks.pkl` are old local RAG artifacts and are not used by the current live medical assistant.
- MedlinePlus, CDC, PubMed, Europe PMC, and OpenFDA are queried live at question time.
- The answer generator is instructed to answer only from retrieved sources, avoid diagnosis, and refuse when evidence is weak.
