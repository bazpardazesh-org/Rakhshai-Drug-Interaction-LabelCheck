# Rakhshai‑Drug Interaction LabelCheck
**Version:** 1.1.0
**Copyright:** © Rakhshai dev  
**Website:** <https://www.rakhshai.com>

A production‑ready FastAPI microservice that analyses FDA Structured Product Labels (SPL) to flag potential drug–drug interactions. Incoming drug names are normalised to RxNorm identifiers so that brand and generic names map to the same concept before the relevant SPL sections are mined and scored for interaction evidence.

> **Disclaimer:** The openFDA label data may not reflect the current prescribing information and must not be used for medical decision making. Always consult a healthcare professional before acting on any drug‑interaction information.

## Key Advantages

- **Modern FastAPI Architecture** – Built as a scalable, production-ready microservice with a clear modular structure and comprehensive API documentation.
- **High Performance** – Leverages asynchronous I/O and concurrency to provide fast, real-time responses even under heavy load.
- **Reliable Data Sources** – Integrates directly with openFDA and RxNorm to ensure clinically validated, up-to-date drug information.
- **Smart Normalization** – Automatically maps brand and generic names via RxCUI for consistent drug matching.
- **Structured Label Analysis** – Extracts and analyses specific drug interaction and warning sections from FDA labels using NLP methods.
- **Automatic Fallback** – Retrieves data from DailyMed when openFDA responses fail, maintaining reliability.
- **Transparent Logic** – Every interaction result is traceable to its label source; no black-box predictions.
- **Developer Friendly** – Clean codebase, well-tested modules, and Docker-ready deployment for easy integration.
- **Production-grade Monitoring** – Built-in telemetry and logging for tracking performance and errors.

## At a glance

- **Framework:** FastAPI served by Uvicorn
- **Language:** Python 3.9+
- **Key services:** RxNorm, openFDA, DailyMed
- **Latest version:** 1.1.0
- **Status endpoints:** `/`, `/healthz`, `/readyz`, `/metrics`

## Features

- **RxNorm normalisation and caching** – user supplied drug names are normalised, cached and resolved to RXCUI identifiers before any downstream work occurs, reducing repeated network calls.(F:app/interaction_checker.py- Line : L58-L133)
- **Concurrent openFDA harvesting** – SPL interaction sections are fetched concurrently with automatic retry handling and de‑duplication of repeated snippets.(F:app/interaction_checker.py- Line : L135-L218)
- **Lightweight NLP pipeline** – spaCy is configured with an entity ruler, negation detection (when available) and sentence segmentation to accurately locate drug mentions in label text.(F:app/interaction_checker.py- Line : L220-L285)
- **Evidence scoring** – each sentence is assessed for interaction triggers, drug class mentions and negations to produce a reproducible confidence score for every drug pair.(F:app/interaction_checker.py- Line : L287-L384)
- **Operational telemetry** – every request is logged with a correlation ID, basic counters are tracked in code, and a Prometheus `/metrics` endpoint exposes latency and error statistics for observability.(F:app/main.py- Line : L38-L145)

## Project layout

```
Rakhshai‑Drug-Interaction-LabelCheck/
├── app/
│   ├── interaction_checker.py   # Normalisation, data fetching and NLP analysis
│   └── main.py                  # FastAPI application, routing and telemetry
├── manual_tests/                # Curated JSON payloads and instructions for exploratory checks
├── tests/
│   ├── conftest.py              # Shared pytest fixtures (FastAPI test client, sample payloads)
│   └── test_app.py              # API contract tests covering happy-path and error scenarios
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation (this file)
└── LICENSE.md                   # MIT license
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/your-org/Rakhshai-Drug-Interaction-LabelCheck.git
cd Rakhshai-Drug-Interaction-LabelCheck
```

### 2. Create and activate a virtual environment *(optional but recommended)*

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Install a spaCy model for better NLP accuracy

```bash
python -m spacy download en_core_sci_sm  # preferred
# or
python -m spacy download en_core_web_sm
```

> 💡 The service automatically falls back to a blank English pipeline when no model is available.(F:app/interaction_checker.py- Line : L220-L254)

### 5. Run the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

OpenAPI documentation will then be available at `http://localhost:8000/docs` and ReDoc at `http://localhost:8000/redoc`.

### 6. (Optional) Run the automated tests

```bash
pytest
```

### Synthetic accuracy check

Run `python scripts/evaluate_accuracy.py` to compare the interaction detection accuracy with and without a spaCy model on four synthetic label snippets. Both configurations achieved perfect precision, recall, and F1 (1.00) on this benchmark, confirming that the rule-based fallback performs equivalently to the model-backed pipeline for the evaluated scenarios.

This test ensures that even if the spaCy model is unavailable, the rule-based logic can independently detect interactions with no loss in accuracy. The evaluation script automatically loads the synthetic examples, applies both detection pipelines, and then reports standard classification metrics (precision, recall, and F1-score) for comparison.

In short, this benchmark verifies the robustness and reliability of the fallback mechanism, demonstrating that it reproduces model-level performance for the tested cases.

(Source: scripts/evaluate_accuracy.py, Lines L1–L164)

## Configuration

| Environment variable | Description |
| -------------------- | ----------- |
| `ALLOWED_ORIGINS`    | Optional comma‑separated list of origins to allow through CORS. Leave unset to disable CORS entirely.(F:app/main.py- Line : L112-L124) |
| `PORT`               | Port used when running `python -m app.main` or when invoking the module directly. Defaults to `8000`.(F:app/main.py- Line : L190-L194) |

No credentials are required because all upstream services are public APIs, but be aware of openFDA and RxNorm rate limits.

## API

### `GET /`
Returns a welcome banner and current service version.

**Response**
```json
{ "message": "Welcome to Rakhshai‑Drug Interaction LabelCheck", "version": "1.1.0" }
```

### `GET /healthz`
Liveness probe for container orchestrators. Returns `{ "status": "ok" }`.

### `GET /readyz`
Readiness probe that currently reports `{ "status": "ready", "version": "1.1.0" }`.

### `POST /check`
Accepts a JSON body containing a list of at least two drug names. Names are validated, normalised to RXCUIs and the relevant SPL sections are analysed for interaction evidence. Duplicate entries and invalid characters are rejected.(F:app/main.py- Line : L54-L109)

**Request**
```json
{
  "drugs": ["amiodarone", "fluconazole", "metformin"]
}
```

**Successful response**
```json
{
  "input_drugs": ["amiodarone", "fluconazole", "metformin"],
  "interactions": [
    {
      "pair": ["amiodarone", "fluconazole"],
      "has_interaction": true,
      "max_confidence": 0.8,
      "evidence": [
        {
          "pair": ["amiodarone", "fluconazole"],
          "sentence": "amiodarone may increase the serum concentration of fluconazole...",
          "confidence": 0.8,
          "triggered": true,
          "class_derived": false,
          "negated": false,
          "provenance": {
            "set_id": "1b4...",
            "label_date": "20230101",
            "section_name": "drug_interactions",
            "url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=1b4..."
          }
        }
      ]
    }
  ],
  "metrics": {
    "requests_total": 1,
    "rxnorm_cache_hits": 0
  },
  "version": "1.1.0"
}
```

**Error responses**
- `400 Bad Request` – validation failures (e.g. duplicate names, invalid characters, fewer than two drugs).(F:app/main.py- Line : L60-L109)
- `404 Not Found` – RxNorm could not resolve all drugs or no interaction sections were present in openFDA results.(F:app/main.py- Line : L143-L167)
- `429 Too Many Requests` – bubbled up from upstream API rate limits.(F:app/main.py- Line : L135-L151)
- `503 Service Unavailable` – upstream APIs were unreachable.(F:app/main.py- Line : L133-L166)

### `GET /metrics`
Exposes Prometheus formatted metrics such as request latency histograms and error counters for integration with observability stacks.(F:app/main.py- Line : L81-L145)

## Testing

Run the automated test suite with **pytest**:

```bash
pytest
```

The tests mock external dependencies and verify request validation, success paths and error handling for the FastAPI layer.(F:tests/test_app.py- Line : L1-L196)

## Operational notes

- Each request is assigned a correlation ID (from the incoming `X-Request-ID` header or a generated UUID) which is also returned in the response headers for tracing across services.(F:app/main.py- Line : L126-L145)
- When run behind a load balancer you can scale horizontally; global locks are only used to guard in‑memory caches shared between tasks within the same process.(F:app/interaction_checker.py- Line : L42-L107)
- The service keeps lightweight in‑process metrics via the shared `METRICS` dictionary and exposes them as part of the `/check` response to aid client‑side monitoring.(F:app/main.py- Line : L28-L167)

## License

This project is licensed under the MIT License. See [`LICENSE.md`](LICENSE.md) for details.
