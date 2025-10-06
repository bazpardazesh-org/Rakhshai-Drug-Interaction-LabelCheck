# Manual drug interaction checks

The FastAPI service was installed with `pip install -r requirements.txt` and exercised locally using Uvicorn. Three representative payloads were submitted to `POST /check` to verify successful RxNorm lookups, label harvesting, and evidence scoring.

| Sample | Payload | Result highlights |
| --- | --- | --- |
| 1 | `["warfarin", "aspirin", "ibuprofen"]` | Aspirin/Ibuprofen flagged with `has_interaction=true` and 27 evidence sentences. Warfarin pairs returned `has_interaction=false`. |
| 2 | `["amiodarone", "fluconazole", "metformin"]` | Amiodarone/Fluconazole reported `has_interaction=true` with three corroborating snippets describing QT prolongation risk. |
| 3 | `["simvastatin", "clarithromycin"]` | Interaction detected with `max_confidence=0.8` and 11 supporting evidence sentences regarding elevated myopathy risk. |

Full JSON responses for each request are saved alongside this note for reference:

- [`manual_tests/warfarin_aspirin_ibuprofen.json`](warfarin_aspirin_ibuprofen.json)
- [`manual_tests/amiodarone_fluconazole_metformin.json`](amiodarone_fluconazole_metformin.json)
- [`manual_tests/simvastatin_clarithromycin.json`](simvastatin_clarithromycin.json)

These runs confirm that the application is functioning end-to-end in the provided environment.
