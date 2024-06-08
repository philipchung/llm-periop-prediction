# Large Language Model Capabilities in Perioperative Risk Prediction and Prognostication

This repo contains code for [Chung P, Fong CT, Walters AM, Aghaeepour N, Yetisgen M, O’Reilly-Shah VN. Large Language Model Capabilities in Perioperative Risk Prediction and Prognostication. JAMA Surg. Published online June 05, 2024. doi:10.1001/jamasurg.2024.1621](https://jamanetwork.com/journals/jamasurgery/fullarticle/2819795).

This research study examines how well general-domain LLMs such as GPT-4 Turbo can predict a patient's risk classification and outcomes after surgery or a procedure. Specifically, GPT-4 Turbo is used to predict a patient's pre-operative illness severity (ASA Physical Status Classification), whether a patient will have post-operative hospitalization/ICU admission/unanticipated admission/death, and duration of time in PACU phase 1/hospital admission/ICU admission. 

GPT-4 Turbo is provided pre-procedure clinical notes as a representation of the patient and the procedure case booking information to make the prediction. These input data are derived from real retrospective electronic health record data. Ground truth labels are also derived from real retrospective electronic health record data, meaning that GPT-4 Turbo's predictions are compared against actual real-world outcomes rather than human ratings or preferences.

## Data

Data is derived from electronic health record (EHR) used for routine clinical care. As such, it needs to be cleaned and processed before it is suitable for machine learning. Task-specific datasets are created for each of the 8 prediction tasks studied. Finally prompts are created from the data, predictions are generated from LLM, and outputs are extracted from text.

* `data_query` contains scripts to query SQL database and export tables.
* `make_dataset` transforms the raw data tables into cleaned data and then generates the task-specific datasets.
* `llm` contains core components for llm experiments.  This contains wrappers around OpenAI's ChatCompletion API, prompt composition, output parsing/validation, and persistence of prompts+results in database.
* `scripts` contains the main inferencing script used to generate LLM outputs for experiments
* `analysis` contains scripts and jupyter notebooks executing the experiments, analysis, and results

## Secrets

OpenAI API secrets are stored in file `config.py` (which is not in this repo because of `.gitignore`). We also store postgres config in this file. Create this file and put with your own API keys and endpoint. 

```python
## config.py

# OpenAI API secrets
USWEST_OPENAI_API_KEY = "my-api-key"
USWEST_OPENAI_API_ENDPOINT = "https://chat_model_endpoint.com/"
USWEST_OPENAI_API_VERSION = "2023-09-01-preview"

# Postgres
postgres_user = "postgres"
postgres_password = "postgres"
postgres_host = "localhost"
postgres_port = 5432
postgres_database = "postgresdb"
```

Notes on PostgresSQL are in `Postgres.md`.

## Python & Poetry

This project uses pyenv and targets python 3.10.11. All dependencies are managed with poetry. `pyenv_poetry_setup.sh` is a shell script that installs these tools.
