# Namma-Flow: thin convenience wrappers around the project's CLIs (see README.md).
#
#   make help                                   list the targets
#   make dataset ARGS=--force                   extra flags reach the underlying CLI through ARGS
#   make train ARGS="--num-threads 7"
#   make offline-demo && make app CONFIG=/tmp/namma-flow-demo/config.yaml
#   make clean-demo                             deletes only what offline-demo created in DEMO_DIR
#   make check-env                              fails unless the installed packages equal constraints.txt
#
# Every variable below can be overridden on the command line (make VAR=value target).
# Requires GNU make (macOS ships 3.81, which is enough).

PY             ?= .venv/bin/python
CONFIG         ?= config/config.yaml
ARGS           ?=
SYSTEM_PYTHON  ?= python3
REPORTS_DIR    ?= artifacts/reports
PRESET         ?= cloudburst
PREDICTOR      ?= --physics
BACKTEST_START ?= 2024-08-01
BACKTEST_END   ?= 2024-11-28
CALIBRATE_ARGS ?= --strict
PYTEST_ARGS    ?=
STREAMLIT_ARGS ?=

# Offline demo: every path of CONFIG is re-rooted under DEMO_DIR, so the demo can never
# overwrite the real data/ or artifacts/ directories. src/utils/demo_dir.py refuses a DEMO_DIR that
# is /, $HOME, the project or an ancestor of either, lies inside / contains the project's data/,
# artifacts/ or config/, or would make the demo config CONFIG itself; it adopts only a new, empty or
# previously prepared folder (marker file .namma-flow-demo), and clean-demo deletes only what the
# marker lists (the demo config and the re-rooted data/ and artifacts/ trees).
DEMO_DIR       ?= /tmp/namma-flow-demo
DEMO_EPOCHS    ?= 3
DEMO_WINDOWS   ?= 64
DEMO_ROOT      := $(abspath $(DEMO_DIR))
DEMO_CONFIG    := $(DEMO_ROOT)/config.yaml
DEMO_REPORTS   := $(DEMO_ROOT)/artifacts/reports
DEMO_RUN       := NAMMA_FLOW_OFFLINE=1 $(PY)
DEMO_TOOL       = $(PY) -m src.utils.demo_dir
DEMO_TOOL_ARGS  = --root "$(DEMO_ROOT)" --project "$(CURDIR)"

# Lowest Python the pinned stack (constraints.txt) installs on; keep equal to pyproject's requires-python.
MIN_PYTHON     := 3.12
PY_VERSION_CHECK = -c 'import sys; need = tuple(int(p) for p in "$(MIN_PYTHON)".split(".")); \
	sys.exit(0 if sys.version_info[:2] >= need else "Namma-Flow needs Python >= $(MIN_PYTHON) (tested 3.13): " \
	+ sys.executable + " is " + sys.version.split()[0] \
	+ ". Use e.g. make setup SYSTEM_PYTHON=python3.13 (delete an old .venv first).")'

.DEFAULT_GOAL := help
.PHONY: help setup network elevation weather dataset train predict backtest app test coverage check-env calibrate \
	offline-demo demo-config clean-reports clean-demo

help: ## List the targets and the main variables
	@echo "Namma-Flow targets (CONFIG=$(CONFIG), PY=$(PY), Python >= $(MIN_PYTHON)):"
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-14s %s\n", $$1, $$2}'
	@echo "Extra CLI flags: ARGS=...  (e.g. make dataset ARGS=--force). Offline: NAMMA_FLOW_OFFLINE=1 make <target>."

setup: ## Create .venv (if missing; checks Python >= MIN_PYTHON) and install the pinned dependencies
	test -x $(PY) || { $(SYSTEM_PYTHON) $(PY_VERSION_CHECK) && $(SYSTEM_PYTHON) -m venv .venv; }
	$(PY) $(PY_VERSION_CHECK)
	$(PY) -m pip install -r requirements.txt -c constraints.txt

network: ## Stage 01: OSM road graph + elevation + drains (paths.graph_file)
	$(PY) src/data_pipeline/01_extract_network.py --config $(CONFIG) $(ARGS)

elevation: ## Stage 02 (optional): re-run elevation / drain enrichment of the graph
	$(PY) src/data_pipeline/02_elevation_engine.py --config $(CONFIG) $(ARGS)

weather: ## Stage 03 (optional): fetch / extend the hourly Open-Meteo rainfall record
	$(PY) src/data_pipeline/03_weather_ingestion.py --config $(CONFIG) $(ARGS)

dataset: ## Stage 04: train / validation / test datasets with simulated flood labels
	$(PY) src/data_pipeline/04_dataset_builder.py --config $(CONFIG) $(ARGS)

train: ## Train, calibrate and evaluate the GNN (publishes best.pt + metrics.json)
	$(PY) src/training/train.py --config $(CONFIG) $(ARGS)

predict: ## What-if design storm PRESET (physics baseline; PREDICTOR= uses the trained GNN)
	$(PY) -m src.inference.predict --config $(CONFIG) --scenario design --preset $(PRESET) $(PREDICTOR) $(ARGS)

backtest: ## Forecast backtest BACKTEST_START..BACKTEST_END (online; ARGS=--physics without a model)
	$(PY) -m src.inference.predict --config $(CONFIG) --backtest $(BACKTEST_START) $(BACKTEST_END) $(ARGS)

app: ## Streamlit dashboard on CONFIG (passed as NAMMA_FLOW_CONFIG)
	NAMMA_FLOW_CONFIG=$(CONFIG) $(PY) -m streamlit run app/app.py $(STREAMLIT_ARGS)

# The test suite builds its own config; an exported NAMMA_FLOW_CONFIG must not leak into it.
test: ## Offline test suite (PYTEST_ARGS='-m unit' etc.)
	NAMMA_FLOW_CONFIG= $(PY) -m pytest $(PYTEST_ARGS)

coverage: ## Test suite with line coverage of src/ and app/
	NAMMA_FLOW_CONFIG= $(PY) -m pytest --cov --cov-report=term-missing:skip-covered $(PYTEST_ARGS)

# `make test` only reports version drift (skip); this target fails unless every installed package is pinned exactly.
check-env: ## Fail unless the installed packages equal the constraints.txt pins exactly
	NAMMA_FLOW_CONFIG= NAMMA_FLOW_STRICT_ENV=1 $(PY) -m pytest tests/test_utils.py \
		-k "constraints or python_floor or osmnx" $(PYTEST_ARGS)

calibrate: ## Hydrology label calibration diagnostics, criteria (a)-(h) (~10 s, ~1.5 GB)
	$(PY) -m src.hydrology.calibrate --config $(CONFIG) $(CALIBRATE_ARGS) $(ARGS)

# Phony, so the demo config is always regenerated (and the folder re-checked) even when it looks up to date.
demo-config: ## Check DEMO_DIR and write the offline demo config (CONFIG re-rooted under DEMO_DIR)
	$(DEMO_TOOL) prepare $(DEMO_TOOL_ARGS) --config "$(CONFIG)"

offline-demo: demo-config ## Full pipeline offline on synthetic data under DEMO_DIR (a few minutes)
	$(DEMO_TOOL) check $(DEMO_TOOL_ARGS) --config "$(CONFIG)"
	$(DEMO_RUN) src/data_pipeline/01_extract_network.py --config "$(DEMO_CONFIG)" --offline
	$(DEMO_RUN) src/data_pipeline/03_weather_ingestion.py --config "$(DEMO_CONFIG)" --offline
	$(DEMO_RUN) src/data_pipeline/04_dataset_builder.py --config "$(DEMO_CONFIG)" --offline
	$(DEMO_RUN) src/training/train.py --config "$(DEMO_CONFIG)" --offline --epochs $(DEMO_EPOCHS) --windows-per-epoch $(DEMO_WINDOWS)
	$(DEMO_RUN) -m src.inference.predict --config "$(DEMO_CONFIG)" --offline --scenario forecast --hours 24 \
		--out "$(DEMO_REPORTS)/forecast_fallback.geojson"
	$(DEMO_RUN) -m src.inference.predict --config "$(DEMO_CONFIG)" --offline --scenario design --preset $(PRESET) --mc 5 \
		--out "$(DEMO_REPORTS)/design_storm.geojson"
	@echo "Offline demo ready in $(DEMO_ROOT). Dashboard: make app CONFIG=$(DEMO_CONFIG)"

clean-reports: ## Delete regenerable CLI outputs in REPORTS_DIR (prediction.*, backtest.json, backtest_physics.json)
	rm -f $(REPORTS_DIR)/prediction.geojson $(REPORTS_DIR)/prediction.csv $(REPORTS_DIR)/backtest.json \
		$(REPORTS_DIR)/backtest_physics.json

clean-demo: ## Delete what offline-demo created in DEMO_DIR (only a folder with its .namma-flow-demo marker)
	$(DEMO_TOOL) clean $(DEMO_TOOL_ARGS)
