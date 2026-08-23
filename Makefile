.PHONY: setup run app clean

VENV := .venv
PORT ?= 8000

# Build the virtualenv and install dependencies. The uvicorn binary doubles
# as a build stamp: if it's missing (fresh worktree) or requirements.txt is
# newer, the venv is (re)built. This lets `run` self-provision so a skipped or
# interrupted `setup` can't leave a half-built or missing .venv behind.
$(VENV)/bin/uvicorn: requirements.txt
	python3.12 -m venv $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

setup: $(VENV)/bin/uvicorn
	@echo ""
	@echo "Setup complete. Now configure your API key:"
	@echo "  cp .env.example .env"
	@echo "  Then edit .env and add your HuggingFace token"

# Always launch via the venv's own interpreter (never `source activate`), so
# the workspace's Python is used regardless of any venv active in the parent
# shell. Override the port with `make run PORT=1234` (Conductor passes
# $CONDUCTOR_PORT this way).
run: $(VENV)/bin/uvicorn
	PORT=$(PORT) $(VENV)/bin/python run.py

# Build the native macOS app (debug) and open it. Incremental: only changed
# Swift recompiles. Dev builds embed the lightweight stub service, so this is a
# fast Swift/SwiftUI iteration loop — it does NOT do real transcription (that
# needs the PyInstaller bundle at dist/MeetingTranscriber). For transcription
# work use `make run` instead.
app:
	bash macos/scripts/build_app.sh debug
	open macos/build/Blah.app

clean:
	rm -rf $(VENV) data/meetings/*
