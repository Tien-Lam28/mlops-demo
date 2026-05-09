install:
	uv sync

train:
	dvc exp run

serve:
	uv run uvicorn src.api:app --reload

monitor:
	uv run python src/monitor.py