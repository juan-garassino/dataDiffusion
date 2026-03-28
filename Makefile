.PHONY: install test train evaluate improve clean lint mlflow help

# --- Setup ---
install:                          ## Install all dependencies
	pip install -r requirements.txt

# --- Core pipeline ---
train:                            ## Train diffusion model (single run)
	python main.py --dataset california --num_epochs 100

train-quick:                      ## Quick training run (10 epochs, for sanity checking)
	python main.py --dataset california --num_epochs 10

evaluate:                         ## Train + generate + evaluate synthetic vs real
	python main.py --dataset california --num_epochs 50

evaluate-report:                  ## Run evaluation and print detailed quality report
	python main.py --dataset california --num_epochs 50 --verbose

# --- Self-improvement loop ---
improve:                          ## Run self-improvement loop (train → eval → adjust → repeat)
	python main.py --dataset california --self_improve \
		--max_iterations 10 --quality_threshold 0.8

improve-quick:                    ## Quick self-improvement (3 iterations, lower bar)
	python main.py --dataset california --self_improve \
		--max_iterations 3 --quality_threshold 0.5 --num_epochs 20

# --- Testing ---
test:                             ## Run all tests
	python -m pytest tests/ -v

test-fast:                        ## Run tests excluding slow integration tests
	python -m pytest tests/ -v -m "not slow"

test-eval:                        ## Run only evaluation tests
	python -m pytest tests/test_evaluation.py -v

test-pipeline:                    ## Run only self-improvement pipeline tests
	python -m pytest tests/test_pipeline.py -v

# --- Experiment tracking ---
mlflow:                           ## Launch MLflow UI to inspect runs
	mlflow ui --port 5000

# --- Cleanup ---
clean:                            ## Remove generated artifacts
	rm -rf __pycache__ datadiffusion/__pycache__ datadiffusion/**/__pycache__
	rm -rf logs/*.log
	rm -rf results/
	rm -rf .pytest_cache

clean-experiments:                ## Remove all experiment data (destructive!)
	rm -rf experiments/ mlruns/

# --- Quality ---
lint:                             ## Type check and lint
	python -m py_compile main.py
	python -m pytest tests/ --co -q

help:                             ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
