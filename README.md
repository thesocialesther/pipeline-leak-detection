# Pipeline leak detection and localization

**Author: Oluwaferanmi Esther Onifade**

A flow-based monitoring project that detects leaks and identifies which of two pipeline segments is affected. Three synchronized flow measurements—inlet, midpoint, and outlet—allow the system to compare water entering and leaving each segment.

## How it works

A K-nearest-neighbours (KNN) classifier uses four features per segment: flow mismatch, its rate of change, rolling variance, and accumulated mismatch volume. A five-cycle window supplies the recent history. Each segment is classified as normal, minor leak, or major leak; simultaneous leaks produce a critical system state. Missing, stale, faulty, or insufficient readings are reported as unavailable rather than a confident classification.

The Python implementation provides training, replay, interactive prediction, and simulation. A generated C++11 model supports embedded inference without dynamic allocation. A Supabase worker provides a separate telemetry integration path.

## Repository contents

- `smart_flow_model.py`: model, feature extraction, training, and command-line interface.
- `flow_artifacts/`: exported model configuration and evaluation results.
- `embedded/`: inference header, integration example, and parity checks.
- `supabase_worker.py` and its tests: backend adapter.
- [MODEL_README.md](MODEL_README.md): feature definitions and model usage.
- [embedded/README.md](embedded/README.md): embedded interface.
- [SUPABASE_SETUP.md](SUPABASE_SETUP.md): backend configuration.
- [DATASETS.md](DATASETS.md): dataset provenance and access information.

## Run locally

Use Python 3.10 or newer. The Python model uses the standard library.

~~~sh
python smart_flow_model.py self-test
python smart_flow_model.py demo
python smart_flow_model.py interactive
python -m unittest test_supabase_worker -v
~~~

To rebuild training artifacts, run `python smart_flow_model.py train`. This modifies generated model files. Follow the embedded guide to compile or integrate the C++ implementation. Configure the Supabase adapter separately with your backend schema and environment credentials.

## Evaluation and current scope

The supplied synthetic dataset contains 90 runs, 36 cycles per run, and three nodes: 9,720 rows. Whole runs were separated for evaluation, with grouped cross-validation for model selection. The saved evaluation reports **92.10% accuracy and 92.49% macro F1** across 1,152 segment samples, using K=3.

These are synthetic-data results. Physical pipeline performance still requires calibrated sensor readings and rig testing. The embedded example demonstrates the inference interface; complete sensor firmware, communications, and dashboard source are not yet included.
