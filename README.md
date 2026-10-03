# Pipeline leak detection
Author: Oluwaferanmi Esther Onifade

Flow-based KNN model for two pipeline segments using three synchronized sensor nodes. Includes a Python simulator and inference CLI, generated embedded C++ model, and an offline-tested Supabase adapter.

## Run
Python 3.10 or newer; the Python model uses the standard library.
~~~sh
python smart_flow_model.py self-test
python smart_flow_model.py train
python smart_flow_model.py demo
python -m unittest test_supabase_worker -v
~~~
Training generates synthetic data when no training CSV is present. These results are simulation results, not proof of accuracy on physical pipes. The Supabase adapter needs your own backend environment credentials and confirmed telemetry schema.

See [MODEL_README.md](MODEL_README.md), [embedded/README.md](embedded/README.md), and [SUPABASE_SETUP.md](SUPABASE_SETUP.md) for the feature contract, hardware interface, and limitations. Never commit backend keys.

## Dataset access
See [DATASETS.md](DATASETS.md) for included data, exact file manifests, and setup instructions.
