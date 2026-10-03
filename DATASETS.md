# Dataset
flow_training.csv is the original synthetic training dataset saved beside the model source. It contains synchronized three-node flow readings and segment labels. Its generation assumptions and train/test split procedure are documented in MODEL_README.md. It is simulated data, not a physical sensor recording.
Run python smart_flow_model.py train to reproduce training from this CSV. The generated model and evaluation are in flow_artifacts/.
