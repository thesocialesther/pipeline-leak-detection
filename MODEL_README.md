# Flow-only pipeline model

Complete standalone implementation: `smart_flow_model.py`. Python 3.10+; no pip packages needed. The existing dashboard prototype in `smart_flow.py` is separate and does not automatically load this new model.

## VS Code Run button and reading the output

You can now open `smart_flow_model.py` and click **Run Python File**. With no command, it loads the saved model next to the script and runs a readable simulated demo. It does not retrain automatically. If no model is saved yet, run `python smart_flow_model.py train` in the project terminal first.

`demo` now prints plain-language results. Use `python smart_flow_model.py demo --json` to see the original full JSON output. The `predict` command still emits JSONL for hardware integration.

The earlier `arguments are required: command` error meant no action had been selected, not that training failed. Running without arguments now selects the demo.

- **PASS** means the calculation and data-handling checks passed; it is not an accuracy score.
- **Validation fold 1/3 ... 3/3** means the script is comparing K values on three separate validation partitions.
- **Selected K = 3** means the three nearest stored examples vote on each prediction.
- **Accuracy = 92.10%** means 1,061 of 1,152 held-out segment examples were classified correctly in this synthetic dataset.
- **WARMING_UP** means collecting the first five synchronized cycles. `null` means no prediction yet, not Normal or a leak. At 20-second spacing, five samples span 80 seconds. The demo advances a simulated clock instantly.
- **Segment 1 minor / Segment 2 normal** is the result for Q1=10, Q2=9, Q3=8.95 L/min. The measured losses are 10% and about 0.56%, respectively.
- **neighbor_vote_fraction = 1.0** means all selected neighbors agreed; it does not mean the prediction is guaranteed correct.
- **delta_q_roc = 0 and rolling_variance = 0** occur in this demo because the readings stay constant each cycle.
- **cumulative_mismatch = 1.333 L** for Segment 1 is 1 L/min multiplied by the 80-second window divided by 60. It is a rolling volume mismatch, not total leakage since startup.

For the initial confusion matrix, the row `[475, 51, 2]` means 528 truly normal examples were predicted as Normal 475 times, Minor 51 times, and Major twice. The other rows follow the same actual-versus-predicted convention.

## Run

```powershell
python -B smart_flow_model.py self-test
python -B smart_flow_model.py train
python -B smart_flow_model.py demo
python -B smart_flow_model.py predict --input hardware_example.jsonl --replay
```

`train` generates `flow_training.csv` if it does not exist, then trains from that file. It saves `flow_artifacts/model.json` and `flow_artifacts/evaluation.json`. Repeated training reuses the CSV and replaces these artifacts. `generate` explicitly replaces the named CSV.

To generate a different simulation and retrain:

```powershell
python -B smart_flow_model.py generate --config model_config.json --data custom_training.csv --runs 90 --cycles 36
python -B smart_flow_model.py train --config model_config.json --data custom_training.csv --out custom_artifacts
```

Use the SAME configuration for generation and training. Inference loads configuration from the saved artifact. Keep the model and evaluation report together when versioning experiments.

## Type your own flow readings

Run `python smart_flow_model.py interactive` in the VS Code terminal.
At `Q1 Q2 Q3 >`, enter three flows in L/min, for example `10 9 8.95`,
then press Enter. The script prints both segment predictions and overall status.
Enter another set to test again; type `quit` to finish. Commas also work.

Each entry is an independent test that assumes the supplied flows stayed constant
for five simulated cycles (an 80-second window at the default cadence). This
explicit assumption provides the history needed by the model; it is not a
prediction based on measured temporal behavior. It does not retrain the model.

To enter a changing sequence instead, use:

```powershell
python smart_flow_model.py interactive --sequence
```

Each entry then adds one simulated 20-second cycle. Enter at least five triples
for the first prediction. Type `reset` to start a new history. For actual hardware
timestamps, continue to use the JSONL `predict` command.

## Hardware input

Each ESP32 supplies one JSON object per line. Convert calibrated sensor pulse counts to **L/min in firmware**, using the measured calibration factor for that individual sensor and the acquisition duration. This script consumes flow, not raw pulses. Use measurement timestamps from synchronized clocks and matching acquisition windows, rather than server arrival timestamps. GPS is optional metadata and is not a model feature.

```json
{"node_id":"node_1","timestamp":"2026-09-16T12:00:00+01:00","flow_lpm":10.0,"device_status":"OK"}
{"node_id":"node_2","timestamp":"2026-09-16T12:00:00+01:00","flow_lpm":9.0,"device_status":"OK"}
{"node_id":"node_3","timestamp":"2026-09-16T12:00:00+01:00","flow_lpm":8.95,"device_status":"OK"}
```

Node 1 is the inlet; Node 2 the midpoint; Node 3 the outlet. Each segment is 3 m in the specification. Repeat all three readings each cycle with new timestamps. Numeric Unix seconds are also accepted. ISO timestamps must include a timezone. `device_status` defaults to `OK`; other values suppress inference.

Live line-by-line input:

```powershell
python -B smart_flow_model.py predict
```

Paste readings, or pipe JSONL from your telemetry adapter. In Python, connect your Supabase/serial/MQTT receiver to the same stateful detector:

```python
from smart_flow_model import load, Detector
config, model = load('flow_artifacts/model.json')
detector = Detector(config, model)
# Keep this detector alive across messages; do not recreate it for every request.
result = detector.ingest({
    'node_id': 'node_1',
    'timestamp': '2026-09-16T12:00:00+01:00',  # replace with actual measurement time
    'flow_lpm': 10.0,
    'device_status': 'OK',
})
```

Call `ingest` serially (or protect it with a lock). Cloud ingestion/network drivers and ESP32 firmware are not part of this model script. Run it on the host that receives all three nodes. A single ESP32 reading alone cannot localize a segment leak.

`--replay` is ONLY for historical offline data: it uses each message timestamp as the replay clock. Omit it for live hardware so stale timestamps are rejected. `demo` also uses a simulated clock so it can run immediately.

## Output

One JSON object is emitted for every message. The first two messages of a cycle normally return `MISSING_NODE_DATA`. Predictions become available after five complete synchronized cycles (about 80 seconds between the first and fifth samples). Example abbreviated result for the flows above:

```json
{
  "q1": 10.0,
  "q2": 9.0,
  "q3": 8.95,
  "segment_1_state": "minor",
  "segment_2_state": "normal",
  "overall_status": "MINOR LEAK",
  "affected_segments": ["segment_1"],
  "data_quality": "OK",
  "model_version": "flow-knn-2"
}
```

The full output also includes the timestamp, each segment's four features, measured flow-loss percentage, class ID, neighbor vote fraction, and warnings. IDs: 0 normal, 1 minor, 2 major. Both segments leaking produce `CRITICAL`. The neighbor vote fraction is NOT a calibrated probability.

Unavailable outputs have null segment states, an `INFERENCE_UNAVAILABLE` overall status, and a reason in `data_quality`. Reasons cover warmup, missing nodes, timestamp skew, stale/future timestamps, duplicate/out-of-order samples, malformed payloads and sensor faults/out-of-range flows. Readings are consumed once per inference cycle. A history gap exceeding 45 seconds requires a fresh warmup. When all messages stop, the CLI emits nothing: the receiving application must time out its last result; never retain a displayed Normal state indefinitely.

## Feature contract

For each segment, upstream flow minus downstream flow produces delta_q. Input vector order:

1. `delta_q`: L/min, signed.
2. `delta_q_roc`: change in delta_q divided by elapsed seconds.
3. `rolling_variance`: population variance over the last N deltas, (L/min)^2.
4. `cumulative_mismatch`: integral across the N-1 intervals between those N samples; each interval uses its ending delta times seconds / 60, yielding litres.

The default N=5 creates an approximately 80-second integration horizon at a 20-second cadence. This explicit bounded horizon prevents lifetime cumulative drift. Training and runtime share this exact implementation. The scaler is fitted exclusively on each training partition. Scaling constants, reference vectors, labels, k and configuration are saved together.

## Synthetic data and evaluation

Defaults: 90 independent runs, 36 cycles per run, 9,720 node rows. Inlet range 8-12 L/min; 20-second cadence; 2% Gaussian relative reading noise; independent per-run sensor residual biases within +/-0.5%; small timestamp jitter; pump level changes; leak opening and recovery. Conservation is imposed on true flows: Q2=Q1*(1-loss1), Q3=Q2*(1-loss2). Sensor noise is then applied independently. This is a simplified mass-balance simulator, not EPANET or a physically calibrated sensor model. It does not model pipe storage, sensor transport delay, blockages, or pulse quantization.

Ground truth comes from the injected leak, not thresholds applied to noisy readings. Normal true loss is zero; minor is 5-15%; major is 25-50%. The unresolved 15-25% range is excluded from generation. At inference a measured loss in that range adds a warning and `REVIEW_REQUIRED`; a KNN label is still supplied provisionally. Inlet readings outside the configured training range also require review. Zero/very low flow is outside this model's operating envelope and suppresses inference.

An entire run stays in one partition. 20% of runs are held out for a final test. Three grouped validation folds over the remaining runs select k from 1,3,5,7,9 using macro F1. Each fold fits its own scaler and chooses its own balanced compact reference set using only that fold's training runs. The final 180-reference model is refitted on development runs and tested once on the held-out runs. The saved model is the exact model evaluated; there is no untested post-evaluation refit.

The report contains split run IDs, validation scores, accuracy, per-class precision/recall/F1, confusion matrix (rows=true, columns=predicted), reference count, estimated packed float32 reference storage and measured Python inference latency. Storage estimates exclude Python object overhead and ESP32 firmware/runtime buffers. No embedded firmware or parity claim is made.

Initial fixed-seed result: k=3, accuracy 92.10%, macro F1 92.49%, minor recall 91.76%, major recall 96.92%. These are **synthetic segment-sample metrics**, not hardware validation or event detection guarantees. All warm post-startup windows, including leak onset and recovery transitions, are evaluated; no transition windows are silently removed.

## Training with measured rig data

Use a CSV with these required columns:

```text
scenario_id,cycle_id,timestamp,node_id,flow_lpm,device_status,segment_1_label,segment_2_label
```

There must be three rows per cycle, one per node, with consistent labels on all three rows. `scenario_id` identifies an independent experimental run; never invent separate scenario IDs for neighboring windows from one experiment. Labels are `normal`, `minor`, or `major` for each segment, obtained from controlled leak tests. At least 12 complete independent runs are required; collect enough runs of each class to cover every training fold. Synthetic CSV adds known true loss percentages for audit; these are NOT model features.

```powershell
python -B smart_flow_model.py train --data measured_rig.csv --config model_config.json --out measured_artifacts
```

Calibrate sensors individually; measure no-leak mismatch, noise, actual flow envelope and acquisition cadence; update configuration; collect controlled normal, minor, major, simultaneous leak, onset/recovery and pump-change runs. Test on untouched physical runs. Four absolute-flow features do not uniquely determine percentage severity over arbitrary operating ranges: widening the range can degrade classification, particularly downstream when Segment 1 also leaks. If physical tests require broad flow support, revise and revalidate the specified feature contract to include upstream flow or normalized loss. Synthetic performance alone cannot establish suitability for the working hardware.
