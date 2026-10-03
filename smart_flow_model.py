"""Learn leak patterns, then classify new pipeline sensor readings.

START HERE (Python 3.10+, no extra packages):
    python smart_flow_model.py train       Train and save a model.
    python smart_flow_model.py demo        Show an explained example.
    python smart_flow_model.py self-test   Check the calculations.

PIPELINE: Node 1 (Q1) -- Segment 1 -- Node 2 (Q2) -- Segment 2 -- Node 3 (Q3)
All flow readings are in litres per minute (L/min).

READ THIS FILE IN THIS ORDER:
    Config                 Timing, sensor noise and operating flow settings.
    History                Calculate four features from recent flow readings.
    generate/read_dataset  Create/read labelled training examples.
    KNN                    Compare new features with stored examples.
    metrics/train/load     Evaluate, save and reload the model.
    Detector               Join hardware readings and produce predictions.
    main                   Choose which command to run.

Training needs known labels (normal/minor/major). Prediction needs only the
sensor readings and saved model. Synthetic accuracy is not hardware accuracy.
For detailed input/output examples, see MODEL_README.md.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import statistics as stats
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

# Tuple positions are class IDs: 0 normal, 1 minor, 2 major.
LABELS = ('normal', 'minor', 'major')
FEATURES = ('delta_q', 'delta_q_roc', 'rolling_variance', 'cumulative_mismatch')
NODES = ('node_1', 'node_2', 'node_3')
VERSION = 'flow-knn-2'

@dataclass
class Config:
    """Defaults; change them with model_config.json and the --config option."""
    window: int = 5             # Complete cycles required for a feature window.
    interval_s: float = 20.0    # Expected seconds between measurement cycles.
    tolerance_s: float = 2.0    # Allowed timestamp difference between nodes.
    stale_s: float = 45.0       # Older live readings are rejected.
    min_inlet_lpm: float = 8.0
    max_inlet_lpm: float = 12.0
    noise_fraction: float = 0.02  # Simulated random noise: 2% standard deviation.
    bias_fraction: float = 0.005  # Fixed simulated sensor offset up to +/-0.5%.
    min_valid_flow: float = 0.5
    max_valid_flow: float = 15.0
    seed: int = 42              # Repeatable random generation and data splits.

    def validate(self):
        if self.window < 2 or not 0 < self.tolerance_s < self.interval_s < self.stale_s:
            raise ValueError('Require window >=2 and 0 < tolerance < interval < stale.')
        if not 0 < self.min_valid_flow < self.min_inlet_lpm <= self.max_inlet_lpm < self.max_valid_flow:
            raise ValueError('Require 0 < valid minimum < inlet minimum <= inlet maximum < valid maximum.')
        if not 0 <= self.noise_fraction < 0.2 or not 0 <= self.bias_fraction < 0.1:
            raise ValueError('Invalid noise or calibration-bias assumption.')


def stamp(value):
    """Convert a timezone-aware date string or Unix timestamp to seconds."""
    if isinstance(value, bool):
        raise ValueError('Boolean timestamp is invalid')
    if isinstance(value, (float, int)):
        result = float(value)
    else:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        if dt.tzinfo is None:
            raise ValueError('ISO timestamps must include a timezone')
        result = dt.timestamp()
    if not math.isfinite(result):
        raise ValueError('Timestamp must be finite')
    return result


def iso(value):
    """Convert numeric seconds to a readable UTC timestamp."""
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


class History:
    """Remember one segment's recent differences and compute four features.

    delta = upstream flow - downstream flow (for example, 10 - 9 = 1 L/min).
    The deque automatically drops old samples when the window is full.
    Training and live prediction use these exact same formulas.
    """
    def __init__(self, config):
        self.cfg = config
        self.samples = deque(maxlen=config.window)

    def update(self, delta, timestamp):
        """Return [difference, change rate, variance, lost volume], or None to wait."""
        if self.samples:
            elapsed = timestamp - self.samples[-1][0]
            if elapsed <= 0:
                raise ValueError('Cycle timestamps must strictly increase')
            if elapsed > self.cfg.stale_s:
                self.samples.clear()
        self.samples.append((timestamp, delta))
        # Warmup is a lack of history, not a leak or a sensor fault.
        if len(self.samples) < self.cfg.window:
            return None
        pairs = list(self.samples)
        deltas = [p[1] for p in pairs]
        # Change in mismatch divided by elapsed seconds.
        roc = (deltas[-1] - deltas[-2]) / (pairs[-1][0] - pairs[-2][0])
        # Integrate over N-1 intervals; divide seconds by 60 to get litres.
        volume = sum(b[1] * (b[0] - a[0]) / 60 for a, b in zip(pairs, pairs[1:]))
        return [delta, roc, stats.pvariance(deltas), volume]


def generate(path, cfg, runs=90, cycles=36):
    """Write simulated sensor readings with known labels to a CSV.

    A run is an independent experiment; each cycle contains three node readings.
    True downstream flow = upstream flow minus injected leakage. Noise is added
    afterwards, so labels come from the experiment, not noisy sensor thresholds.
    This is a simplified mass-balance simulator, not a calibrated rig model.
    """
    if runs < 18 or cycles < cfg.window + 6:
        raise ValueError('Use at least 18 runs and window+6 cycles')
    rng = random.Random(cfg.seed)
    fields = ['scenario_id', 'cycle_id', 'timestamp', 'node_id', 'flow_lpm',
              'device_status', 'segment_1_label', 'segment_2_label',
              'true_loss_1_percent', 'true_loss_2_percent']
    with Path(path).open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for run in range(runs):
            states = (LABELS[(run // 3) % 3], LABELS[run % 3])
            # Each run has its own calibration residuals, operating point and pump changes.
            biases = [rng.uniform(-cfg.bias_fraction, cfg.bias_fraction) for _ in NODES]
            losses = [0 if s == 'normal' else rng.uniform(.05, .15) if s == 'minor'
                      else rng.uniform(.25, .50) for s in states]
            base = rng.uniform(cfg.min_inlet_lpm, cfg.max_inlet_lpm)
            timestamp = 1700000000 + run * cycles * cfg.interval_s * 2
            for cycle in range(cycles):
                timestamp += cfg.interval_s * rng.uniform(.98, 1.02)
                if cycle in (cycles // 3, 2 * cycles // 3):
                    base = rng.uniform(cfg.min_inlet_lpm, cfg.max_inlet_lpm)
                q1 = min(cfg.max_inlet_lpm, max(cfg.min_inlet_lpm, base + rng.gauss(0, .06)))
                # Normal startup, controlled opening, then recovery: temporal transitions included.
                active = cfg.window <= cycle < cycles - cfg.window
                f1, f2 = losses if active else (0., 0.)
                actual = (q1, q1 * (1-f1), q1 * (1-f1) * (1-f2))
                labels = states if active else ('normal', 'normal')
                for idx, node in enumerate(NODES):
                    measured = max(0., actual[idx] * (1 + biases[idx] + rng.gauss(0, cfg.noise_fraction)))
                    writer.writerow(dict(scenario_id=f'run_{run:04}', cycle_id=cycle,
                        timestamp=iso(timestamp + rng.uniform(-.2, .2)), node_id=node,
                        flow_lpm=round(measured, 6), device_status='OK',
                        segment_1_label=labels[0], segment_2_label=labels[1],
                        true_loss_1_percent=100*f1, true_loss_2_percent=100*f2))


def read_dataset(path, cfg):
    """Join CSV rows into cycles, validate them, and calculate labelled features.

    Return {run_name: [(four_feature_values, class_id), ...]} so that whole
    experiments can be kept separate when dividing training and test data.
    """
    grouped = {}
    with Path(path).open(newline='', encoding='utf-8-sig') as src:
        for row in csv.DictReader(src):
            key = (row['scenario_id'], row['cycle_id'])
            bucket = grouped.setdefault(key, {})
            node = row['node_id']
            if node not in NODES or node in bucket:
                raise ValueError(f'Unknown or duplicate node in {key}')
            bucket[node] = row
    by_run = {}
    for (run, cycle), rows in grouped.items():
        if set(rows) != set(NODES):
            raise ValueError(f'Missing node: {run}/{cycle}')
        ordered = [rows[n] for n in NODES]
        ts = [stamp(r['timestamp']) for r in ordered]
        q = [float(r['flow_lpm']) for r in ordered]
        if max(ts)-min(ts) > cfg.tolerance_s or any(r.get('device_status', 'OK') != 'OK' for r in ordered):
            raise ValueError(f'Unsynchronized/faulty training cycle: {run}/{cycle}')
        if any(not math.isfinite(v) or not cfg.min_valid_flow <= v <= cfg.max_valid_flow for v in q):
            raise ValueError(f'Invalid training flow: {run}/{cycle}')
        labels = [ordered[0][f'segment_{s}_label'] for s in (1, 2)]
        if any(label not in LABELS for label in labels):
            raise ValueError('Labels must be normal, minor or major')
        if any([r[f'segment_{s}_label'] for s in (1, 2)] != labels for r in ordered):
            raise ValueError('Node labels disagree within a cycle')
        by_run.setdefault(run, []).append((max(ts), q, labels))
    result = {}
    for run, cycles in by_run.items():
        histories = [History(cfg), History(cfg)]
        examples = []
        for timestamp, q, labels in sorted(cycles):
            for s in range(2):
                vector = histories[s].update(q[s]-q[s+1], timestamp)
                if vector is not None:
                    examples.append((vector, LABELS.index(labels[s])))
        if examples:
            result[run] = examples
    if len(result) < 12:
        raise ValueError('At least 12 independent labelled runs with complete windows are required')
    return result


class KNN:
    """K-Nearest Neighbors: vote among the most similar stored examples.

    fit() stores examples and scaling constants. predict() measures similarity
    and lets the nearest k examples vote for normal, minor or major.
    """
    def __init__(self, references, labels, means, scales, k=1):
        self.references, self.labels = references, labels
        self.means, self.scales, self.k = means, scales, k

    @classmethod
    def fit(cls, examples, budget, seed):
        """Learn scaling from training data; retain up to budget examples."""
        # Features have different units. Scaling prevents one from dominating
        # distance simply because its numeric values happen to be larger.
        cols = list(zip(*(x for x, _ in examples)))
        means = [stats.fmean(col) for col in cols]
        scales = [max(stats.pstdev(col), 1e-9) for col in cols]
        rng = random.Random(seed)
        references, labels = [], []
        for label in range(3):
            rows = [x for x, y in examples if y == label]
            if not rows:
                raise ValueError('Each training partition must contain all three classes')
            # Balanced random reference reduction, evaluated as the actual deployed model.
            for row in rng.sample(rows, min(len(rows), budget // 3)):
                references.append([(v-m)/s for v, m, s in zip(row, means, scales)])
                labels.append(label)
        return cls(references, labels, means, scales)

    def neighbors(self, vector):
        """Order stored examples from most to least similar to these features."""
        x = [(v-m)/s for v, m, s in zip(vector, self.means, self.scales)]
        return sorted(range(len(self.labels)), key=lambda i:
            (sum((a-b)**2 for a,b in zip(x, self.references[i])), i))

    def predict(self, vector, k=None):
        """Return (winning class ID, fraction of neighbors voting for it)."""
        ids = self.neighbors(vector)[:k or self.k]
        votes = [sum(self.labels[i] == label for i in ids) for label in range(3)]
        # Stable class-index tie rule; vote fraction is NOT a calibrated probability.
        label = max(range(3), key=lambda c: (votes[c], -c))
        return label, votes[label]/len(ids)


def metrics(truth, predicted):
    """Compare predictions with known answers.

    Precision: of predictions of a class, how many were correct?
    Recall: of actual examples of a class, how many did the model find?
    F1 balances precision and recall; support counts the actual examples.
    Confusion matrix: rows are true classes, columns are predicted classes.
    """
    matrix = [[0]*3 for _ in range(3)]
    for a,b in zip(truth, predicted):
        matrix[a][b] += 1
    per_class = {}
    for c, label in enumerate(LABELS):
        tp = matrix[c][c]
        precision = tp / max(1, sum(row[c] for row in matrix))
        recall = tp / max(1, sum(matrix[c]))
        per_class[label] = dict(precision=precision, recall=recall,
            f1=2*precision*recall/(precision+recall) if precision+recall else 0., support=sum(matrix[c]))
    return dict(accuracy=sum(matrix[c][c] for c in range(3))/max(1,len(truth)),
                macro_f1=stats.fmean(v['f1'] for v in per_class.values()),
                per_class=per_class, confusion_matrix=matrix, class_order=list(LABELS))


def flatten(runs, names):
    """Collect examples only from the named experimental runs."""
    return [example for name in names for example in runs[name]]


def train(data, out, cfg, budget):
    """Choose k on validation runs, evaluate on untouched runs, save the model."""
    runs = read_dataset(data, cfg)
    names = sorted(runs)
    random.Random(cfg.seed).shuffle(names)
    # Reserve whole runs for testing; never use these runs to select k.
    test_names, development = names[:max(3, len(names)//5)], names[max(3, len(names)//5):]
    candidates = (1, 3, 5, 7, 9)
    scores = {k: [] for k in candidates}
    folds = []
    for fold in range(3):
        # Rotate validation runs. Refit scaling without those validation runs.
        val_names = development[fold::3]
        fit_names = [n for n in development if n not in val_names]
        model = KNN.fit(flatten(runs, fit_names), budget, cfg.seed+fold)
        truth, predictions = [], {k: [] for k in candidates}
        for x,y in flatten(runs, val_names):
            neighbors = model.neighbors(x)
            truth.append(y)
            for k in candidates:
                votes = [sum(model.labels[i] == c for i in neighbors[:k]) for c in range(3)]
                predictions[k].append(max(range(3), key=lambda c:(votes[c],-c)))
        for k in candidates:
            scores[k].append(metrics(truth, predictions[k])['macro_f1'])
        folds.append(dict(training_runs=fit_names, validation_runs=val_names))
        print(f'Validation fold {fold+1}/3 completed', flush=True)
    # Best average F1 wins; a tie favors the smaller k.
    selected = max(candidates, key=lambda k:(stats.fmean(scores[k]), -k))
    model = KNN.fit(flatten(runs, development), budget, cfg.seed)
    model.k = selected
    test = flatten(runs, test_names)
    start = time.perf_counter()
    predictions = [model.predict(x)[0] for x,y in test]
    latency = (time.perf_counter()-start)*1000/len(test)
    report = metrics([y for x,y in test], predictions)
    report.update(selected_k=selected, validation_macro_f1=scores,
        development_runs=development, test_runs=test_names, folds=folds,
        test_samples=len(test), reference_count=len(model.labels),
        estimated_reference_float32_bytes=len(model.labels)*17+32,
        python_ms_per_segment=latency, source_csv=str(Path(data).resolve()),
        limitation='Held-out run results only; synthetic accuracy is not physical hardware accuracy. '
        'Four absolute-flow features cannot uniquely identify percentage severity over arbitrary flow ranges.')
    artifact = dict(version=VERSION, created_at=iso(time.time()), config=asdict(cfg),
        feature_names=list(FEATURES), classes=list(LABELS),
        cumulative_definition='N samples, N-1 right-endpoint intervals, litres',
        transition_policy='Synthetic 15-25 percent loss excluded; inference flags this measured range',
        model=vars(model))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out/'model.json').write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding='utf-8')
    (out/'evaluation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print_training_report(report)
    print(f'Model: {out / "model.json"}\nReport: {out / "evaluation.json"}')


def load(path):
    """Reload stored examples together with their original scaler and settings."""
    artifact = json.loads(Path(path).read_text(encoding='utf-8'))
    if artifact['version'] != VERSION or artifact['feature_names'] != list(FEATURES) or artifact['classes'] != list(LABELS):
        raise ValueError('Incompatible model format')
    cfg = Config(**artifact['config'])
    cfg.validate()
    model = KNN(**artifact['model'])
    if not 1 <= model.k <= len(model.labels) or len(model.references) != len(model.labels):
        raise ValueError('Invalid KNN artifact')
    if len(model.means) != 4 or len(model.scales) != 4 or any(s <= 0 for s in model.scales):
        raise ValueError('Invalid scaler')
    if any(len(r) != 4 for r in model.references) or any(y not in (0,1,2) for y in model.labels):
        raise ValueError('Invalid reference set')
    if any(not math.isfinite(v) for r in [model.means, model.scales, *model.references] for v in r):
        raise ValueError('Nonfinite model values')
    return cfg, model


class Detector:
    """Join hardware messages and classify each complete measurement cycle.

    Keep this object alive between messages so it remembers previous cycles.
    pending: readings waiting for the other nodes; seen: last node timestamps;
    history: one rolling feature calculator per segment. Readings are used once.
    """
    def __init__(self, cfg, model):
        self.cfg, self.model = cfg, model
        self.pending, self.seen = {}, {}
        self.history = [History(cfg), History(cfg)]

    def unavailable(self, reason):
        """Explain missing/invalid data without claiming the pipeline is Normal."""
        return dict(model_version=VERSION, data_quality=reason,
            overall_status='INFERENCE_UNAVAILABLE', segment_1_state=None, segment_2_state=None)

    def ingest(self, payload, now=None):
        """Accept one node message; return a prediction or a waiting/error reason.

        Expected keys: node_id, timestamp, flow_lpm; device_status is optional.
        now uses the real clock for hardware and a simulated clock for replay.
        """
        now = time.time() if now is None else now
        try:
            node = payload['node_id']
            if node not in NODES:
                return self.unavailable('UNKNOWN_NODE')
            timestamp = stamp(payload['timestamp'])
            if isinstance(payload['flow_lpm'], bool):
                raise ValueError('Invalid flow')
            flow = float(payload['flow_lpm'])
            if not math.isfinite(flow) or not self.cfg.min_valid_flow <= flow <= self.cfg.max_valid_flow or payload.get('device_status','OK') != 'OK':
                self.pending.pop(node, None)
                self.history = [History(self.cfg), History(self.cfg)]
                return self.unavailable('SENSOR_FAULT_OR_FLOW_OUT_OF_RANGE')
            if now-timestamp > self.cfg.stale_s or timestamp-now > self.cfg.tolerance_s:
                return self.unavailable('STALE_OR_FUTURE_DATA')
            if timestamp <= self.seen.get(node, -math.inf):
                return self.unavailable('DUPLICATE_OR_OUT_OF_ORDER')
            self.seen[node] = timestamp
            self.pending[node] = (timestamp, flow)
        except (KeyError, TypeError, ValueError, OverflowError):
            return self.unavailable('INVALID_PAYLOAD')
        self.pending = {n:v for n,v in self.pending.items() if now-v[0] <= self.cfg.stale_s}
        if len(self.pending) != 3:
            return self.unavailable('MISSING_NODE_DATA')
        ts = [self.pending[n][0] for n in NODES]
        if max(ts)-min(ts) > self.cfg.tolerance_s:
            oldest = min(self.pending, key=lambda n:self.pending[n][0])
            del self.pending[oldest]
            return self.unavailable('UNSYNCHRONIZED_DATA')
        q = [self.pending[n][1] for n in NODES]
        # All three readings now belong to one cycle; consume them once.
        self.pending.clear()
        vectors = [self.history[s].update(q[s]-q[s+1], max(ts)) for s in range(2)]
        if any(x is None for x in vectors):
            return self.unavailable('WARMING_UP')
        states, details = [], {}
        warnings = []
        if not self.cfg.min_inlet_lpm <= q[0] <= self.cfg.max_inlet_lpm:
            warnings.append('INLET_OUTSIDE_TRAINING_RANGE')
        # Apply the SAME learned classifier independently to each segment.
        for s,x in enumerate(vectors):
            label, vote = self.model.predict(x)
            states.append(LABELS[label])
            loss = 100*(q[s]-q[s+1])/q[s]
            if 15 < loss < 25:
                warnings.append(f'SEGMENT_{s+1}_UNRESOLVED_SEVERITY_RANGE')
            details[f'segment_{s+1}'] = dict(class_id=label, state=LABELS[label],
                neighbor_vote_fraction=vote, flow_loss_percent=loss,
                features=dict(zip(FEATURES,x)))
        # Critical combines two leak predictions; it is not a fourth KNN class.
        affected = [f'segment_{i+1}' for i,s in enumerate(states) if s != 'normal']
        status = 'CRITICAL' if len(affected)==2 else 'MAJOR LEAK' if 'major' in states else 'MINOR LEAK' if 'minor' in states else 'NORMAL'
        return dict(timestamp=iso(max(ts)), q1=q[0], q2=q[1], q3=q[2],
            segment_1_state=states[0], segment_2_state=states[1], overall_status=status,
            affected_segments=affected, segments=details, warnings=warnings,
            data_quality='REVIEW_REQUIRED' if warnings else 'OK', model_version=VERSION)


def self_test():
    """Check known calculations and data handling; not a hardware accuracy test."""
    cfg = Config(window=3)
    h = History(cfg)
    assert h.update(1,0) is None
    assert h.update(2,20) is None
    x = h.update(3,40)
    assert all(math.isclose(a,b) for a,b in zip(x,[3,.05,2/3,5/3]))
    assert math.isclose(h.update(4,60)[3],7/3)  # rolling integral, not lifetime drift
    assert h.update(1,120) is None  # gap resets history
    model = KNN.fit([([0,0,0,0],0),([1,0,0,1],1),([4,0,0,4],2)]*4, 12, 2)
    for x,y in [([0,0,0,0],0),([1,0,0,1],1),([4,0,0,4],2)]:
        assert model.predict(x)[0] == y
    detector = Detector(cfg,model)
    for cycle in range(3):
        now = 1000+20*cycle
        for n in NODES:
            result = detector.ingest(dict(node_id=n,timestamp=now,flow_lpm=10),now)
    assert result['overall_status']=='NORMAL'
    assert detector.ingest(dict(node_id='node_1',timestamp=1040,flow_lpm=10),1040)['data_quality']=='DUPLICATE_OR_OUT_OF_ORDER'
    assert detector.ingest(dict(node_id='node_1',timestamp=1060,flow_lpm=float('nan')),1060)['segment_1_state'] is None
    assert detector.ingest(dict(node_id='node_1',timestamp=1000,flow_lpm=10),1100)['data_quality']=='STALE_OR_FUTURE_DATA'
    print('PASS: feature units/window, gap reset, scaling/KNN, synchronization, warmup, duplicate/stale/NaN rejection')


def print_training_report(report):
    """Display percentages with explanations; the full report stays in JSON."""
    print('\nTRAINING COMPLETE')
    print(f"Selected K: {report['selected_k']} (number of nearby examples that vote)")
    print(f"Stored reference examples: {report['reference_count']}")
    print(f"Test accuracy: {report['accuracy']:.2%} (correct segment predictions)")
    print(f"Macro F1: {report['macro_f1']:.2%} (equal-weight average across classes)")
    print('\nClass       Precision    Recall        F1    Test examples')
    for label, values in report['per_class'].items():
        print(f"{label:<10} {values['precision']:>9.2%} {values['recall']:>9.2%} "
              f"{values['f1']:>9.2%} {values['support']:>12}")
    print('Precision: how often a predicted class was correct.')
    print('Recall: how many actual examples of a class were found.')
    print('\nConfusion matrix: rows = actual; columns = predicted')
    print('Actual       Normal    Minor    Major')
    for label, counts in zip(LABELS, report['confusion_matrix']):
        print(f'{label:<10} {counts[0]:>8} {counts[1]:>8} {counts[2]:>8}')
    print(f"\nAverage Python prediction time: {report['python_ms_per_segment']:.3f} ms per segment")
    print('These results describe the test dataset, not verified hardware accuracy.')


def print_demo_result(cycle, result, cfg, flows=(10., 9., 8.95)):
    """Explain one simulated cycle without requiring the reader to parse JSON."""
    print(f'\nCycle {cycle}: Q1={flows[0]:.2f}, Q2={flows[1]:.2f}, Q3={flows[2]:.2f} L/min')
    if result['data_quality'] == 'WARMING_UP':
        print(f'  Collecting history: {cycle}/{cfg.window} complete cycles.')
        print('  No prediction yet; this is expected during startup.')
        return
    if result['overall_status'] == 'INFERENCE_UNAVAILABLE':
        print(f"  Cannot predict: {result['data_quality']}")
        return
    for number in (1, 2):
        segment = result['segments'][f'segment_{number}']
        features = segment['features']
        print(f"  Segment {number}: {segment['state'].upper()} | "
              f"measured flow loss: {segment['flow_loss_percent']:.2f}%")
        print(f"    Flow difference: {features['delta_q']:.3f} L/min; "
              f"volume mismatch over window: {features['cumulative_mismatch']:.3f} L")
    print(f"  Overall: {result['overall_status']} | Data quality: {result['data_quality']}")
    for warning in result['warnings']:
        print(f'  Review: {warning}')


def interactive(cfg, model, sequence=False):
    """Try typed flows using a simulated clock, without modifying the model.

    Default: each entry is an independent, constant-flow test over a full window.
    --sequence: each entry adds one cycle to a continuing time series instead.
    Neither mode supplies real measurement timing; use predict for hardware.
    """
    print('MANUAL MODEL TEST - enter Q1 Q2 Q3 in L/min, for example: 10 9 8.95')
    print('Commas are also accepted. Type quit to finish or reset to clear history.')
    if sequence:
        print(f'Each entry adds one simulated {cfg.interval_s:g}-second cycle; '
              f'predictions start after {cfg.window} entries.')
    else:
        print(f'Each entry assumes those flows stayed constant for {cfg.window} cycles '
              f'({(cfg.window-1)*cfg.interval_s:g} seconds).')
        print('This gives an immediate steady-flow test, not a measured history.')
    detector = Detector(cfg, model)
    cycle = 0
    start = time.time()
    while True:
        try:
            line = input('\nQ1 Q2 Q3 > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nManual test ended.')
            return
        if line.lower() in ('quit', 'exit', 'q'):
            return
        if line.lower() == 'reset':
            detector, cycle = Detector(cfg, model), 0
            print('History cleared.')
            continue
        try:
            flows = [float(value) for value in line.replace(',', ' ').split()]
            if len(flows) != 3 or any(not math.isfinite(q) or not cfg.min_valid_flow <= q <= cfg.max_valid_flow for q in flows):
                raise ValueError
        except ValueError:
            print(f'Enter exactly three finite numbers between {cfg.min_valid_flow:g} '
                  f'and {cfg.max_valid_flow:g} L/min. Example: 10 9 8.95')
            continue
        if not sequence:
            detector, cycle = Detector(cfg, model), 0
        for _ in range(1 if sequence else cfg.window):
            timestamp = start + cycle * cfg.interval_s
            for node, flow in zip(NODES, flows):
                result = detector.ingest(dict(node_id=node, timestamp=timestamp,
                                             flow_lpm=flow), now=timestamp)
            cycle += 1
        print_demo_result(cycle, result, cfg, flows)


def main():
    """Read the chosen command and run generation, training, testing or prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('generate','train'):
        p = sub.add_parser(name)
        p.add_argument('--data', default='flow_training.csv')
        p.add_argument('--config', help='JSON object of Config overrides')
        p.add_argument('--runs', type=int, default=90)
        p.add_argument('--cycles', type=int, default=36)
        if name=='train':
            p.add_argument('--out', default='flow_artifacts')
            p.add_argument('--references', type=int, default=180)
    for name in ('predict','demo','interactive'):
        p = sub.add_parser(name)
        p.add_argument('--model',default='flow_artifacts/model.json')
        if name=='interactive':
            p.add_argument('--sequence', action='store_true',
                           help='enter successive cycles instead of independent constant-flow tests')
        if name=='demo':
            p.add_argument('--json', action='store_true', help='show full machine-readable demo output')
        if name=='predict':
            p.add_argument('--input',default='-',help='JSONL file, or - for stdin')
            p.add_argument('--replay',action='store_true',help='Historical offline replay; disables wall-clock age comparison')
    sub.add_parser('self-test')
    # VS Code's Run Python File button supplies no command-line arguments.
    # In that case, show the saved-model demo instead of an argparse error.
    if len(sys.argv) == 1:
        print('No command supplied: running the explained demo.\n')
        default_model = Path(__file__).resolve().parent / 'flow_artifacts' / 'model.json'
        args = parser.parse_args(['demo', '--model', str(default_model)])
    else:
        args = parser.parse_args()
    if args.command in ('generate','train'):
        cfg = Config(**(json.loads(Path(args.config).read_text()) if args.config else {}))
        cfg.validate()
        if args.command=='generate' or not Path(args.data).exists():
            generate(args.data,cfg,args.runs,args.cycles)
            print(f'Synthetic raw telemetry: {args.data}',flush=True)
        if args.command=='train':
            if args.references < 9:
                raise ValueError('At least 9 references required')
            train(args.data,args.out,cfg,args.references)
    elif args.command=='self-test':
        self_test()
    else:
        cfg, model = load(args.model)
        detector = Detector(cfg,model)
        if args.command=='interactive':
            interactive(cfg, model, args.sequence)
        elif args.command=='demo':
            if not args.json:
                print('SIMULATED INPUT DEMO (not live hardware)')
                print(f'Model loaded: K={model.k}, {len(model.labels)} stored examples.')
                print(f'One cycle contains all three sensor readings; history needs {cfg.window} cycles.')
                print(f'Simulated spacing: {cfg.interval_s:g} seconds. The demo runs without waiting.')
                print('Segment 1 is Node 1 to Node 2; Segment 2 is Node 2 to Node 3.')
            now = time.time()
            for cycle in range(cfg.window+2):
                for n,q in zip(NODES,(10.,9.,8.95)):
                    payload = dict(node_id=n,timestamp=iso(now+cycle*cfg.interval_s),flow_lpm=q,device_status='OK')
                    result = detector.ingest(payload,now+cycle*cfg.interval_s)
                if args.json:
                    print(json.dumps(dict(cycle=cycle+1,output=result),allow_nan=False))
                else:
                    print_demo_result(cycle+1, result, cfg)
        else:
            src = sys.stdin if args.input=='-' else Path(args.input).open(encoding='utf-8')
            try:
                for line in src:
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                        if not isinstance(payload,dict):
                            raise ValueError('Payload must be an object')
                        now = stamp(payload['timestamp']) if args.replay else None
                        result = detector.ingest(payload,now)
                    except (ValueError,KeyError,TypeError):
                        result = detector.unavailable('INVALID_PAYLOAD')
                    print(json.dumps(result,allow_nan=False),flush=True)
            finally:
                if src is not sys.stdin:
                    src.close()

if __name__=='__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        sys.exit(1)
