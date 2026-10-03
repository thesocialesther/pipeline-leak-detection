"""Export the saved KNN model as a standalone C++11/ESP32 header.

Run: python export_flow_header.py
The reference vectors are ALREADY scaled; do not scale them a second time.
Only new feature vectors are scaled inside predict(). No retraining occurs.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
from smart_flow_model import load

ROOT = Path(__file__).resolve().parent


def float32(value):
    return struct.unpack('f', struct.pack('f', value))[0]


def literal(value):
    text = format(float32(value), '.9g')
    if '.' not in text and 'e' not in text:
        text += '.0'
    return text + 'f'


RUNTIME = r'''
// Class IDs match Python: 0 normal, 1 minor, 2 major; -1 means unavailable.
struct Prediction {
  int class_id;
  double vote_fraction;  // Neighbor agreement, NOT calibrated probability.
};

inline const char* className(int id) {
  return id == 0 ? "normal" : id == 1 ? "minor" : id == 2 ? "major" : "unavailable";
}

// Input order: delta_q, delta_q_roc, rolling_variance, cumulative_mismatch.
// Units: L/min, (L/min)/second, (L/min)^2, litres.
inline Prediction predict(const double features[4]) {
  double scaled[4];
  for (int j = 0; j < 4; ++j) {
    if (!std::isfinite(features[j])) return {-1, 0.0};
    scaled[j] = (features[j] - MEANS[j]) / SCALES[j];
    if (!std::isfinite(scaled[j])) return {-1, 0.0};
  }
  double best[K];
  int nearest[K];
  for (int k = 0; k < K; ++k) {
    best[k] = std::numeric_limits<double>::infinity();
    nearest[k] = -1;
  }
  // Scan in original reference order. Strict < retains lower index on ties,
  // matching Python's sorting by (distance, reference index).
  for (int i = 0; i < REFERENCE_COUNT; ++i) {
    double distance = 0.0;
    for (int j = 0; j < 4; ++j) {
      const double difference = scaled[j] - REFERENCES[i][j];
      distance += difference * difference;
    }
    if (!std::isfinite(distance)) return {-1, 0.0};
    for (int k = 0; k < K; ++k) {
      if (distance < best[k]) {
        for (int next = K - 1; next > k; --next) {
          best[next] = best[next - 1]; nearest[next] = nearest[next - 1];
        }
        best[k] = distance; nearest[k] = i;
        break;
      }
    }
  }
  int votes[3] = {0, 0, 0};
  for (int k = 0; k < K; ++k) {
    if (nearest[k] < 0) return {-1, 0.0};
    ++votes[LABELS[nearest[k]]];
  }
  int winner = 0;
  for (int c = 1; c < 3; ++c) if (votes[c] > votes[winner]) winner = c;
  return {winner, static_cast<double>(votes[winner]) / K};
}

// One history per segment. Timestamps are DOUBLE seconds, never float epoch
// seconds (float cannot preserve measurement timing at current epoch values).
class History {
  double times_[WINDOW];
  double deltas_[WINDOW];
  int count_ = 0;
 public:
  void reset() { count_ = 0; }
  bool update(double delta, double timestamp, double out[4]) {
    if (!std::isfinite(delta) || !std::isfinite(timestamp)) { reset(); return false; }
    if (count_ && timestamp <= times_[count_ - 1]) return false;
    if (count_ && timestamp - times_[count_ - 1] > STALE_SECONDS) reset();
    if (count_ == WINDOW) {
      for (int i = 1; i < WINDOW; ++i) {
        times_[i-1] = times_[i]; deltas_[i-1] = deltas_[i];
      }
      --count_;
    }
    times_[count_] = timestamp; deltas_[count_++] = delta;
    if (count_ < WINDOW) return false;
    double mean = 0.0, variance = 0.0, volume = 0.0;
    for (int i = 0; i < WINDOW; ++i) mean += deltas_[i] / WINDOW;
    for (int i = 0; i < WINDOW; ++i) {
      const double difference = deltas_[i] - mean;
      variance += difference * difference / WINDOW;
      if (i) volume += deltas_[i] * (times_[i] - times_[i-1]) / 60.0;
    }
    out[0] = delta;
    out[1] = (deltas_[WINDOW-1] - deltas_[WINDOW-2]) /
             (times_[WINDOW-1] - times_[WINDOW-2]);
    out[2] = variance; out[3] = volume;
    return true;
  }
};

struct CycleResult {
  Prediction segment[2];
  double features[2][4];
  double loss_percent[2];
  const char* overall;
  const char* data_quality;
  bool review_inlet;
  bool review_severity[2];
};

// Pass ONE complete cycle in node order: inlet, midpoint, outlet.
// The firmware must collect all three measurements before calling update().
// This class rejects duplicates rather than reusing readings across cycles.
class Pipeline {
  History history_[2];
  double last_[3] = {0, 0, 0};
  bool has_last_ = false;
 public:
  void reset() { history_[0].reset(); history_[1].reset(); has_last_ = false; }
  CycleResult update(const double q[3], const double timestamps[3], double now,
                     const bool sensor_ok[3]) {
    CycleResult result = {};
    result.segment[0] = {-1, 0.0}; result.segment[1] = {-1, 0.0};
    result.overall = "INFERENCE_UNAVAILABLE";
    result.data_quality = "INVALID_PAYLOAD";
    if (!std::isfinite(now)) return result;
    double earliest = timestamps[0], latest = timestamps[0];
    for (int i = 0; i < 3; ++i) {
      if (!std::isfinite(q[i]) || !std::isfinite(timestamps[i])) { reset(); return result; }
      if (!sensor_ok[i] || q[i] < MIN_VALID_FLOW || q[i] > MAX_VALID_FLOW) {
        history_[0].reset(); history_[1].reset();
        result.data_quality = "SENSOR_FAULT_OR_FLOW_OUT_OF_RANGE"; return result;
      }
      if (now - timestamps[i] > STALE_SECONDS || timestamps[i] - now > TOLERANCE_SECONDS) {
        result.data_quality = "STALE_OR_FUTURE_DATA"; return result;
      }
      if (has_last_ && timestamps[i] <= last_[i]) {
        result.data_quality = "DUPLICATE_OR_OUT_OF_ORDER"; return result;
      }
      if (timestamps[i] < earliest) earliest = timestamps[i];
      if (timestamps[i] > latest) latest = timestamps[i];
    }
    if (latest - earliest > TOLERANCE_SECONDS) {
      result.data_quality = "UNSYNCHRONIZED_DATA"; return result;
    }
    for (int i = 0; i < 3; ++i) last_[i] = timestamps[i];
    has_last_ = true;
    bool ready1 = history_[0].update(q[0]-q[1], latest, result.features[0]);
    bool ready2 = history_[1].update(q[1]-q[2], latest, result.features[1]);
    if (!ready1 || !ready2) { result.data_quality = "WARMING_UP"; return result; }
    result.review_inlet = q[0] < MIN_INLET_FLOW || q[0] > MAX_INLET_FLOW;
    bool review = result.review_inlet;
    for (int s = 0; s < 2; ++s) {
      result.segment[s] = predict(result.features[s]);
      if (result.segment[s].class_id < 0) return result;
      result.loss_percent[s] = 100.0 * (q[s]-q[s+1]) / q[s];
      result.review_severity[s] = result.loss_percent[s] > 15 && result.loss_percent[s] < 25;
      review = review || result.review_severity[s];
    }
    int a = result.segment[0].class_id, b = result.segment[1].class_id;
    result.overall = a > 0 && b > 0 ? "CRITICAL" : a == 2 || b == 2 ? "MAJOR LEAK" :
                     a == 1 || b == 1 ? "MINOR LEAK" : "NORMAL";
    result.data_quality = review ? "REVIEW_REQUIRED" : "OK";
    return result;
  }
};
} // namespace flow_model
#endif
'''


def export(model_path, output):
    cfg, model = load(model_path)
    source_hash = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    lines = ['// GENERATED by export_flow_header.py; do not edit model constants.',
             '// Source SHA256: ' + source_hash,
             '#ifndef FLOW_MODEL_H', '#define FLOW_MODEL_H',
             '#include <cmath>', '#include <cstdint>', '#include <limits>',
             'namespace flow_model {',
             'static const char MODEL_VERSION[] = "flow-knn-2";',
             f'static const int REFERENCE_COUNT = {len(model.labels)};',
             f'static const int K = {model.k};', f'static const int WINDOW = {cfg.window};']
    for name, value in [('INTERVAL_SECONDS',cfg.interval_s), ('TOLERANCE_SECONDS',cfg.tolerance_s),
                        ('STALE_SECONDS',cfg.stale_s), ('MIN_VALID_FLOW',cfg.min_valid_flow),
                        ('MAX_VALID_FLOW',cfg.max_valid_flow), ('MIN_INLET_FLOW',cfg.min_inlet_lpm),
                        ('MAX_INLET_FLOW',cfg.max_inlet_lpm)]:
        lines.append(f'static const double {name} = {value!r};')
    for name, values in [('MEANS',model.means), ('SCALES',model.scales)]:
        lines.append(f'static const float {name}[4] = {{' + ', '.join(map(literal, values)) + '};')
    lines.append('static const float REFERENCES[REFERENCE_COUNT][4] = {')
    lines.extend('  {' + ', '.join(map(literal, row)) + '},' for row in model.references)
    lines.append('};')
    lines.append('static const std::uint8_t LABELS[REFERENCE_COUNT] = {' + ','.join(map(str,model.labels)) + '};')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text('\n'.join(lines) + '\n' + RUNTIME, encoding='utf-8')
    print(f'Exported {output}: {len(model.labels)} references, K={model.k}, window={cfg.window}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT/'flow_artifacts/model.json')
    parser.add_argument('--out', type=Path, default=ROOT/'embedded/flow_model.h')
    args = parser.parse_args()
    export(args.model, args.out)
