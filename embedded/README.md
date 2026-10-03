# ESP32 / C++ model export

`flow_model.h` contains the saved model and standalone C++11 inference code.
It requires no Python, JSON parser, external ML library, or dynamic allocation.
The exporter uses the current `flow_artifacts/model.json`; it does not retrain.

## Arduino example

Create a sketch folder called `flow_model_example`. Copy `flow_model_example.ino`
and `flow_model.h` into it. Select your ESP32 board, compile/upload, and open the
serial monitor at 115200 baud. The example uses simulated inputs 10, 9, 8.95 L/min.
Expect warmup first, then the saved model's two segment predictions.

For real readings, keep one `flow_model::Pipeline` instance alive and call:

```cpp
double q[3] = {q1, q2, q3};            // Calibrated L/min, in node order.
double timestamps[3] = {t1, t2, t3};   // Measurement times in DOUBLE seconds.
bool healthy[3] = {true, true, true};  // Replace with actual sensor health.
auto result = detector.update(q, timestamps, now_seconds, healthy);
```

The coordinator must receive a fresh reading from every node before this call.
Use a common synchronized timebase for all timestamps and `now_seconds`. Do not
use a float for Unix epoch seconds, or pass milliseconds without dividing by 1000.
Handle timer rollover in firmware if using a wrapping timer. Packets, pulse
calibration, Wi-Fi/ESP-NOW, Supabase uploads and sensor drivers are not included.

`result.segment[0/1].class_id` is 0 normal, 1 minor, 2 major, -1 unavailable.
`result.overall` combines both segments; both leaking means CRITICAL.
Check `data_quality` and class IDs before using numerical fields: zero-filled
fields in an unavailable result are placeholders, not measurements. The result
also includes features, measured percentage losses, neighbor vote fractions,
and inlet/severity review flags. Vote fractions are not probabilities.

Five cycles are needed for warmup. Feature formulas match the Python definition:
difference, change per second, population variance, and right-endpoint integrated
volume across the N-1 intervals in the window. Gaps reset history. Current defaults
are 20-second cadence, 2-second synchronization tolerance and 45-second staleness.
Firmware must sample at the configured cadence and expire old results when no
new cycle arrives. This API consumes complete cycles; Python Detector accepts
individual messages. Their packet aggregation interfaces are intentionally different.

## Export and validation

From the project root:

```powershell
python -B export_flow_header.py
python -B verify_flow_header.py
```

The verification script checks float32 storage against Python predictions on
every processed example in `flow_training.csv`. If g++ or clang++ is on PATH, it
also compiles `parity_check.cpp`, runs feature/warmup/staleness/duplicate checks,
and compares actual C++ predictions against Python on the same vectors.
See `verification.json` for what was actually run; absence of a compiler is
reported explicitly. This does not replace an ESP32 build and hardware validation.

The 180 reference vectors, uint8 labels, and scaler occupy 3,092 bytes of array
data. Code, configuration, names, stack and histories are additional. Constants
are declared static const, with actual placement determined by the compiler.
Use the header in one implementation file to avoid copies across translation units.
Storage is float32; intermediate distance/feature/time arithmetic is double.
Tiny rounding differences may change predictions near a decision boundary even
if the supplied test set matches. The model is still synthetic-trained.

No live dashboard, Supabase project or ESP32 firmware has been changed by export.
