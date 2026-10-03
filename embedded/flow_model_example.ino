// Copy this sketch and flow_model.h into an Arduino sketch folder named
// flow_model_example. This runs simulated cycles; it does not read hardware.
#include "flow_model.h"

flow_model::Pipeline detector;

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("SIMULATED FLOW MODEL TEST - not live sensor readings");
  const double flows[3] = {10.0, 9.0, 8.95};
  const bool sensor_ok[3] = {true, true, true};
  for (int cycle = 0; cycle < flow_model::WINDOW + 2; ++cycle) {
    // Simulated seconds: no real-time delay is required for this demo.
    double now = 1000.0 + cycle * flow_model::INTERVAL_SECONDS;
    const double timestamps[3] = {now, now, now};
    flow_model::CycleResult result = detector.update(flows, timestamps, now, sensor_ok);
    Serial.printf("Cycle %d: %s | %s\n", cycle+1, result.overall, result.data_quality);
    if (result.segment[0].class_id >= 0 && result.segment[1].class_id >= 0) {
      Serial.printf("Segment 1: %s, loss %.2f%%; Segment 2: %s, loss %.2f%%\n",
        flow_model::className(result.segment[0].class_id), result.loss_percent[0],
        flow_model::className(result.segment[1].class_id), result.loss_percent[1]);
    }
  }
}

void loop() {
  // Replace the setup demo with your coordinator's real complete-cycle handler.
  // Supply all three calibrated L/min values, their measurement timestamps
  // and device health. Keep detector alive between cycles; do not recreate it.
  // If packets stop, expire the last displayed result after STALE_SECONDS.
}
