// Desktop test: g++ -std=c++11 parity_check.cpp -o parity_check
#include "flow_model.h"
#include <cassert>
#include <cstring>
#include <iostream>

int main() {
  flow_model::History history;
  double features[4];
  for (int i = 0; i < flow_model::WINDOW; ++i) {
    bool ready = history.update(1.0, 1000 + i * 20.0, features);
    assert(ready == (i == flow_model::WINDOW - 1));
  }
  assert(std::abs(features[3] - (flow_model::WINDOW-1)/3.0) < 1e-10);
  assert(features[1] == 0.0 && features[2] == 0.0);
  assert(!history.update(1, 10000, features)); // Gap resets the window.
  flow_model::Pipeline pipeline;
  const double q[3] = {10, 9, 8.95};
  const bool ok[3] = {true,true,true};
  flow_model::CycleResult result;
  double ts[3];
  for (int i = 0; i < flow_model::WINDOW; ++i) {
    ts[0] = ts[1] = ts[2] = 1000+i*20;
    result = pipeline.update(q, ts, ts[0], ok);
  }
  assert(result.segment[0].class_id >= 0 && result.segment[1].class_id >= 0);
  result = pipeline.update(q, ts, ts[0], ok);
  assert(std::strcmp(result.data_quality,"DUPLICATE_OR_OUT_OF_ORDER") == 0);
  result = pipeline.update(q, ts, ts[0]+100, ok);
  assert(std::strcmp(result.data_quality,"STALE_OR_FUTURE_DATA") == 0);
  // Python writes feature vectors to stdin, then compares these class IDs.
  while (std::cin >> features[0] >> features[1] >> features[2] >> features[3])
    std::cout << flow_model::predict(features).class_id << '\n';
}
