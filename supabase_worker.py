"""Supabase-to-model worker. See SUPABASE_SETUP.md before connecting live data.

No third-party packages. Rebuilds the short feature window on each poll, so
restarts and repeated polls do not double-count readings or accumulated volume.
Writes one current-state row; it does not create leak events or historical data.
"""
import argparse
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from smart_flow_model import Detector, NODES, iso, load, stamp

ROOT = Path(__file__).resolve().parent


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    for name in ('source_table', 'result_table', 'time_column', 'node_column', 'flow_column'):
        if not re.fullmatch(r'[a-zA-Z_][a-zA-Z_0-9]*', cfg[name]):
            raise ValueError('Invalid table/column identifier: ' + name)
    if cfg.get('status_column') and not re.fullmatch(r'[a-zA-Z_][a-zA-Z_0-9]*', cfg['status_column']):
        raise ValueError('Invalid status column')
    if sorted(cfg['node_map'].values()) != sorted(NODES):
        raise ValueError('node_map must map three distinct hardware IDs to node_1, node_2, node_3')
    if cfg['poll_seconds'] < 1:
        raise ValueError('Polling interval must be at least one second')
    return cfg


class Rest:
    def __init__(self):
        url = os.environ.get('SUPABASE_URL', '').rstrip('/')
        key = os.environ.get('SUPABASE_SECRET_KEY', '')
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.path or parsed.query or parsed.fragment or parsed.username:
            raise ValueError('Set SUPABASE_URL to the HTTPS project origin, not the Lovable URL')
        if not key:
            raise ValueError('Set SUPABASE_SECRET_KEY in this backend process environment')
        self.url = url + '/rest/v1/'
        self.headers = {'apikey': key, 'Content-Type': 'application/json'}
        if not key.startswith('sb_secret_'):
            self.headers['Authorization'] = 'Bearer ' + key  # legacy service-role JWT

    def request(self, table, params, data=None):
        headers = dict(self.headers)
        if data is not None:
            headers['Prefer'] = 'resolution=merge-duplicates,return=minimal'
        request = urllib.request.Request(self.url + table + '?' + urllib.parse.urlencode(params),
            data=None if data is None else json.dumps(data, allow_nan=False).encode(), headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            # Never print request headers, keys or database response bodies.
            raise RuntimeError(f'Supabase HTTP {exc.code}: check table mapping, key and access grants') from None
        except urllib.error.URLError:
            raise RuntimeError('Supabase connection failed; check network and project URL') from None


def fetch_rows(rest, mapping, model_cfg, now):
    # A bounded recent window is sufficient for these rolling model features.
    horizon = (model_cfg.window + 2) * model_cfg.stale_s
    columns = [mapping[k] for k in ('time_column', 'node_column', 'flow_column')]
    if mapping.get('status_column'):
        columns.append(mapping['status_column'])
    rows = []
    # Query each configured node separately; unrelated nodes never enter the model.
    # Fail closed at the limit rather than infer from silently truncated history.
    for hardware_id in mapping['node_map']:
        batch = rest.request(mapping['source_table'], {
            'select': ','.join(columns), mapping['node_column']: 'eq.' + hardware_id,
            mapping['time_column']: 'gte.' + iso(now-horizon),
            'order': mapping['time_column'] + '.desc', 'limit': '500'})
        if not isinstance(batch, list) or len(batch) >= 500:
            raise ValueError('Too many rows or unexpected source response; verify sampling cadence')
        rows.extend(batch)
    return rows


def infer_rows(rows, mapping, cfg, model, now):
    detector = Detector(cfg, model)
    messages = {}
    try:
        for row in rows:
            node = mapping['node_map'][str(row[mapping['node_column']])]
            timestamp = stamp(row[mapping['time_column']])
            value = row[mapping['flow_column']]
            if isinstance(value, bool):
                raise ValueError('Boolean flow')
            flow = float(value)
            if not math.isfinite(flow):
                raise ValueError('Nonfinite flow')
            status = row.get(mapping.get('status_column'), 'OK')
            item = dict(node_id=node, timestamp=timestamp, flow_lpm=flow, device_status=status)
            key = (timestamp, node)
            if key in messages and messages[key] != item:
                return detector.unavailable('CONFLICTING_DUPLICATE_READINGS')
            messages[key] = item
    except (KeyError, ValueError, TypeError, OverflowError):
        return detector.unavailable('INVALID_SOURCE_ROW')
    latest = {}
    for (timestamp, node), item in sorted(messages.items()):
        latest[node] = item
    if set(latest) != set(NODES):
        return detector.unavailable('MISSING_NODE_DATA')
    if any(v['timestamp'] > now + cfg.tolerance_s for v in latest.values()):
        return detector.unavailable('FUTURE_DATA')
    if any(now-v['timestamp'] > cfg.stale_s for v in latest.values()):
        return detector.unavailable('STALE_DATA')
    if any(v['device_status'] != 'OK' or not cfg.min_valid_flow <= v['flow_lpm'] <= cfg.max_valid_flow for v in latest.values()):
        return detector.unavailable('SENSOR_FAULT_OR_FLOW_OUT_OF_RANGE')
    result = detector.unavailable('WARMING_UP')
    completed = []
    for (timestamp, node), item in sorted(messages.items()):
        # Replay actual recorded measurement intervals to rebuild history.
        # Freshness is checked against the real clock above and below.
        current = detector.ingest(item, now=timestamp)
        if current['data_quality'] != 'MISSING_NODE_DATA':
            result = current
        if 'timestamp' in current:
            completed.append(stamp(current['timestamp']))
        elif current['data_quality'] == 'WARMING_UP':
            completed.append(timestamp)
    if 'timestamp' in result and now-stamp(result['timestamp']) > cfg.stale_s:
        return detector.unavailable('STALE_DATA')
    recent = completed[-cfg.window:]
    if len(recent) >= 2 and any(abs(b-a-cfg.interval_s) > max(cfg.tolerance_s*2, cfg.interval_s*.25)
                                            for a,b in zip(recent, recent[1:])):
        return detector.unavailable('SAMPLING_INTERVAL_MISMATCH')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'supabase_config.json'))
    parser.add_argument('--model', default=str(ROOT/'flow_artifacts/model.json'))
    parser.add_argument('--once', action='store_true', help='process one polling cycle')
    parser.add_argument('--dry-run', action='store_true', help='read live data, print results, do not write')
    args = parser.parse_args()
    mapping = read_config(args.config)
    cfg, model = load(args.model)
    rest = Rest()
    while True:
        now = time.time()
        try:
            result = infer_rows(fetch_rows(rest, mapping, cfg, now), mapping, cfg, model, now)
        except (RuntimeError, ValueError) as exc:
            print(str(exc), flush=True)
            result = Detector(cfg, model).unavailable('SOURCE_READ_FAILED')
        output = dict(pipeline_id=mapping['pipeline_id'], updated_at=iso(time.time()), prediction=result)
        if args.dry_run:
            print(json.dumps(output, allow_nan=False), flush=True)
        else:
            try:
                rest.request(mapping['result_table'], {'on_conflict':'pipeline_id'}, output)
                print(f"{output['updated_at']} {result['overall_status']} / {result['data_quality']}", flush=True)
            except RuntimeError as exc:
                print(str(exc), flush=True)
                if args.once:
                    raise
        if args.once:
            return
        time.sleep(mapping['poll_seconds'])


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nWorker stopped.')
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        raise SystemExit(str(exc))
