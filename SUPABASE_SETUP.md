# Connect the existing Supabase app to the Python model

Status: connector implemented and tested OFFLINE. No live database changes have
been made. The workspace contains no Supabase URL, credentials, app source or
confirmed table schema. Configuration names below are placeholders, not facts
about your live app.

## What is ready

- `supabase_worker.py`: standard-library Python worker; loads the existing model.
- `supabase_config.json`: configurable table/column names and hardware node IDs.
- `supabase_model_state.sql`: proposed current-result table with RLS enabled.
- `test_supabase_worker.py`: offline tests; no database required.

The worker reads existing telemetry, reconstructs the short rolling history using
recorded timestamps, and upserts one current result in `flow_model_state`. This
allows restart recovery without a local checkpoint and prevents duplicate polls
from accumulating the same volume twice. It does not alter sensor rows or train
the model. It is a current-state bridge, not a persistent event/history service.

## Information needed to finish

Provide the Supabase project URL, sensor table name and columns, one example row
with no secrets, the three hardware node identifiers, flow units and sampling
interval. Also provide the app source/repository to wire the dashboard and match
its existing authorization policy. Keep credentials outside chat and source code.

This first adapter expects one row per node reading with a timezone-aware SQL
timestamp column, numeric flow in L/min, and a node identifier. A table containing
all Q1/Q2/Q3 in one row needs a different mapping adapter. Scope the source table
or a protected view to one rig; the three configured IDs must uniquely identify
that rig. Do not mix experiments or installations.

## Configure once the schema is confirmed

1. Update `supabase_config.json` with the actual names. Set `status_column` if the
   hardware stores health; otherwise null assumes each supplied reading is OK.
   Map actual node IDs to `node_1`, `node_2`, `node_3` in inlet-to-outlet order.
2. Apply `supabase_model_state.sql` in the existing project's SQL editor after
   checking that the table does not already exist. It grants backend access only.
   Add dashboard SELECT permission and an RLS read policy matching the app's
   existing user/rig authorization. Do not disable RLS or use a broad public
   policy merely to make the page work.
3. Set `SUPABASE_URL` and `SUPABASE_SECRET_KEY` in the backend host environment.
   Use the project origin such as `https://PROJECT.supabase.co`, not the Lovable
   website. The secret belongs only in the backend. Legacy service-role JWTs
   are supported as well. No environment-file loader is included.

For a local PowerShell session, enter the key with a hidden prompt:

```powershell
$env:SUPABASE_URL = 'https://YOUR_PROJECT.supabase.co'
$flowCredential = Read-Host 'Supabase backend secret key' -AsSecureString
$env:SUPABASE_SECRET_KEY = [System.Net.NetworkCredential]::new('', $flowCredential).Password
python -B supabase_worker.py --once --dry-run
```

The dry run reads live rows and prints the prediction but writes nothing. Verify
the node mapping, timestamps, units and prediction before starting continuous writes:

```powershell
python -B supabase_worker.py
```

Stop with Ctrl+C. This terminal must stay open during local testing. For continuous
operation deploy one worker per rig on an always-on Python host, supplying the
script, `smart_flow_model.py`, configuration and `flow_artifacts/model.json`.
Do not run two writers against the same `pipeline_id`. Polling defaults to 5 seconds.

## Dashboard contract

Read `flow_model_state` filtered by `pipeline_id = rig_1` (or the configured ID).
Each row contains `pipeline_id`, `updated_at`, and a JSON object `prediction`.
Poll every five seconds using the app's existing authenticated Supabase client.
Realtime is not required, and the SQL migration does not enable it.

Example client query (reuse the existing app's Supabase client):

```javascript
const { data, error } = await supabase
  .from('flow_model_state')
  .select('updated_at,prediction')
  .eq('pipeline_id', 'rig_1')
  .maybeSingle();
// Treat error, no row, or updated_at older than 45 seconds as unavailable.
// Otherwise render data.prediction and its data_quality.
```

Display `prediction.q1/q2/q3`, `segment_1_state`, `segment_2_state`,
`overall_status`, `affected_segments`, `segments`, `warnings`, and `data_quality`.
These optional fields are absent when inference is unavailable: use null/--,
never zero or Normal. Show the measurement time from `prediction.timestamp`;
`updated_at` is the worker heartbeat, not a new measurement time. Mark a prediction
stale when its measurement timestamp exceeds the model's 45-second limit, even
if the heartbeat is current. Show `REVIEW_REQUIRED` prominently.

Paste into Lovable after the table and access policy are configured:

> Read current model results from flow_model_state where pipeline_id is rig_1,
> polling every five seconds with the existing Supabase client. Render the
> prediction JSON fields described above. Handle errors, no rows, expired
> heartbeat, expired measurement timestamps, null fields, WARMING_UP and all
> INFERENCE_UNAVAILABLE reasons explicitly. Replace locally simulated leak
> classifications with backend results. Keep existing hardware telemetry and
> authentication. Never put the backend secret key in frontend code.

## Operational limits

- The model was trained for 8-12 L/min inlet and 20-second cycles; startup needs
  five synchronized cycles. The worker rejects substantially different cadence
  rather than silently applying that model to 1 Hz data. Faster sampling needs
  agreed aggregation or retraining, not just a faster poll.
- Reads are bounded to a recent window and at most 499 rows per node. A full
  500-row response is rejected as possible truncation. Database API row limits
  must permit at least 500 rows per request. Timestamp filtering assumes a SQL
  timestamp column; numeric epoch columns require adapter changes.
- Missing/stale data, invalid rows and source read failures publish unavailable
  state. If the result write or worker fails, the frontend heartbeat timeout is
  essential. Database access failures are logged without credentials.
- Backend outages longer than the short read window are not backfilled as events.
- The migration deliberately does not grant frontend access until the app's
  actual tenant/user model is known. It has not been applied or tested live.

Validation: `python -B -m unittest test_supabase_worker -v`.

Reference: https://supabase.com/docs/guides/api and
https://supabase.com/docs/guides/getting-started/api-keys
