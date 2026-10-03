-- Proposed migration only: not yet applied to your Supabase project.
-- Creates a separate current-state table; does not modify hardware readings.
create table public.flow_model_state (
  pipeline_id text primary key,
  updated_at timestamptz not null,
  prediction jsonb not null
);
alter table public.flow_model_state enable row level security;
revoke all on public.flow_model_state from anon, authenticated;
grant select, insert, update on public.flow_model_state to service_role;
-- Dashboard read access must follow the existing project's user/tenant policy.
-- No public read policy is created until that access model is known.
