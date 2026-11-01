-- R3 schema v1. Sandbox tables are reset per eval episode; the rest persist.
create table if not exists customers (
  customer_id text primary key, first_name text not null, last_name text not null,
  phone text not null, date_of_birth date not null, branch_id text not null, created_at timestamptz default now());
create index if not exists customers_phone on customers(phone);
create table if not exists cards (
  card_id text primary key, customer_id text references customers, card_last4 text not null,
  status text not null check (status in ('active','blocked')), replacement text);
create table if not exists appointments (
  appointment_id text primary key, customer_id text references customers, branch_id text not null,
  service_type text not null, slot_start timestamp not null, status text not null default 'booked'
    check (status in ('booked','cancelled')), idempotency_key text unique);
create table if not exists tickets (
  ticket_id text primary key, customer_id text references customers, category text not null,
  summary text not null, label text, label_source text, created_at timestamptz default now(), idempotency_key text unique);
create table if not exists sessions (
  call_id text primary key, model_id text not null, started_at timestamptz not null, ended_at timestamptz,
  outcome text, state jsonb not null default '{}'::jsonb, scenario_id text, run_id text);
create table if not exists turn_events (
  id bigserial primary key, call_id text not null, turn_idx int not null, model_id text not null,
  event text not null, ts_ms bigint not null, wall_ts timestamptz not null default now(), payload jsonb not null default '{}'::jsonb);
create index if not exists turn_events_call on turn_events(call_id, turn_idx);
create table if not exists eval_results (
  id bigserial primary key, run_id text not null, model_id text not null, scenario_id text not null,
  family text not null, scorer text not null, pass boolean not null, detail jsonb, created_at timestamptz default now());
create index if not exists eval_results_run on eval_results(run_id, model_id);
create table if not exists cost_ledger (
  id bigserial primary key, run_id text, model_id text, item text not null, qty numeric not null,
  unit text, unit_cost numeric, measured boolean not null default true, note text, created_at timestamptz default now());
insert into cost_ledger(item, qty, unit, measured, note)
  values ('machine', 36, 'GB unified memory', true, 'Apple M5 Max, macOS 26.7, Homebrew PostgreSQL 15');
