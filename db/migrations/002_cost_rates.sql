-- Shared pricing constants for the cost ledger (plan: cost per successfully completed task, all three paths).
-- Every row names its source and date. "list" = vendor list price as published (via secondary sources on Nov 1,
-- flagged in note until checked on the vendor page); "assumption" = our accounting choice.
create table if not exists cost_rates (
  key        text primary key,
  value      numeric not null,
  unit       text not null,
  kind       text not null check (kind in ('list', 'assumption', 'measured')),
  source     text,
  as_of      date not null default current_date,
  note       text
);
insert into cost_rates (key, value, unit, kind, source, note) values
  ('deepgram.nova3.streaming_usd_per_min', 0.0077, 'USD/min', 'list', 'https://convertaudiototext.com/blog/deepgram-nova-3-explained', 'Nova-3 monolingual streaming pay-as-you-go; secondary source, confirm on deepgram.com'),
  ('cartesia.sonic.usd_per_char', 0.0000392, 'USD/char', 'list', 'https://soniox.com/pricing/tts/soniox-vs-cartesia', 'Startup plan $49 / 1.25M credits, 1 credit = 1 character (Sonic 3.6); Scale plan is $0.0000374'),
  ('livekit.agent_session_usd_per_min', 0.01, 'USD/min', 'list', 'https://www.forasoft.com/blog/article/livekit-vs-agora-cost-analysis', 'agent session minute, LiveKit Cloud; secondary source, confirm on livekit.com/pricing'),
  ('livekit.webrtc_participant_usd_per_min', 0.0005, 'USD/min', 'list', 'https://www.forasoft.com/blog/article/livekit-vs-agora-cost-analysis', 'per participant minute (caller + agent = 2); Ship-plan overage'),
  ('anthropic.claude-sonnet-5-5.input_usd_per_mtok', 2.00, 'USD/Mtok', 'list', 'https://pricepertoken.com/pricing-page/model/anthropic-claude-sonnet-5.5', 'aggregator citing Anthropic docs'),
  ('anthropic.claude-sonnet-5-5.output_usd_per_mtok', 10.00, 'USD/Mtok', 'list', 'https://pricepertoken.com/pricing-page/model/anthropic-claude-sonnet-5.5', 'aggregator citing Anthropic docs'),
  ('anthropic.claude-sonnet-5-5.cache_read_usd_per_mtok', 0.20, 'USD/Mtok', 'list', 'https://apidog.com/blog/claude-sonnet-5-5-pricing/', 'not applied: the proxy does not request caching'),
  ('mac.purchase_usd', 4099, 'USD', 'list', 'https://www.iclarified.com/102828/14-inch-macbook-pro-m5-max-drops-to-3299', '14-inch MacBook Pro M5 Max 36 GB / 2 TB release-date retail'),
  ('mac.amortization_hours', 26280, 'hours', 'assumption', null, '3 years x 24 h: the machine serves around the clock'),
  ('mac.workday_amortization_hours', 6000, 'hours', 'assumption', null, 'alternative: 3 years x 250 days x 8 h'),
  ('mac.power_kw', 0.10, 'kW', 'assumption', null, 'M5 Max under sustained inference load, whole machine'),
  ('electricity_usd_per_kwh', 0.15, 'USD/kWh', 'assumption', null, 'US average residential'),
  ('admission.concurrent_calls', 1, 'calls', 'measured', null, 'placeholder until the load test sets it (scripts/cost_report.py --set-admission N)')
on conflict (key) do nothing;

-- Nov 1 04:24: Cartesia quota exhausted (402 quota_exceeded); Deepgram Aura-2 is the fallback TTS (AGENT_TTS=deepgram)
insert into cost_rates(key, value, unit, kind, source, as_of, note) values
  ('deepgram.aura2.usd_per_char', 0.00003, 'usd/char', 'assumption', 'https://deepgram.com/pricing (Aura-2 TTS pay-as-you-go $0.030 per 1K characters)', '2026-11-01',
   'fallback TTS after Cartesia quota_exceeded at 04:24; override if the account is on a different plan') on conflict (key) do nothing;
