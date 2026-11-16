# free-model-aggregator-02

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/gmxbtt/free-model-aggregator-02/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/gmxbtt/free-model-aggregator-02/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

Free-model aggregator: one server serving the OpenAI and Anthropic wires, every answer a real completion from a free OpenRouter model, chosen by the brief's quality ratings with retry and failover.

## Run it

```sh
npm install
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
npm run smoke        # one live OpenRouter request, end to end
node scripts/mock-trial.ts   # the six practice prompts against a mock env + gateway
node main.ts --list  # prints the catalog the server registers
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=vg0kgv1n&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>

## The server

`main.ts` is the entry: Node 22 standard library only, no dependencies at run time
(the trial runs `node main.ts` with nothing installed). It listens on `0.0.0.0` at a port
it picks, registers `{protocol_version: 1, port, catalog}` with `POST $MAKEFUN_ENV_URL/register`,
serves `GET /v1/models` (OpenAI shape for `Authorization: Bearer`, Anthropic shape for
`x-api-key`), `POST /v1/chat/completions` and `POST /v1/messages` in JSON and in streaming,
and polls `GET $MAKEFUN_ENV_URL/work` until the env says done, then exits 0.

Every prompt is answered by a real free model dialled through the grader's gateway on its
Chat Completions route (`$OPENAI_BASE_URL/chat/completions`, key from `OPENAI_API_KEY` in the
trial — a dummy the gateway swaps for the entrant's pushed OpenRouter key). Anthropic-shaped
requests are translated to Chat Completions and back by `lib/wire.ts`.

How a prompt is served, in the order the score pays for it:

- **Quality (60%).** The brief rates every free model (`rating_milli`). The server tries the
  highest-rated free model that meets the prompt's requirements (tools / vision / JSON mode /
  context) — currently `openrouter/thinkingmachines/inkling-2-small:free` (1475) where it
  qualifies, else `openrouter/nvidia/nemotron-3.5-super-vl:free` (1303) for JSON mode.
- **Served (15%).** An answer only counts when the provider's own body carries the prompt's
  unique token, so the server reads that token out of the request and checks the model's
  reply for it. A reply without the token is not served: the server fails over to the next
  model, and only if no model echoes the token does it serve its best attempt as a fallback.
- **Wire (10%).** Both wires are answered exactly as asked — JSON or server-sent events,
  usage and stop reason whole, the `model` field the id actually dialled on every answer and
  every streamed chunk.
- **Failover (10%).** A 429 or 5xx is retried once on the same model, then the next model is
  tried. Every refusal a served answer recovered from is reported in the
  `x-makefun-upstream-failures` header. A model that refused once is *not* demoted for later
  prompts: a free-pool 429 is usually a brief burst, and skipping the best model would cost
  quality to save at most two gateway calls.
- **Catalog (5%).** Exactly the fourteen free ids, once each, on the register body, on
  `--list`, and on both `GET /v1/models` answers.

Determinism: every choice is a function of the brief, the requests and the gateway's
answers only — no wall clock, no unseeded randomness. A trial makes at most 3 models × 2
attempts per prompt (36 calls), under the gateway's 60-call cap.

`lib/` is the starter kit's brief parser and wire translation, vendored unchanged; the
routing, failover, token check and trial loop in `main.ts` are this entry's own.

## OpenRouter integration

`src/openrouter-client.ts` is the client used by the smoke test. It uses the OpenAI SDK
pointed at an OpenAI-compatible base URL, so the same code dials OpenRouter directly and
the grader's gateway: the gateway address comes from the environment, and the key is read
from the environment only. It is never written to a file, a log or the repository.

- `createClient()` reads `OPENROUTER_API_KEY` and `OPENAI_BASE_URL`
  (default `https://openrouter.ai/api/v1`). A missing key fails at once
  with a clear message.
- `chatCompletion()` sends one completion. It always sends `max_tokens`,
  because a request without an output cap is refused. Small free models
  spend much of their budget on reasoning, so the default cap is 256.
- `chatWithFailover()` tries each model in order, retries a 429 or 5xx
  once, then fails over to the next free model. It returns the first
  reply with visible text plus the trail of failures.

## One working request, end to end

`scripts/smoke.ts` sends one prompt asking for a unique token and checks
the reply carries it back.

```sh
cd ~/workspace/makefun/free-model-aggregator-02-entry
npm install
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
npm run smoke
```

Expected output on success:

```text
SMOKE OK: model=cohere/north-mini-code:free finish=stop usage=15+54 tokens
Reply: SMOKE-OK-4711
```

Verified live 2026-11-17: the same request answered with the token,
cost $0. A free model can answer 429 on the shared pool; the smoke test
fails over to the next free model instead of failing. A cap below 256
can end the reply with no visible text, because the model spends the
budget on reasoning.

## Mock trial

`scripts/mock-trial.ts` runs the server against a mock env and a mock gateway
through all six practice prompts (trial-0) and checks what the scorer checks:
token echoed, model field a free id meeting the requirements, usage whole,
stop reason valid, JSON-mode answers parse, streaming event order right,
failover header on the recovered 429 — then flips `/work` to done and
requires exit 0. The mock gateway rate-limits the top model once and
answers empty once, so the retry and failover paths are exercised.

```sh
node scripts/mock-trial.ts   # expect: MOCK TRIAL: ALL CHECKS PASSED
```

## Environment

| Variable | Meaning | Default |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | Your OpenRouter key (starts `sk-or-v1-`) | required, no default |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL to dial | `https://openrouter.ai/api/v1` |
| `OPENROUTER_MODEL` | First model the smoke test tries | `cohere/north-mini-code:free` |
| `MAKEFUN_ENV_URL` | Trial env origin (set by the trial) | — |
| `MAKEFUN_PROMPT_FILE` | Brief file (set by the trial) | — |
| `OPENAI_API_KEY` | Trial dummy key, swapped by the gateway | — |

Keep the key out of the repository: `.env` is gitignored. The key is
also pushed to the competition as a sealed secret before submitting
(`makefun secrets push free-model-aggregator-02 --dotenv .env`), so the
grader's gateway can dial models on it without the key ever entering
the entry.

## Free model limits

Free models cost $0. An account that has bought under 10 credits gets
50 free requests a day; 10 or more credits bought raises that to 1,000
a day. The free shared pool can still rate limit a single model, which
is why the client retries and fails over.
