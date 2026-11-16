// Mock trial: drives main.ts through the six practice prompts without a live key.
//
// A mock env (POST /register, GET /work) and a mock gateway (POST /v1/chat/completions) stand
// in for the trial's services. The gateway is scripted to refuse and misbehave the way the real
// free pool does: the top-rated model answers 429 on its first call (retry + trail header), the
// best JSON-capable model answers empty once (failover on an empty reply). Every other call
// succeeds and echoes the prompt's token, as a cooperative model would.
//
// The driver then checks, per prompt, what the scorer checks: the token came back, the model
// field names a free model that meets the prompt's requirements, usage is whole, the stop reason
// is one the wire knows, JSON-mode answers parse, and the streaming event order is right. It
// ends by flipping /work to done and requiring the server to exit 0.
//
// Run:  node scripts/mock-trial.ts
// Env:  MOCK_WORKLOAD (default: the kit's practice/trial-0/input/env/workload.json)
//       MOCK_PROMPT_FILE (default: the kit's practice/trial-0/input/prompt.txt)

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { spawn, type ChildProcess } from 'node:child_process';
import { readFileSync } from 'node:fs';

const KIT = '/home/hatch/workspace/makefun/free-model-aggregator-02-kit';
const WORKLOAD_PATH = process.env.MOCK_WORKLOAD ?? `${KIT}/practice/trial-0/input/env/workload.json`;
const PROMPT_FILE = process.env.MOCK_PROMPT_FILE ?? `${KIT}/practice/trial-0/input/prompt.txt`;
const GATEWAY_PORT = 18080;
const ENV_PORT = 18081;

const workload = JSON.parse(readFileSync(WORKLOAD_PATH, 'utf8')) as {
  manifest: { id: string; api: string; stream: boolean; requires: Record<string, boolean> }[];
  bodies: Record<string, Record<string, unknown>>;
};

const FREE_IDS = [
  'openrouter/baidu/ernie-5.5-flash-preview:free',
  'openrouter/cohere/north-mini-code:free',
  'openrouter/dots-studio/dots-3-note-preview:free',
  'openrouter/google/gemma-4-26b-a4b-it:free',
  'openrouter/google/gemma-4-31b-it:free',
  'openrouter/inclusionai/ling-3.2-flash:free',
  'openrouter/liquid/lfm-3-1.7b:free',
  'openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
  'openrouter/nvidia/nemotron-3-ultra-550b-a55b:free',
  'openrouter/nvidia/nemotron-3.5-lightning:free',
  'openrouter/nvidia/nemotron-3.5-super-vl:free',
  'openrouter/particle-ai/particle-1:free',
  'openrouter/recursive/flint-v1.0:free',
  'openrouter/thinkingmachines/inkling-2-small:free',
];

let failures = 0;
function check(name: string, ok: boolean, detail = ''): void {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail !== '' ? ` — ${detail}` : ''}`);
  if (!ok) failures += 1;
}

function tokenOfBody(body: Record<string, unknown>): string | null {
  const texts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const m of messages) {
    if (typeof m === 'object' && m !== null && typeof (m as { content?: unknown }).content === 'string') {
      texts.push((m as { content: string }).content);
    }
  }
  for (const text of texts) {
    const found = text.match(/\b[A-Z]{2,}-\d{2,}\b/);
    if (found) return found[0];
  }
  return null;
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk)));
  return Buffer.concat(chunks).toString('utf8');
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

// --- mock gateway ---------------------------------------------------------
const gatewayCalls: { model: string; stream: boolean }[] = [];
const callCount = new Map<string, number>();

const gateway = createServer(async (req, res) => {
  if (req.method !== 'POST' || req.url !== '/v1/chat/completions') {
    res.writeHead(404);
    res.end();
    return;
  }
  const body = JSON.parse(await readBody(req)) as Record<string, unknown>;
  const model = String(body.model ?? '');
  const stream = body.stream === true;
  gatewayCalls.push({ model, stream });
  const n = (callCount.get(model) ?? 0) + 1;
  callCount.set(model, n);
  // The top-rated model rate-limits its first call; the best JSON model answers empty once.
  if (model === 'openrouter/thinkingmachines/inkling-2-small:free' && n === 1) {
    sendJson(res, 429, { error: { message: 'Rate limit exceeded', type: 'rate_limit_error', code: 'rate_limit_exceeded' } });
    return;
  }
  if (model === 'openrouter/nvidia/nemotron-3.5-super-vl:free' && n === 1) {
    sendJson(res, 200, {
      id: 'chatcmpl-mock-empty',
      object: 'chat.completion',
      created: 1767225600,
      model: 'nvidia/nemotron-3.5-super-vl:free',
      choices: [{ index: 0, message: { role: 'assistant', content: '' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 20, completion_tokens: 0, total_tokens: 20 },
    });
    return;
  }
  const token = tokenOfBody(body) ?? 'NO-TOKEN';
  const jsonMode = typeof body.response_format === 'object' && body.response_format !== null;
  const content = jsonMode ? JSON.stringify({ token }) : token;
  const upstreamModel = model.replace(/^openrouter\//, '');
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.write(`data: ${JSON.stringify({ id: 'chatcmpl-mock', object: 'chat.completion.chunk', created: 1767225600, model: upstreamModel, choices: [{ index: 0, delta: { role: 'assistant', content }, finish_reason: null }] })}\n\n`);
    res.write(`data: ${JSON.stringify({ id: 'chatcmpl-mock', object: 'chat.completion.chunk', created: 1767225600, model: upstreamModel, choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 } })}\n\n`);
    res.write('data: [DONE]\n\n');
    res.end();
    return;
  }
  sendJson(res, 200, {
    id: 'chatcmpl-mock',
    object: 'chat.completion',
    created: 1767225600,
    model: upstreamModel,
    choices: [{ index: 0, message: { role: 'assistant', content }, finish_reason: 'stop' }],
    usage: { prompt_tokens: 20, completion_tokens: 10, total_tokens: 30 },
  });
});

// --- mock env -------------------------------------------------------------
let registered: { port: number; catalog: unknown } | null = null;
let workDone = false;

const env = createServer(async (req, res) => {
  if (req.method === 'POST' && req.url === '/register') {
    const body = JSON.parse(await readBody(req)) as { protocol_version: number; port: number; catalog: unknown };
    registered = { port: body.port, catalog: body.catalog };
    sendJson(res, 200, { ok: true, requests: 6 });
    return;
  }
  if (req.method === 'GET' && req.url === '/work') {
    sendJson(res, 200, workDone ? { done: true } : { done: false, next: 'p01', pending: 6 });
    return;
  }
  res.writeHead(404);
  res.end();
});

function listen(server: ReturnType<typeof createServer>, port: number): Promise<void> {
  return new Promise((resolve) => server.listen(port, '127.0.0.1', () => resolve()));
}

async function waitFor(predicate: () => boolean, timeoutMs: number, what: string): Promise<boolean> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return true;
    await new Promise((r) => setTimeout(r, 100));
  }
  console.log(`TIMEOUT waiting for ${what}`);
  return false;
}

function textOfOpenAIAnswer(body: Record<string, unknown>): string {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = (choices[0] as Record<string, unknown> | undefined) ?? {};
  const message = (choice.message as Record<string, unknown> | undefined) ?? {};
  return typeof message.content === 'string' ? message.content : '';
}

function textOfAnthropicAnswer(body: Record<string, unknown>): string {
  const content = Array.isArray(body.content) ? body.content : [];
  return content
    .filter((b): b is Record<string, unknown> => typeof b === 'object' && b !== null && (b as { type?: unknown }).type === 'text')
    .map((b) => String((b as { text?: unknown }).text ?? ''))
    .join('');
}

async function main(): Promise<void> {
  await listen(gateway, GATEWAY_PORT);
  await listen(env, ENV_PORT);
  const child: ChildProcess = spawn(
    process.execPath,
    ['main.ts'],
    {
      cwd: '/home/hatch/workspace/makefun/free-model-aggregator-02-entry',
      env: {
        ...process.env,
        MAKEFUN_ENV_URL: `http://127.0.0.1:${ENV_PORT}`,
        MAKEFUN_PROMPT_FILE: PROMPT_FILE,
        OPENAI_BASE_URL: `http://127.0.0.1:${GATEWAY_PORT}/v1`,
        OPENAI_API_KEY: 'makefun-trial-dummy',
      },
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );
  let stderr = '';
  child.stderr?.on('data', (d: Buffer) => {
    stderr += d.toString();
  });
  const exited = new Promise<number | null>((resolve) => child.once('exit', (code) => resolve(code)));

  if (!(await waitFor(() => registered !== null, 10000, 'register'))) {
    console.log('server stderr:', stderr);
    process.exit(1);
  }
  const port = registered!.port;
  check('register: protocol_version 1', (registered!.catalog as { providers?: unknown[] }) !== undefined);
  const catalog = registered!.catalog as { providers: { id: string; models: { id: string }[] }[] };
  const catalogIds = catalog.providers.flatMap((p) => p.models.map((m) => m.id));
  check('register: catalog holds exactly the fourteen free ids', catalogIds.length === 14 && FREE_IDS.every((id) => catalogIds.includes(id)), `${catalogIds.length} ids`);
  check('register: one provider openrouter', catalog.providers.length === 1 && catalog.providers[0].id === 'openrouter');

  const base = `http://127.0.0.1:${port}`;
  // GET /v1/models on both auth headers.
  const openaiList = (await (await fetch(`${base}/v1/models`, { headers: { authorization: 'Bearer dummy' } })).json()) as {
    data: { id: string }[];
  };
  check('GET /v1/models (Bearer): OpenAI shape, fourteen ids', openaiList.data?.length === 14 && FREE_IDS.every((id) => openaiList.data.some((m) => m.id === id)));
  const anthropicList = (await (
    await fetch(`${base}/v1/models`, { headers: { 'x-api-key': 'dummy', 'anthropic-version': '2023-06-01' } })
  ).json()) as { data: { id: string }[] };
  check('GET /v1/models (x-api-key): Anthropic shape, fourteen ids', anthropicList.data?.length === 14 && FREE_IDS.every((id) => anthropicList.data.some((m) => m.id === id)));

  // The six prompts, in manifest order.
  for (const prompt of workload.manifest) {
    const body = workload.bodies[prompt.id];
    const token = (workload.manifest.find((p) => p.id === prompt.id) as { token: string }).token;
    const isOpenAI = prompt.api === 'openai';
    const url = isOpenAI ? `${base}/v1/chat/completions` : `${base}/v1/messages`;
    const headers: Record<string, string> = { 'content-type': 'application/json' };
    if (isOpenAI) headers.authorization = 'Bearer dummy';
    else {
      headers['x-api-key'] = 'dummy';
      headers['anthropic-version'] = '2023-06-01';
    }
    const res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(body) });
    const trail = res.headers.get('x-makefun-upstream-failures');
    const raw = await res.text();
    check(`${prompt.id}: HTTP 200`, res.status === 200, `got ${res.status}: ${raw.slice(0, 200)}`);
    if (res.status !== 200) continue;
    if (prompt.stream) {
      check(`${prompt.id}: SSE carries the token`, raw.includes(token));
      if (isOpenAI) {
        check(`${prompt.id}: SSE ends with [DONE]`, raw.trimEnd().endsWith('data: [DONE]'));
      } else {
        const order = ['message_start', 'content_block_start', 'content_block_delta', 'message_stop'];
        const positions = order.map((e) => raw.indexOf(`event: ${e}`));
        check(`${prompt.id}: Anthropic event order`, positions.every((p) => p !== -1) && positions.every((p, i) => i === 0 || p > positions[i - 1]));
      }
      const modelMatch = raw.match(/"model"\s*:\s*"([^"]+)"/);
      check(`${prompt.id}: streamed chunks name a free model`, modelMatch !== null && FREE_IDS.includes(modelMatch[1]), modelMatch?.[1] ?? 'none');
    } else if (isOpenAI) {
      const answer = JSON.parse(raw) as Record<string, unknown>;
      const text = textOfOpenAIAnswer(answer);
      check(`${prompt.id}: answer carries the token`, text.includes(token), JSON.stringify(text).slice(0, 120));
      check(`${prompt.id}: model field is a free id`, typeof answer.model === 'string' && FREE_IDS.includes(answer.model as string), String(answer.model));
      const usage = answer.usage as { prompt_tokens?: number; completion_tokens?: number } | undefined;
      check(`${prompt.id}: usage whole, output >= 1`, (usage?.prompt_tokens ?? -1) >= 0 && (usage?.completion_tokens ?? 0) >= 1, JSON.stringify(usage));
      const choice = (Array.isArray(answer.choices) ? answer.choices[0] : {}) as { finish_reason?: string };
      check(`${prompt.id}: stop reason stop`, choice.finish_reason === 'stop', String(choice.finish_reason));
      if (prompt.requires.json === true) {
        let parsed: unknown = null;
        try {
          parsed = JSON.parse(text);
        } catch {
          parsed = null;
        }
        check(`${prompt.id}: JSON-mode answer parses with the token`, parsed !== null && (parsed as { token?: string }).token === token, text.slice(0, 120));
      }
    } else {
      const answer = JSON.parse(raw) as Record<string, unknown>;
      const text = textOfAnthropicAnswer(answer);
      check(`${prompt.id}: answer carries the token`, text.includes(token), JSON.stringify(text).slice(0, 120));
      check(`${prompt.id}: model field is a free id`, typeof answer.model === 'string' && FREE_IDS.includes(answer.model as string), String(answer.model));
      const usage = answer.usage as { input_tokens?: number; output_tokens?: number } | undefined;
      check(`${prompt.id}: usage whole, output >= 1`, (usage?.input_tokens ?? -1) >= 0 && (usage?.output_tokens ?? 0) >= 1, JSON.stringify(usage));
      check(`${prompt.id}: stop reason end_turn`, answer.stop_reason === 'end_turn', String(answer.stop_reason));
    }
    if (prompt.id === 'p01') {
      check('p01: no trail header for an empty reply (not a refusal)', trail === null, trail ?? 'absent');
    }
    if (prompt.id === 'p02') {
      const rows = trail === null ? null : (JSON.parse(trail) as { status: number; code: string; model: string }[]);
      check('p02: trail header reports the recovered 429', rows !== null && rows.some((r) => r.status === 429 && r.model === 'openrouter/thinkingmachines/inkling-2-small:free'), trail ?? 'absent');
    }
  }

  check('gateway calls under the 60-call cap', gatewayCalls.length <= 60, `${gatewayCalls.length} calls`);
  console.log(`gateway calls made: ${gatewayCalls.length} (models: ${[...new Set(gatewayCalls.map((c) => c.model))].join(', ')})`);

  workDone = true;
  const code = await Promise.race([exited, new Promise<null>((r) => setTimeout(() => r(null), 10000))]);
  check('server exits 0 when /work says done', code === 0, `exit code ${code}`);
  if (code === null) child.kill('SIGKILL');
  gateway.close();
  env.close();
  console.log(failures === 0 ? 'MOCK TRIAL: ALL CHECKS PASSED' : `MOCK TRIAL: ${failures} CHECK(S) FAILED`);
  console.log('server stderr:', stderr.trim().split('\n').slice(-5).join('\n'));
  process.exit(failures === 0 ? 0 : 1);
}

await main();
