// free-model-aggregator-02: one server, two client wires, one real free model behind every answer.
//
// The server listens on 0.0.0.0 at a port it picks, registers that port and its catalog with the
// trial env, and answers the six prompts the env sends it — three on the OpenAI wire
// (POST /v1/chat/completions), three on the Anthropic wire (POST /v1/messages) — each with a real
// completion from one of the fourteen free OpenRouter models the brief lists, dialled through the
// grader's gateway on its Chat Completions route. GET /v1/models answers in the wire its auth
// header names. `node main.ts --list` prints the catalog and exits 0. The server polls
// GET $MAKEFUN_ENV_URL/work and exits 0 when the env says done.
//
// HOW A PROMPT IS SERVED (the scoring columns, in the order they pay)
//   quality (60%): the brief rates every free model (rating_milli). For each prompt the server
//     tries the highest-rated free model that meets the prompt's requirements (tools / vision /
//     JSON mode / context), so every prompt is served by the best model it may use.
//   served (15%): an answer only counts when the provider's own body carries the prompt's unique
//     token, so the server reads that token out of the request and checks the model's reply for
//     it. A reply without the token is not served: the server fails over to the next model, and
//     only if no model echoes the token does it serve its best attempt as a fallback.
//   wire (10%): both wires are answered exactly as the client asked — JSON or server-sent events,
//     usage and stop reason whole, the model field the id actually dialled on every answer and
//     every streamed chunk. The translation between the wires is lib/wire.ts (the starter kit's,
//     vendored unchanged); this file only routes, dials and validates.
//   failover (10%): a 429 or 5xx is retried once on the same model, then the next model is tried.
//     Every refusal a served answer recovered from is reported in the
//     x-makefun-upstream-failures header, so the column can see it. A 401 stops the prompt at
//     once (the key is missing: every model would refuse the same way).
//   catalog (5%): the catalog is exactly the fourteen free ids, once each, on the register body,
//     on --list, and on both GET /v1/models answers.
//
// DETERMINISM (the replay must reproduce the state export byte for byte)
//   Every choice is a function of the brief, the requests and the gateway's answers only. There
//   is no wall clock and no unseeded randomness anywhere in this file: candidates are always
//   the brief's rating order, retries wait a fixed delay, and the model listings carry fixed
//   timestamps (lib/wire.ts). A model that refused once is NOT demoted for later prompts: a
//   free-pool 429 is usually a brief burst, and skipping the best model on the next prompt
//   would cost quality (60% of the score) to save at most two gateway calls, which the
//   per-prompt cap (3 models x 2 attempts, at most 36 calls a trial) already bounds. A trial
//   makes at most 3 models x 2 attempts per prompt, so at most 36 gateway calls — under the
//   gateway's 60-call cap.
//
// The brief and the wire translation come from the starter kit (lib/, vendored unchanged); the
// routing, the failover, the token check and the trial loop below are this entry's own.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** How many models are tried for one prompt, best first. */
const MAX_MODELS_PER_PROMPT = 3;
/** How many attempts one model gets for one prompt (the first, then one retry). */
const MAX_ATTEMPTS_PER_MODEL = 2;
/** The fixed wait between a refused attempt and its retry, in milliseconds. */
const RETRY_DELAY_MS = 1000;
/** The most refusals one served answer reports in its trail header. */
const MAX_TRAIL_ROWS = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** One refusal the trail header reports: the status, the error code and the model that refused. */
export interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}

/** The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach it. */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** The prompt's unique token, read out of the request's own text (e.g. ZEBRA-4721), or null. */
function tokenOf(body: Json): string | null {
  const texts: string[] = [];
  if (typeof body.system === 'string') texts.push(body.system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    if (typeof message.content === 'string') texts.push(message.content);
    else if (Array.isArray(message.content)) {
      for (const block of message.content) {
        if (isRecord(block) && typeof block.text === 'string') texts.push(block.text);
      }
    }
  }
  for (const text of texts) {
    const found = text.match(/\b[A-Z]{2,}-\d{2,}\b/);
    if (found !== null) return found[0];
  }
  return null;
}

/** The visible text of a Chat Completions answer: its message text, plus any tool call arguments. */
function textOfAnswer(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const parts: string[] = [];
  if (typeof message.content === 'string') parts.push(message.content);
  else if (Array.isArray(message.content)) {
    for (const block of message.content) {
      if (isRecord(block) && typeof block.text === 'string') parts.push(block.text);
    }
  }
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    if (typeof call.function.arguments === 'string') parts.push(call.function.arguments);
  }
  return parts.join('\n');
}

/** The visible text of buffered SSE event strings: the delta text of each chunk. */
function textOfEvents(events: readonly string[]): string {
  const parts: string[] = [];
  for (const event of events) {
    for (const line of event.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '' || payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      // An OpenAI chunk: choices[0].delta.content. An Anthropic event: a text_delta.
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice !== null) {
        const delta = isRecord(choice.delta) ? choice.delta : {};
        if (typeof delta.content === 'string') parts.push(delta.content);
      }
      const delta = isRecord(chunk.delta) ? chunk.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string') parts.push(delta.text);
    }
  }
  return parts.join('');
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** The trail header for a served answer that recovered from refusals, or no header at all. */
function trailHeader(failures: readonly UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_TRAIL_ROWS)) };
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** One gateway call's outcome. `ok` carries the answer to serve; `refused` a gateway error;
 * `empty` a 200 whose body carries no visible text (a free model that spent its budget reasoning). */
type Attempt =
  | { kind: 'ok'; model: string; text: string; json: Json | null; events: string[] | null }
  | { kind: 'refused'; model: string; status: number; code: string; retryable: boolean }
  | { kind: 'empty'; model: string };

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

/** The error code of a refused gateway call, from its error body when it names one. */
async function refusalOf(res: Response, model: string): Promise<Attempt> {
  const status = res.status;
  let code = status === 429 ? 'rate_limited' : status >= 500 ? 'upstream_error' : 'invalid_request';
  try {
    const body: unknown = await res.json();
    if (isRecord(body) && isRecord(body.error) && typeof body.error.code === 'string' && body.error.code !== '') {
      code = body.error.code;
    } else if (isRecord(body) && typeof body.code === 'string' && body.code !== '') {
      code = body.code;
    }
  } catch {
    // The refusal carried no JSON body: keep the code the status implies.
  }
  const retryable = status === 429 || status >= 500;
  return { kind: 'refused', model, status, code, retryable };
}

/** One Chat Completions call on one model: JSON when the client asked for JSON, SSE frames when it asked for events. */
async function dial(model: string, body: Json, stream: boolean): Promise<Attempt> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (stream) headers.accept = 'text/event-stream';
  let upstream: Response;
  try {
    upstream = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...body, model, ...(stream ? { stream: true } : {}) }),
    });
  } catch (err) {
    return { kind: 'refused', model, status: 0, code: 'network_error', retryable: true };
  }
  if (!upstream.ok || upstream.body === null) return refusalOf(upstream, model);
  if (!stream) {
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      return { kind: 'refused', model, status: 502, code: 'bad_upstream_body', retryable: true };
    }
    if (!isRecord(answer)) return { kind: 'refused', model, status: 502, code: 'bad_upstream_body', retryable: true };
    const text = textOfAnswer(answer);
    if (text.trim() === '') return { kind: 'empty', model };
    return { kind: 'ok', model, text, json: answer, events: null };
  }
  // Streaming: buffer the wire's events (with the dialled model on every chunk), then validate.
  const events: string[] = [];
  try {
    for await (const frame of openAIStreamWithModel(upstream.body, model)) events.push(frame);
  } catch {
    return { kind: 'refused', model, status: 502, code: 'bad_upstream_stream', retryable: true };
  }
  const text = textOfEvents(events);
  if (text.trim() === '') return { kind: 'empty', model };
  return { kind: 'ok', model, text, json: null, events };
}

/** One Anthropic-shaped call, translated to Chat Completions and back: the events are already in
 * the Anthropic order, buffered so the token check can run before anything reaches the client. */
async function dialAnthropic(model: string, body: Json, stream: boolean): Promise<Attempt> {
  const upstreamBody = anthropicToOpenAI(body, model);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (stream) headers.accept = 'text/event-stream';
  let upstream: Response;
  try {
    upstream = await fetch(chatCompletionsUrl(), {
      method: 'POST',
      headers,
      body: JSON.stringify({ ...upstreamBody, ...(stream ? { stream: true } : {}) }),
    });
  } catch {
    return { kind: 'refused', model, status: 0, code: 'network_error', retryable: true };
  }
  if (!upstream.ok || upstream.body === null) return refusalOf(upstream, model);
  if (!stream) {
    let answer: unknown;
    try {
      answer = await upstream.json();
    } catch {
      return { kind: 'refused', model, status: 502, code: 'bad_upstream_body', retryable: true };
    }
    if (!isRecord(answer)) return { kind: 'refused', model, status: 502, code: 'bad_upstream_body', retryable: true };
    const text = textOfAnswer(answer);
    if (text.trim() === '') return { kind: 'empty', model };
    return { kind: 'ok', model, text, json: answer, events: null };
  }
  const events: string[] = [];
  try {
    for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) events.push(event);
  } catch {
    return { kind: 'refused', model, status: 502, code: 'bad_upstream_stream', retryable: true };
  }
  const text = textOfEvents(events);
  if (text.trim() === '') return { kind: 'empty', model };
  return { kind: 'ok', model, text, json: null, events };
}

/**
 * The free models that may serve a prompt, best first: the brief's rating order. A model that
 * refused an earlier prompt is tried again at its rated place: a free-pool 429 is usually a
 * brief burst, and demoting the best model would cost quality to save at most two calls.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const eligible = freeModels(brief).filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  return [...eligible].sort((a, b) => b.rating_milli - a.rating_milli);
}

/** What serving one prompt found: the attempt to answer with and the refusals it recovered from. */
interface Served {
  attempt: Attempt & { kind: 'ok' };
  failures: UpstreamFailure[];
}

/** Serve one prompt: try the best models first, retry a refusal once, fail over, and keep the first
 * attempt whose text carries the prompt's token — or the first usable attempt when none does. */
async function serve(
  wire: 'openai' | 'anthropic',
  body: Json,
  stream: boolean,
  requires: Requirement,
  token: string | null,
): Promise<Served | null> {
  const failures: UpstreamFailure[] = [];
  let fallback: (Attempt & { kind: 'ok' }) | null = null;
  for (const model of candidatesFor(requires).slice(0, MAX_MODELS_PER_PROMPT)) {
    for (let attempt = 0; attempt < MAX_ATTEMPTS_PER_MODEL; attempt++) {
      const result = wire === 'anthropic' ? await dialAnthropic(model.id, body, stream) : await dial(model.id, body, stream);
      if (result.kind === 'ok') {
        if (token === null || result.text.includes(token)) return { attempt: result, failures };
        if (fallback === null) fallback = result;
        break;
      }
      if (result.kind === 'empty') break;
      // A refusal: record it for the trail header, and retry once when it may pass.
      failures.push({ status: result.status, code: result.code, model: result.model });
      if (result.status === 401) return fallback === null ? null : { attempt: fallback, failures };
      if (!result.retryable || attempt === MAX_ATTEMPTS_PER_MODEL - 1) break;
      await sleep(RETRY_DELAY_MS);
    }
  }
  return fallback === null ? null : { attempt: fallback, failures };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  if (typeof body.max_tokens !== 'number' && typeof body.max_completion_tokens !== 'number') body.max_tokens = 512;
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const served = await serve('openai', body, stream, requires, tokenOf(body));
  if (served === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'every free model refused this prompt');
  }
  const { attempt, failures } = served;
  const headers = trailHeader(failures);
  if (stream && attempt.events !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for (const frame of attempt.events) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  if (attempt.json === null) return sendError(res, 'openai', 502, 'upstream_error', 'the gateway answer had no body');
  sendJson(res, 200, openAIJsonWithModel(attempt.json, attempt.model), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  if (typeof body.max_tokens !== 'number') body.max_tokens = 512;
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const served = await serve('anthropic', body, stream, requires, tokenOf(body));
  if (served === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'every free model refused this prompt');
  }
  const { attempt, failures } = served;
  const headers = trailHeader(failures);
  if (stream && attempt.events !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for (const event of attempt.events) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  if (attempt.json === null) return sendError(res, 'anthropic', 502, 'api_error', 'the gateway answer had no body');
  sendJson(res, 200, openAIToAnthropic(attempt.json, attempt.model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
