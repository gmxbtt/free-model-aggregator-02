/**
 * OpenRouter client for the free-model aggregator.
 *
 * The client is OpenAI-compatible, so the same code dials OpenRouter
 * directly and the grader's gateway: the gateway is an OpenAI-compatible
 * endpoint, and its address comes from the environment. The key is read
 * from the environment only. It is never written to a file, a log or
 * the repository.
 *
 * Verified live 2026-11-17: a chat completion on
 * cohere/north-mini-code:free returned the requested token, cost $0.
 * Small free models spend much of their budget on reasoning, so always
 * send max_tokens and give them at least 256.
 */
import OpenAI from "openai";

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export type ChatResult = {
  text: string;
  model: string;
  finishReason: string | null;
  usage: { promptTokens: number; completionTokens: number; totalTokens: number } | null;
};

export type UpstreamFailure = { status: number; code: string; model: string | null };

export class OpenRouterError extends Error {
  status: number | null;
  code: string;
  retryable: boolean;

  constructor(message: string, status: number | null, code: string, retryable: boolean) {
    super(message);
    this.name = "OpenRouterError";
    this.status = status;
    this.code = code;
    this.retryable = retryable;
  }
}

/** Default free models, in the order to try them. Verified live: the first one answers. */
export const DEFAULT_FREE_MODELS = [
  "cohere/north-mini-code:free",
  "liquid/lfm-3-1.7b:free",
  "google/gemma-4-26b-a4b-it:free",
];

export function createClient(): OpenAI {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new OpenRouterError(
      "OPENROUTER_API_KEY is not set. Create a key at https://openrouter.ai/keys and export it before running.",
      null,
      "missing_key",
      false,
    );
  }
  return new OpenAI({
    baseURL: process.env.OPENAI_BASE_URL ?? "https://openrouter.ai/api/v1",
    apiKey,
  });
}

function toError(error: unknown, model: string): OpenRouterError {
  if (error instanceof OpenAI.APIError) {
    const status = error.status ?? null;
    const code = (error as { code?: string }).code ?? error.name ?? "api_error";
    if (status === 401 || status === 403) {
      return new OpenRouterError(
        `OpenRouter rejected the key (HTTP ${status}). Check that OPENROUTER_API_KEY holds a valid key.`,
        status,
        code,
        false,
      );
    }
    if (status === 400) {
      return new OpenRouterError(
        `OpenRouter refused the request (HTTP 400): ${error.message}`,
        status,
        code,
        false,
      );
    }
    if (status === 429 || (status !== null && status >= 500)) {
      return new OpenRouterError(
        `Upstream ${status === 429 ? "rate limit" : "error"} on ${model}: ${error.message}`,
        status,
        code,
        true,
      );
    }
    return new OpenRouterError(`OpenRouter error (HTTP ${status}): ${error.message}`, status, code, false);
  }
  if (error instanceof Error) {
    return new OpenRouterError(
      `Network or client error calling ${model}: ${error.message}`,
      null,
      "network_error",
      true,
    );
  }
  return new OpenRouterError(`Unknown error calling ${model}`, null, "unknown", true);
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** One chat completion on one model. Always sends max_tokens: a request without an output cap is refused. */
export async function chatCompletion(
  client: OpenAI,
  args: { model: string; messages: ChatMessage[]; maxTokens?: number },
): Promise<ChatResult> {
  const maxTokens = args.maxTokens ?? 256;
  try {
    const response = await client.chat.completions.create({
      model: args.model,
      messages: args.messages,
      max_tokens: maxTokens,
    });
    const choice = response.choices[0];
    const text = choice?.message?.content ?? "";
    return {
      text,
      model: response.model,
      finishReason: choice?.finish_reason ?? null,
      usage: response.usage
        ? {
            promptTokens: response.usage.prompt_tokens,
            completionTokens: response.usage.completion_tokens,
            totalTokens: response.usage.total_tokens,
          }
        : null,
    };
  } catch (error) {
    throw toError(error, args.model);
  }
}

/**
 * Try each model in order until one returns a reply with visible text.
 * A retryable failure (429 or 5xx) is retried once on the same model,
 * then the next model is tried. Returns the first success plus the trail
 * of failures, in the order they happened.
 */
export async function chatWithFailover(
  client: OpenAI,
  args: { models?: string[]; messages: ChatMessage[]; maxTokens?: number },
): Promise<{ result: ChatResult; failures: UpstreamFailure[] }> {
  const models = args.models ?? DEFAULT_FREE_MODELS;
  const failures: UpstreamFailure[] = [];
  for (const model of models) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const result = await chatCompletion(client, { model, messages: args.messages, maxTokens: args.maxTokens });
        if (result.text.trim().length > 0) {
          return { result, failures };
        }
        failures.push({ status: 200, code: "empty_reply", model });
        break;
      } catch (error) {
        const failure = error instanceof OpenRouterError
          ? error
          : new OpenRouterError(String(error), null, "unknown", true);
        failures.push({ status: failure.status ?? 0, code: failure.code, model });
        if (!failure.retryable || attempt === 1) break;
        await sleep(1000 * (attempt + 1));
      }
    }
  }
  const last = failures[failures.length - 1];
  throw new OpenRouterError(
    `All ${models.length} free models failed. Last failure: ${last?.code ?? "unknown"} on ${last?.model ?? "unknown"}.`,
    last?.status ?? null,
    last?.code ?? "all_models_failed",
    false,
  );
}
