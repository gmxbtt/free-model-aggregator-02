/**
 * Smoke test: one working OpenRouter request, end to end.
 *
 * Sends a prompt that asks for a unique token, then checks the reply
 * carries that token back. Uses chatWithFailover, so a rate limited
 * model falls over to the next free model instead of failing the run.
 *
 * Run:  npm run smoke
 * Needs: OPENROUTER_API_KEY set in the environment.
 *         OPENROUTER_MODEL overrides the first model to try.
 */
import { chatWithFailover, createClient, DEFAULT_FREE_MODELS, OpenRouterError } from "../src/openrouter-client.ts";

const TOKEN = "SMOKE-OK-4711";

async function main(): Promise<void> {
  const firstModel = process.env.OPENROUTER_MODEL ?? DEFAULT_FREE_MODELS[0];
  const models = [firstModel, ...DEFAULT_FREE_MODELS.filter((m) => m !== firstModel)];
  let client;
  try {
    client = createClient();
  } catch (error) {
    if (error instanceof OpenRouterError) {
      console.error(`SMOKE FAILED: ${error.message}`);
      process.exit(1);
    }
    throw error;
  }
  try {
    const { result, failures } = await chatWithFailover(client, {
      models,
      messages: [{ role: "user", content: `Reply with exactly the token ${TOKEN} and nothing else.` }],
      maxTokens: 256,
    });
    if (!result.text.includes(TOKEN)) {
      console.error(`SMOKE FAILED: reply did not carry ${TOKEN}. Reply was: ${JSON.stringify(result.text)}`);
      process.exit(1);
    }
    if (failures.length > 0) {
      console.log(`Failovers before success: ${JSON.stringify(failures)}`);
    }
    console.log(`SMOKE OK: model=${result.model} finish=${result.finishReason} ` +
      `usage=${result.usage ? `${result.usage.promptTokens}+${result.usage.completionTokens}` : "n/a"} tokens`);
    console.log(`Reply: ${result.text}`);
  } catch (error) {
    if (error instanceof OpenRouterError) {
      console.error(`SMOKE FAILED: ${error.message}`);
      process.exit(1);
    }
    console.error(`SMOKE FAILED: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  }
}

main();
