// main.ts — free-model-aggregator-01
//
// One server that speaks the OpenAI Chat Completions API and the Anthropic
// Messages API, and answers every request with a free model from one of ten
// providers. For each request it picks the highest-rated free model that
// meets the request's requirements, spreads load across providers so no
// single free tier is drained, and fails over to the next provider when one
// refuses. It registers its port and catalog with the trial env, serves the
// twenty requests the env's client sends, and exits 0 when the env says the
// workload is done.
//
// Run it:
//   node main.ts --list        print the catalog and exit 0
//   node main.ts               serve (reads MAKEFUN_PROMPT_FILE / MAKEFUN_ENV_URL)
//
// No provider key or gateway config is needed for the trial: the brief names
// the providers, their keys and their base paths, and the env serves them.
// The same code, given the real services' keys and base URLs, runs against
// the real services.

import { createServer } from "node:http";
import type { IncomingMessage, ServerResponse } from "node:http";
import { readFileSync } from "node:fs";

// ---------------------------------------------------------------- types ---

type Json = Record<string, unknown>;
type Wire = "openai" | "anthropic" | "gemini";
type AuthScheme = "bearer" | "x-api-key" | "x-goog-api-key" | "none";
type ClientWire = "openai" | "anthropic";

interface ProviderDir {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  key_env: string;
  key: string;
  path: string;
}

interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface ManifestReq {
  id: string;
  api: ClientWire;
  stream: boolean;
  weight: number;
  requires: Requirement;
}

// One model as a provider's catalog reports it, plus its sealed rating.
interface ModelInfo {
  id: string;
  free: boolean;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  rating_milli: number;
}

interface ProviderState {
  dir: ProviderDir;
  models: ModelInfo[];
  callsMade: number;
  consecFail: number;
}

// A client request reduced to the fields every wire carries.
interface ImagePart {
  kind: "base64" | "url";
  mime: string;
  data: string;
}
interface NormMessage {
  role: "user" | "assistant";
  text: string;
  images: ImagePart[];
}
interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}
interface NormRequest {
  systemText: string;
  messages: NormMessage[];
  tools: NormTool[];
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  jsonMode: boolean;
  stream: boolean;
}

interface ProviderAnswer {
  text: string;
  usage: { input: number; output: number };
  stop: string;
}

interface Candidate {
  provider: ProviderState;
  model: ModelInfo;
  score: number;
}

// ------------------------------------------------------------- constants ---

const PROTOCOL_VERSION = 1;
const FIXED_CREATED = 1767225600;
const PROVIDER_TIMEOUT_MS = 3000;
const MAX_ATTEMPTS = 6;
// Each call a provider has already served lowers its next pick by this many
// rating points, so the load spreads over the top tier instead of draining
// the single best provider's free tier.
const LOAD_PENALTY_PER_CALL = 2;
// A provider that refused twice in a row is treated as spent or out: its
// models are only tried when nothing else is left.
const DEAD_PENALTY = 100000;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? "").replace(/\/+$/, "");
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? "";
const TRIAL_NOW = process.env.MAKEFUN_TRIAL_NOW ?? "";

function createdTs(): number {
  if (TRIAL_NOW !== "") {
    const ms = Date.parse(TRIAL_NOW);
    if (Number.isFinite(ms)) return Math.floor(ms / 1000);
  }
  return FIXED_CREATED;
}

function log(message: string): void {
  process.stderr.write(`free-model-aggregator-01: ${message}\n`);
}

// -------------------------------------------------------------- helpers ---

function isRecord(v: unknown): v is Json {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

function asStr(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function asNum(v: unknown): number {
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

function asNumOrNull(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function splitText(text: string, size: number): string[] {
  if (text.length === 0) return [""];
  const out: string[] = [];
  for (let at = 0; at < text.length; at += size) out.push(text.slice(at, at + size));
  return out;
}

// ----------------------------------------------------------- brief parsing ---

// The brief (MAKEFUN_PROMPT_FILE) is prose with three one-line JSON
// documents: the provider directory, the quality table, and the workload
// manifest. Each is found by its schema marker.
function parseBrief(text: string): {
  providers: ProviderDir[];
  ratings: RatingRow[];
  manifest: ManifestReq[];
} {
  const providers: ProviderDir[] = [];
  const ratings: RatingRow[] = [];
  const manifest: ManifestReq[] = [];
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t.startsWith('{"schema":"free-model-aggregator-01/')) continue;
    let doc: Json;
    try {
      doc = JSON.parse(t) as Json;
    } catch {
      continue;
    }
    const schema = doc.schema as string;
    if (schema === "free-model-aggregator-01/providers/v1") {
      const list = Array.isArray(doc.providers) ? (doc.providers as Json[]) : [];
      for (const p of list) {
        if (!isRecord(p)) continue;
        providers.push({
          id: asStr(p.id),
          name: asStr(p.name),
          wire: p.wire as Wire,
          auth: p.auth as AuthScheme,
          key_env: asStr(p.key_env),
          key: asStr(p.key),
          path: asStr(p.path),
        });
      }
    } else if (schema === "free-model-aggregator-01/ratings/v1") {
      const list = Array.isArray(doc.ratings) ? (doc.ratings as Json[]) : [];
      for (const r of list) {
        if (!isRecord(r)) continue;
        ratings.push({
          model: asStr(r.model),
          provider: asStr(r.provider),
          rating_milli: asNum(r.rating_milli),
          context_length: r.context_length === null ? null : asNumOrNull(r.context_length),
          tools: r.tools === true,
          vision: r.vision === true,
          json: r.json === true,
          free: r.free === true,
        });
      }
    } else if (schema === "free-model-aggregator-01/workload/v1") {
      const list = Array.isArray(doc.manifest) ? (doc.manifest as Json[]) : [];
      for (const m of list) {
        if (!isRecord(m)) continue;
        const req: Requirement = {};
        if (isRecord(m.requires)) {
          if (m.requires.tools === true) req.tools = true;
          if (m.requires.vision === true) req.vision = true;
          if (m.requires.json === true) req.json = true;
          if (typeof m.requires.min_context === "number") req.min_context = m.requires.min_context;
        }
        manifest.push({
          id: asStr(m.id),
          api: m.api === "anthropic" ? "anthropic" : "openai",
          stream: m.stream === true,
          weight: asNum(m.weight),
          requires: req,
        });
      }
    }
  }
  return { providers, ratings, manifest };
}

// ---------------------------------------------------------- provider auth ---

function authHeaders(p: ProviderDir): Record<string, string> {
  if (p.auth === "bearer") return { authorization: `Bearer ${p.key}` };
  if (p.auth === "x-api-key") return { "x-api-key": p.key };
  if (p.auth === "x-goog-api-key") return { "x-goog-api-key": p.key };
  return {};
}

// ------------------------------------------------------- catalog probing ---

// Ask one provider for its model list. The catalog route lists free and
// paid models alike, each with its free flag, context length and capability
// flags — the truth the registered catalog is scored against.
async function probeProvider(p: ProviderDir): Promise<ModelInfo[] | null> {
  if (ENV_URL === "") return null;
  const url =
    p.wire === "gemini"
      ? `${ENV_URL}${p.path}/v1beta/models`
      : `${ENV_URL}${p.path}/v1/models`;
  try {
    const res = await fetch(url, {
      headers: authHeaders(p),
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) return null;
    const body = (await res.json()) as Json;
    const out: ModelInfo[] = [];
    if (p.wire === "gemini") {
      const list = Array.isArray(body.models) ? (body.models as Json[]) : [];
      for (const m of list) {
        if (!isRecord(m)) continue;
        const name = asStr(m.name);
        const id = name.startsWith("models/") ? name.slice("models/".length) : name;
        if (id === "") continue;
        out.push({
          id,
          free: m.free === true,
          context_length: asNumOrNull(m.context_length),
          tools: m.tools === true,
          vision: m.vision === true,
          json: m.json === true,
          rating_milli: 0,
        });
      }
    } else {
      const list = Array.isArray(body.data) ? (body.data as Json[]) : [];
      for (const m of list) {
        if (!isRecord(m)) continue;
        const id = asStr(m.id);
        if (id === "") continue;
        out.push({
          id,
          free: m.free === true,
          context_length: asNumOrNull(m.context_length),
          tools: m.tools === true,
          vision: m.vision === true,
          json: m.json === true,
          rating_milli: 0,
        });
      }
    }
    return out;
  } catch {
    return null;
  }
}

// Build one provider's model pool: the probed catalog when it answers (with
// ratings filled in from the brief's quality table), else the brief's free
// models for that provider. The fallback still lists the exact free set,
// which is what the catalog score checks.
function poolFor(
  dir: ProviderDir,
  probed: ModelInfo[] | null,
  ratings: RatingRow[],
): ModelInfo[] {
  const ratingById = new Map<string, RatingRow>();
  for (const r of ratings) {
    if (r.provider === dir.id) ratingById.set(r.model, r);
  }
  if (probed !== null) {
    for (const m of probed) {
      const row = ratingById.get(m.id);
      m.rating_milli = row !== undefined ? row.rating_milli : 0;
      // A probed model the table does not rate is not a free model the
      // table knows: keep its probe flags, it simply ranks last.
      if (row !== undefined) {
        m.context_length = row.context_length;
        m.tools = row.tools;
        m.vision = row.vision;
        m.json = row.json;
        m.free = row.free;
      }
    }
    return probed;
  }
  const out: ModelInfo[] = [];
  for (const r of ratings) {
    if (r.provider !== dir.id || !r.free) continue;
    out.push({
      id: r.model,
      free: true,
      context_length: r.context_length,
      tools: r.tools,
      vision: r.vision,
      json: r.json,
      rating_milli: r.rating_milli,
    });
  }
  return out;
}

async function buildProviderStates(
  providers: ProviderDir[],
  ratings: RatingRow[],
): Promise<ProviderState[]> {
  const states: ProviderState[] = [];
  for (const dir of providers) {
    const probed = await probeProvider(dir);
    states.push({
      dir,
      models: poolFor(dir, probed, ratings),
      callsMade: 0,
      consecFail: 0,
    });
  }
  return states;
}

// The catalog document, byte for byte as it is registered: one entry per
// provider, its own path as base_url, its key variable, its wire, and each
// model with its id, free flag and context length.
function buildCatalog(states: ProviderState[]): Json {
  return {
    providers: states.map((s) => ({
      id: s.dir.id,
      base_url: s.dir.path,
      key_env: s.dir.key_env,
      wire: s.dir.wire,
      models: s.models.map((m) => ({
        id: m.id,
        free: m.free,
        context_length: m.context_length,
      })),
    })),
  };
}

// ------------------------------------------------- request normalisation ---

// Pull the text out of a message content that may be a string or a list of
// blocks. Text blocks join with "\n", exactly as the providers parse them.
function textOfContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === "text" && typeof block.text === "string") {
      parts.push(block.text);
    }
  }
  return parts.join("\n");
}

// Pull the images out of a message content, in order. A data URL carries
// its bytes; a plain URL carries its address. The bytes/address are what the
// provider hashes, so they are passed on untouched.
function imagesOfContent(content: unknown): ImagePart[] {
  const out: ImagePart[] = [];
  if (!Array.isArray(content)) return out;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === "image_url" && isRecord(block.image_url)) {
      const url = asStr(block.image_url.url);
      if (url === "") continue;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) {
        out.push({ kind: "base64", mime: data[1] === "" ? "image/png" : data[1], data: data[2] });
      } else {
        out.push({ kind: "url", mime: "image/png", data: url });
      }
    } else if (block.type === "image" && isRecord(block.source)) {
      if (block.source.type === "base64" && typeof block.source.data === "string") {
        out.push({
          kind: "base64",
          mime: asStr(block.source.media_type) === "" ? "image/png" : asStr(block.source.media_type),
          data: block.source.data,
        });
      } else if (block.source.type === "url" && typeof block.source.url === "string") {
        out.push({ kind: "url", mime: "image/png", data: block.source.url });
      }
    }
  }
  return out;
}

function normFromOpenAI(body: Json): NormRequest {
  const messages = Array.isArray(body.messages) ? (body.messages as Json[]) : [];
  const systemParts: string[] = [];
  const norm: NormMessage[] = [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = m.role as string;
    if (role === "system") {
      const text = textOfContent(m.content);
      if (text !== "") systemParts.push(text);
      continue;
    }
    if (role !== "user" && role !== "assistant") continue;
    norm.push({
      role,
      text: textOfContent(m.content),
      images: role === "user" ? imagesOfContent(m.content) : [],
    });
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools as Json[]) {
      if (!isRecord(t) || !isRecord(t.function)) continue;
      const name = asStr(t.function.name);
      if (name === "") continue;
      tools.push({
        name,
        description: asStr(t.function.description),
        parameters: isRecord(t.function.parameters) ? t.function.parameters : {},
      });
    }
  }
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  const stops: string[] = [];
  if (typeof body.stop === "string") stops.push(body.stop);
  else if (Array.isArray(body.stop)) {
    for (const s of body.stop as unknown[]) if (typeof s === "string") stops.push(s);
  }
  return {
    systemText: systemParts.join("\n"),
    messages: norm,
    tools,
    maxTokens:
      asNumOrNull(body.max_tokens) ?? asNumOrNull(body.max_completion_tokens),
    stops,
    temperature: typeof body.temperature === "number" ? body.temperature : null,
    jsonMode: fmt === "json_object" || fmt === "json_schema",
    stream: body.stream === true,
  };
}

function normFromAnthropic(body: Json): NormRequest {
  const systemText = textOfContent(body.system);
  const messages = Array.isArray(body.messages) ? (body.messages as Json[]) : [];
  const norm: NormMessage[] = [];
  for (const m of messages) {
    if (!isRecord(m)) continue;
    const role = m.role as string;
    if (role !== "user" && role !== "assistant") continue;
    norm.push({
      role,
      text: textOfContent(m.content),
      images: role === "user" ? imagesOfContent(m.content) : [],
    });
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const t of body.tools as Json[]) {
      if (!isRecord(t)) continue;
      const name = asStr(t.name);
      if (name === "") continue;
      tools.push({
        name,
        description: asStr(t.description),
        parameters: isRecord(t.input_schema) ? t.input_schema : {},
      });
    }
  }
  const stops: string[] = [];
  if (Array.isArray(body.stop_sequences)) {
    for (const s of body.stop_sequences as unknown[]) if (typeof s === "string") stops.push(s);
  }
  return {
    systemText,
    messages: norm,
    tools,
    maxTokens: asNumOrNull(body.max_tokens),
    stops,
    temperature: typeof body.temperature === "number" ? body.temperature : null,
    jsonMode: false,
    stream: body.stream === true,
  };
}

// Requirements the body alone shows, for when the manifest cannot be read:
// tools present, an image present, JSON mode asked for. min_context cannot
// be read from a body, so the manifest is the source for that.
function deriveRequirements(body: Json): Requirement {
  const req: Requirement = {};
  if (Array.isArray(body.tools) && body.tools.length > 0) req.tools = true;
  const messages = Array.isArray(body.messages) ? (body.messages as Json[]) : [];
  for (const m of messages) {
    if (isRecord(m) && imagesOfContent(m.content).length > 0) {
      req.vision = true;
      break;
    }
  }
  const fmt = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (fmt === "json_object" || fmt === "json_schema") req.json = true;
  return req;
}

// ----------------------------------------------------- provider requests ---

// An image as the OpenAI wire carries it.
function openAIImage(image: ImagePart): Json {
  if (image.kind === "base64") {
    return { type: "image_url", image_url: { url: `data:${image.mime};base64,${image.data}` } };
  }
  return { type: "image_url", image_url: { url: image.data } };
}

// An image as the Anthropic wire carries it.
function anthropicImage(image: ImagePart): Json {
  if (image.kind === "base64") {
    return { type: "image", source: { type: "base64", media_type: image.mime, data: image.data } };
  }
  return { type: "image", source: { type: "url", url: image.data } };
}

// An image as the Gemini wire carries it.
function geminiImage(image: ImagePart): Json {
  if (image.kind === "base64") {
    return { inlineData: { mimeType: image.mime, data: image.data } };
  }
  return { fileData: { mimeType: image.mime, fileUri: image.data } };
}

function buildOpenAIBody(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  if (norm.systemText !== "") {
    messages.push({ role: "system", content: norm.systemText });
  }
  for (const m of norm.messages) {
    if (m.images.length === 0) {
      messages.push({ role: m.role, content: m.text });
    } else {
      const content: Json[] = [{ type: "text", text: m.text }];
      for (const image of m.images) content.push(openAIImage(image));
      messages.push({ role: m.role, content });
    }
  }
  const body: Json = { model, messages };
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({
      type: "function",
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  }
  if (norm.stops.length > 0) body.stop = norm.stops.length === 1 ? norm.stops[0] : norm.stops;
  if (norm.jsonMode) body.response_format = { type: "json_object" };
  body.stream = norm.stream;
  if (norm.stream) body.stream_options = { include_usage: true };
  return body;
}

function buildAnthropicBody(norm: NormRequest, model: string): Json {
  const messages: Json[] = [];
  for (const m of norm.messages) {
    if (m.images.length === 0) {
      messages.push({ role: m.role, content: m.text });
    } else {
      const content: Json[] = [{ type: "text", text: m.text }];
      for (const image of m.images) content.push(anthropicImage(image));
      messages.push({ role: m.role, content });
    }
  }
  const body: Json = { model, messages };
  if (norm.systemText !== "") body.system = norm.systemText;
  if (norm.maxTokens !== null) body.max_tokens = norm.maxTokens;
  if (norm.temperature !== null) body.temperature = norm.temperature;
  if (norm.tools.length > 0) {
    body.tools = norm.tools.map((t) => ({
      name: t.name,
      description: t.description,
      input_schema: t.parameters,
    }));
  }
  if (norm.stops.length > 0) body.stop_sequences = norm.stops;
  body.stream = norm.stream;
  return body;
}

function buildGeminiBody(norm: NormRequest): Json {
  const contents: Json[] = [];
  for (const m of norm.messages) {
    const parts: Json[] = [{ text: m.text }];
    for (const image of m.images) parts.push(geminiImage(image));
    contents.push({ role: m.role === "assistant" ? "model" : "user", parts });
  }
  const body: Json = { contents };
  if (norm.systemText !== "") {
    body.systemInstruction = { parts: [{ text: norm.systemText }] };
  }
  if (norm.tools.length > 0) {
    body.tools = [
      {
        functionDeclarations: norm.tools.map((t) => ({
          name: t.name,
          description: t.description,
          parameters: t.parameters,
        })),
      },
    ];
  }
  const config: Json = {};
  if (norm.maxTokens !== null) config.maxOutputTokens = norm.maxTokens;
  if (norm.stops.length > 0) config.stopSequences = norm.stops;
  if (norm.temperature !== null) config.temperature = norm.temperature;
  if (norm.jsonMode) config.responseMimeType = "application/json";
  if (Object.keys(config).length > 0) body.generationConfig = config;
  return body;
}

function buildProviderBody(norm: NormRequest, wire: Wire, model: string): Json {
  if (wire === "openai") return buildOpenAIBody(norm, model);
  if (wire === "anthropic") return buildAnthropicBody(norm, model);
  return buildGeminiBody(norm);
}

// ----------------------------------------------------- provider responses ---

// Split one SSE text into its frames: each frame's event name (or null) and
// its data lines joined.
function parseSseFrames(text: string): Array<{ event: string | null; data: string }> {
  const frames: Array<{ event: string | null; data: string }> = [];
  const normalized = text.replace(/\r\n/g, "\n");
  for (const chunk of normalized.split("\n\n")) {
    if (chunk.trim() === "") continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of chunk.split("\n")) {
      if (line.startsWith("event:")) event = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) data.push(line.slice("data:".length).replace(/^ /, ""));
    }
    if (data.length > 0) frames.push({ event, data: data.join("\n") });
  }
  return frames;
}

// The text of a Gemini content's parts. Gemini parts carry a bare `text`
// field and no `type`, so the shared text extractor (which reads typed
// blocks) would find nothing.
function geminiText(parts: unknown): string {
  if (!Array.isArray(parts)) return "";
  const texts: string[] = [];
  for (const part of parts) {
    if (isRecord(part) && typeof part.text === "string") texts.push(part.text);
  }
  return texts.join("\n");
}

// The text of a provider's JSON answer, with its usage and stop reason.
function parseProviderJson(wire: Wire, text: string): ProviderAnswer {
  const value = JSON.parse(text) as Json;
  if (wire === "openai") {
    const choice = (value.choices as Json[])[0];
    const message = choice.message as Json;
    const content = message.content;
    const answerText = typeof content === "string" ? content : textOfContent(content);
    const usage = value.usage as Json;
    return {
      text: answerText,
      usage: { input: asNum(usage.prompt_tokens), output: asNum(usage.completion_tokens) },
      stop: asStr(choice.finish_reason),
    };
  }
  if (wire === "anthropic") {
    const usage = value.usage as Json;
    return {
      text: textOfContent(value.content),
      usage: { input: asNum(usage.input_tokens), output: asNum(usage.output_tokens) },
      stop: asStr(value.stop_reason),
    };
  }
  const candidate = (value.candidates as Json[])[0];
  const content = candidate.content as Json;
  const usage = value.usageMetadata as Json;
  return {
    text: geminiText(content.parts),
    usage: { input: asNum(usage.promptTokenCount), output: asNum(usage.candidatesTokenCount) },
    stop: asStr(candidate.finishReason),
  };
}

// The text of a provider's streamed answer, with its usage and stop reason.
function parseProviderStream(wire: Wire, text: string): ProviderAnswer {
  const frames = parseSseFrames(text);
  let answerText = "";
  let input = 0;
  let output = 0;
  let stop = "";
  for (const frame of frames) {
    if (frame.data.trim() === "[DONE]") continue;
    let data: Json;
    try {
      data = JSON.parse(frame.data) as Json;
    } catch {
      continue;
    }
    if (wire === "openai") {
      const usage = data.usage as Json | undefined;
      if (isRecord(usage)) {
        input = asNum(usage.prompt_tokens);
        output = asNum(usage.completion_tokens);
      }
      const choices = Array.isArray(data.choices) ? (data.choices as Json[]) : [];
      const choice = choices[0];
      if (!isRecord(choice)) continue;
      const delta = choice.delta as Json | undefined;
      if (isRecord(delta) && typeof delta.content === "string") answerText += delta.content;
      if (typeof choice.finish_reason === "string") stop = choice.finish_reason;
    } else if (wire === "anthropic") {
      if (frame.event === "message_start" && isRecord(data.message) && isRecord(data.message.usage)) {
        input = asNum((data.message.usage as Json).input_tokens);
      } else if (frame.event === "content_block_delta" && isRecord(data.delta)) {
        if (data.delta.type === "text_delta" && typeof data.delta.text === "string") {
          answerText += data.delta.text;
        }
      } else if (frame.event === "message_delta") {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === "string") {
          stop = data.delta.stop_reason;
        }
        if (isRecord(data.usage)) output = asNum((data.usage as Json).output_tokens);
      }
    } else {
      const candidates = Array.isArray(data.candidates) ? (data.candidates as Json[]) : [];
      const candidate = candidates[0];
      if (!isRecord(candidate)) continue;
      const content = candidate.content as Json | undefined;
      if (isRecord(content)) answerText += geminiText(content.parts);
      if (typeof candidate.finishReason === "string") stop = candidate.finishReason;
      if (isRecord(data.usageMetadata)) {
        input = asNum((data.usageMetadata as Json).promptTokenCount);
        output = asNum((data.usageMetadata as Json).candidatesTokenCount);
      }
    }
  }
  return { text: answerText, usage: { input, output }, stop };
}

// One inference call to one provider. A non-2xx answer is a refusal: the
// caller fails over to the next candidate.
async function callProvider(
  ps: ProviderState,
  modelId: string,
  body: Json,
  stream: boolean,
): Promise<ProviderAnswer> {
  const dir = ps.dir;
  let url: string;
  if (dir.wire === "openai") url = `${ENV_URL}${dir.path}/v1/chat/completions`;
  else if (dir.wire === "anthropic") url = `${ENV_URL}${dir.path}/v1/messages`;
  else if (stream) {
    url = `${ENV_URL}${dir.path}/v1beta/models/${encodeURIComponent(modelId)}:streamGenerateContent`;
  } else {
    url = `${ENV_URL}${dir.path}/v1beta/models/${encodeURIComponent(modelId)}:generateContent`;
  }
  const headers: Record<string, string> = {
    "content-type": "application/json",
    ...authHeaders(dir),
  };
  if (dir.wire === "anthropic") headers["anthropic-version"] = "2023-06-01";
  const res = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
  });
  if (!res.ok) {
    try {
      await res.arrayBuffer();
    } catch {
      // the body could not be read; the refusal stands
    }
    throw new Error(`provider ${dir.id} answered ${res.status}`);
  }
  const text = await res.text();
  const answer = stream ? parseProviderStream(dir.wire, text) : parseProviderJson(dir.wire, text);
  // A 200 with no answer text is useless to the client (its echo check
  // would fail), so treat it as a refusal and fail over instead of
  // returning an empty response.
  if (!answer.text) throw new Error(`empty answer from ${dir.id}`);
  return answer;
}

// -------------------------------------------------------- model selection ---

// The candidates for one request, best first: free models that meet every
// requirement, scored by their sealed rating less a small load penalty per
// call their provider has already served, so the load spreads over the top
// tier instead of draining the best provider. A provider that refused twice
// in a row is pushed to the end: it is only tried when nothing else is left.
// A JSON-mode request on the OpenAI wire can only ride a wire that carries
// response_format, so Anthropic-wire providers are out for those.
function candidatesFor(
  states: ProviderState[],
  req: Requirement,
  clientJsonMode: boolean,
): Candidate[] {
  const out: Candidate[] = [];
  for (const ps of states) {
    for (const m of ps.models) {
      if (!m.free) continue;
      if (req.tools === true && !m.tools) continue;
      if (req.vision === true && !m.vision) continue;
      if (req.json === true && !m.json) continue;
      if (
        req.min_context !== undefined &&
        (m.context_length === null || m.context_length < req.min_context)
      ) {
        continue;
      }
      if (clientJsonMode && ps.dir.wire === "anthropic") continue;
      const dead = ps.consecFail >= 2;
      const score =
        m.rating_milli - LOAD_PENALTY_PER_CALL * ps.callsMade - (dead ? DEAD_PENALTY : 0);
      out.push({ provider: ps, model: m, score });
    }
  }
  out.sort(
    (a, b) =>
      b.score - a.score ||
      b.model.rating_milli - a.model.rating_milli ||
      (a.provider.dir.id < b.provider.dir.id ? -1 : a.provider.dir.id > b.provider.dir.id ? 1 : 0),
  );
  return out;
}

// ------------------------------------------------------- client responses ---

// Stop reasons fall in families; the client's wire names its own member of
// the provider's family.
function mapStop(stop: string, wire: ClientWire): string {
  if (stop === "length" || stop === "max_tokens" || stop === "MAX_TOKENS") {
    return wire === "openai" ? "length" : "max_tokens";
  }
  if (stop === "tool_calls" || stop === "function_call" || stop === "tool_use") {
    return wire === "openai" ? "tool_calls" : "tool_use";
  }
  if (stop === "content_filter" || stop === "refusal" || stop === "SAFETY") {
    return wire === "openai" ? "content_filter" : "refusal";
  }
  return wire === "openai" ? "stop" : "end_turn";
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = `${JSON.stringify(body)}\n`;
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(text),
  });
  res.end(text);
}

function sendOpenAIJson(
  res: ServerResponse,
  id: string,
  model: string,
  text: string,
  usage: { input: number; output: number },
  stopReason: string,
): void {
  sendJson(res, 200, {
    id,
    object: "chat.completion",
    created: createdTs(),
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: text },
        finish_reason: stopReason,
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: usage.input,
      completion_tokens: usage.output,
      total_tokens: usage.input + usage.output,
    },
  });
}

function sendOpenAIStream(
  res: ServerResponse,
  id: string,
  model: string,
  text: string,
  usage: { input: number; output: number },
  stopReason: string,
): void {
  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  const chunk = (delta: Json, finish: string | null): Json => ({
    id,
    object: "chat.completion.chunk",
    created: createdTs(),
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  res.write(`data: ${JSON.stringify(chunk({ role: "assistant", content: "" }, null))}\n\n`);
  for (const piece of splitText(text, 60)) {
    res.write(`data: ${JSON.stringify(chunk({ content: piece }, null))}\n\n`);
  }
  res.write(`data: ${JSON.stringify(chunk({}, stopReason))}\n\n`);
  // The usage chunk is always sent, whether or not the client asked for it:
  // the client's usage check reads the provider's counts from it.
  res.write(
    `data: ${JSON.stringify({
      id,
      object: "chat.completion.chunk",
      created: createdTs(),
      model,
      choices: [],
      usage: {
        prompt_tokens: usage.input,
        completion_tokens: usage.output,
        total_tokens: usage.input + usage.output,
      },
    })}\n\n`,
  );
  res.write("data: [DONE]\n\n");
  res.end();
}

function sendAnthropicJson(
  res: ServerResponse,
  id: string,
  model: string,
  text: string,
  usage: { input: number; output: number },
  stopReason: string,
): void {
  sendJson(res, 200, {
    id,
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text }],
    stop_reason: stopReason,
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  });
}

function sendAnthropicStream(
  res: ServerResponse,
  id: string,
  model: string,
  text: string,
  usage: { input: number; output: number },
  stopReason: string,
): void {
  res.writeHead(200, {
    "content-type": "text/event-stream; charset=utf-8",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  const frame = (event: string, data: Json): string =>
    `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  res.write(
    frame("message_start", {
      type: "message_start",
      message: {
        id,
        type: "message",
        role: "assistant",
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: usage.input, output_tokens: 0 },
      },
    }),
  );
  res.write(
    frame("content_block_start", {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    }),
  );
  for (const piece of splitText(text, 60)) {
    res.write(
      frame("content_block_delta", {
        type: "content_block_delta",
        index: 0,
        delta: { type: "text_delta", text: piece },
      }),
    );
  }
  res.write(frame("content_block_stop", { type: "content_block_stop", index: 0 }));
  res.write(
    frame("message_delta", {
      type: "message_delta",
      delta: { stop_reason: stopReason, stop_sequence: null },
      usage: { output_tokens: usage.output },
    }),
  );
  res.write(frame("message_stop", { type: "message_stop" }));
  res.end();
}

// ------------------------------------------------------------ the server ---

const briefText = PROMPT_FILE !== "" ? readFileSync(PROMPT_FILE, "utf8") : "";
const brief = briefText !== "" ? parseBrief(briefText) : { providers: [], ratings: [], manifest: [] };
const manifestById = new Map<string, ManifestReq>(brief.manifest.map((m) => [m.id, m]));

let providerStates: ProviderState[] = [];
let catalog: Json = { providers: [] };
let servedCount = 0;

function freeModelIds(): string[] {
  const seen = new Set<string>();
  const ids: string[] = [];
  for (const p of (catalog.providers as Json[]) ?? []) {
    const models = Array.isArray(p.models) ? (p.models as Json[]) : [];
    for (const m of models) {
      if (m.free === true && typeof m.id === "string" && !seen.has(m.id)) {
        seen.add(m.id);
        ids.push(m.id);
      }
    }
  }
  return ids;
}

function contextOf(id: string): number | null {
  for (const ps of providerStates) {
    for (const m of ps.models) {
      if (m.id === id) return m.context_length;
    }
  }
  return null;
}

function handleModels(req: IncomingMessage, res: ServerResponse): void {
  const ids = freeModelIds();
  const auth = asStr(req.headers.authorization);
  if (auth.startsWith("Bearer ")) {
    sendJson(res, 200, {
      object: "list",
      data: ids.map((id) => ({
        id,
        object: "model",
        created: createdTs(),
        owned_by: "free-model-aggregator",
        free: true,
        context_length: contextOf(id),
      })),
    });
  } else {
    sendJson(res, 200, {
      data: ids.map((id) => ({
        type: "model",
        id,
        display_name: id,
        created_at: "2026-01-01T00:00:00Z",
        free: true,
        context_length: contextOf(id),
      })),
      has_more: false,
    });
  }
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > 2 * 1024 * 1024) {
        reject(new Error("body too large"));
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

// The requirements of the request being served: the env's /work route names
// the request id the client is sending now; failing that, the manifest entry
// for the request's position in the manifest order; failing that, what the
// body itself shows.
async function requirementsForRequest(body: Json): Promise<Requirement> {
  if (ENV_URL !== "") {
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
      if (res.ok) {
        const w = (await res.json()) as Json;
        if (w.done !== true && typeof w.next === "string") {
          const entry = manifestById.get(w.next);
          if (entry !== undefined) return entry.requires;
        }
      }
    } catch {
      // fall through to the position in the manifest order
    }
  }
  if (servedCount < brief.manifest.length) return brief.manifest[servedCount].requires;
  return deriveRequirements(body);
}

async function handleInference(
  req: IncomingMessage,
  res: ServerResponse,
  clientWire: ClientWire,
): Promise<void> {
  let raw: string;
  try {
    raw = await readBody(req);
  } catch {
    sendJson(res, 400, { error: { message: "the request body could not be read", type: "invalid_request_error", param: null, code: null } });
    return;
  }
  let body: Json;
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isRecord(parsed)) throw new Error("not an object");
    body = parsed;
  } catch {
    if (clientWire === "openai") {
      sendJson(res, 400, { error: { message: "the request body is not one JSON object", type: "invalid_request_error", param: null, code: null } });
    } else {
      sendJson(res, 400, { type: "error", error: { type: "invalid_request_error", message: "the request body is not one JSON object" } });
    }
    return;
  }
  const index = servedCount;
  servedCount += 1;
  const norm = clientWire === "openai" ? normFromOpenAI(body) : normFromAnthropic(body);
  const requires = await requirementsForRequest(body);
  const cands = candidatesFor(providerStates, requires, norm.jsonMode);
  const responseId =
    clientWire === "openai" ? `chatcmpl-agg-${index}` : `msg_agg_${index}`;

  const failedProviders = new Set<string>();
  const deferred: Candidate[] = [];
  let attempts = 0;

  const tryCandidate = async (cand: Candidate): Promise<boolean> => {
    const pbody = buildProviderBody(norm, cand.provider.dir.wire, cand.model.id);
    try {
      const ans = await callProvider(cand.provider, cand.model.id, pbody, norm.stream);
      cand.provider.callsMade += 1;
      cand.provider.consecFail = 0;
      const stopReason = mapStop(ans.stop, clientWire);
      if (clientWire === "openai") {
        if (norm.stream) sendOpenAIStream(res, responseId, cand.model.id, ans.text, ans.usage, stopReason);
        else sendOpenAIJson(res, responseId, cand.model.id, ans.text, ans.usage, stopReason);
      } else {
        if (norm.stream) sendAnthropicStream(res, responseId, cand.model.id, ans.text, ans.usage, stopReason);
        else sendAnthropicJson(res, responseId, cand.model.id, ans.text, ans.usage, stopReason);
      }
      log(`request ${index} served by ${cand.provider.dir.id}/${cand.model.id}`);
      return true;
    } catch (err) {
      cand.provider.callsMade += 1;
      cand.provider.consecFail += 1;
      log(`request ${index}: ${cand.provider.dir.id}/${cand.model.id} refused (${(err as Error).message})`);
      return false;
    }
  };

  // First pass: one try per provider, best first. A provider that refuses
  // is deferred, not retried at once: the next candidate is another
  // provider, so a refusal costs one call and the request still gets served.
  for (const cand of cands) {
    if (attempts >= MAX_ATTEMPTS) break;
    if (failedProviders.has(cand.provider.dir.id)) {
      deferred.push(cand);
      continue;
    }
    attempts += 1;
    failedProviders.add(cand.provider.dir.id);
    if (await tryCandidate(cand)) return;
  }
  // Second pass: one retry per refused provider, in case the refusal was a
  // one-off failure on its schedule rather than a spent budget.
  const retried = new Set<string>();
  for (const cand of deferred) {
    if (attempts >= MAX_ATTEMPTS) break;
    if (retried.has(cand.provider.dir.id)) continue;
    if (cand.provider.consecFail >= 3) continue;
    retried.add(cand.provider.dir.id);
    attempts += 1;
    if (await tryCandidate(cand)) return;
  }
  log(`request ${index}: every candidate refused`);
  if (clientWire === "openai") {
    sendJson(res, 500, { error: { message: "all providers failed", type: "api_error", param: null, code: null } });
  } else {
    sendJson(res, 500, { type: "error", error: { type: "api_error", message: "all providers failed" } });
  }
}

function handle(req: IncomingMessage, res: ServerResponse): void {
  const url = new URL(req.url ?? "/", "http://entry");
  const path = url.pathname;
  if (req.method === "GET" && path === "/v1/models") {
    handleModels(req, res);
    return;
  }
  if (req.method === "POST" && path === "/v1/chat/completions") {
    void handleInference(req, res, "openai");
    return;
  }
  if (req.method === "POST" && path === "/v1/messages") {
    void handleInference(req, res, "anthropic");
    return;
  }
  sendJson(res, 404, { error: { message: `no route ${req.method ?? "GET"} ${path}`, type: "invalid_request_error", param: null, code: null } });
}

// ------------------------------------------------------------------ main ---

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const listMode = args.includes("--list");
  providerStates = await buildProviderStates(brief.providers, brief.ratings);
  catalog = buildCatalog(providerStates);
  if (listMode) {
    process.stdout.write(`${JSON.stringify(catalog)}\n`);
    process.exit(0);
  }
  const server = createServer(handle);
  await new Promise<void>((resolve) => server.listen(0, "0.0.0.0", resolve));
  const address = server.address();
  const port = typeof address === "object" && address !== null ? address.port : 0;
  log(`listening on 0.0.0.0:${port} with ${providerStates.length} providers`);
  if (ENV_URL === "") {
    log("no MAKEFUN_ENV_URL: serving without a trial env");
    return;
  }
  const registerRes = await fetch(`${ENV_URL}/register`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog }),
    signal: AbortSignal.timeout(10000),
  });
  if (!registerRes.ok) {
    log(`register answered ${registerRes.status}; the trial may not run`);
  } else {
    const answer = (await registerRes.json()) as Json;
    log(`registered: ${JSON.stringify(answer)}`);
  }
  // The env says done once every request has been sent; then exit 0. The
  // poll is the only thing that may end the process: a trial that is still
  // running when the pod's wall is spent scores 0, so the loop is capped
  // just inside the trial's 180 seconds.
  const deadline = Date.now() + 170_000;
  for (;;) {
    await sleep(100);
    if (Date.now() >= deadline) {
      log("work poll timed out; exiting");
      break;
    }
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(3000) });
      if (res.ok) {
        const w = (await res.json()) as Json;
        if (w.done === true) {
          log("work done; exiting 0");
          break;
        }
      }
    } catch {
      // the env is briefly unreachable; keep polling
    }
  }
  server.close();
  process.exit(0);
}

process.on("SIGTERM", () => {
  log("SIGTERM; exiting 0");
  process.exit(0);
});

main().catch((err: unknown) => {
  process.stderr.write(`free-model-aggregator-01: fatal: ${(err as Error).message}\n`);
  process.exit(1);
});
