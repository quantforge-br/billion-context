// Anthropic-wire summary voice (opt-in, compress.anthropicSummaryAsSystem /
// BILI_ANTHROPIC_SUMMARY_AS_SYSTEM): a rendered acp_summary that sits between
// a user-voiced message and an assistant message leaves as its own
// mid-conversation role:"system" message instead of a text block merged into
// the preceding user message.
//
// Why: a fold that starts right after the conversation's first user message
// used to rewrite that message (it gained the summary block). Upstreams that
// key their prompt cache on the leading user run — gateways that relocate the
// caller's system prompt behind it — then re-wrote the whole system prompt on
// every fold. With the summary voiced as system the head stays byte-stable.
//
// Covered here: the knob tiers and config plumbing, the position rule of the
// pure transform, head/append stability on the wire, the round-2 adapter
// mirror, and one kernel-driven fold through prepareCountTokens.
import test from "node:test";
import assert from "node:assert/strict";
import { createCore, createInitialState, defaultConfig, isRenderedSummaryMessage, type CoreMessage } from "acp-kernel";
import { anthropicToCore, coreToAnthropic, type AnthropicRequestBody, type BiliMessage } from "acp-kernel/wire";
import { applyCompressSettings, mergeCompress, resolveRequestConfig } from "../src/compress-settings.ts";
import { parseCompressSettings } from "../src/config.ts";
import { resolveAnthropicSummaryAsSystem } from "../src/knobs.ts";
import { createAnthropicAdapter } from "../src/loop/index.ts";
import { prepareCountTokens } from "../src/server.ts";
import type { Session } from "../src/session.ts";
import { summariesToAnthropicSystem } from "../src/util.ts";
import { VALIDATORS } from "./wire-contract-fakes.ts";

const ENV = "BILI_ANTHROPIC_SUMMARY_AS_SYSTEM";
const SUM = "[Compressed conversation section] early work";

function withEnv(value: string | undefined, fn: () => void): void {
    const prev = process.env[ENV];
    if (value === undefined) delete process.env[ENV];
    else process.env[ENV] = value;
    try {
        fn();
    } finally {
        if (prev === undefined) delete process.env[ENV];
        else process.env[ENV] = prev;
    }
}

const user = (id: string, text: string): BiliMessage => ({ id, role: "user", contentType: "text", text });
const assistant = (id: string, text: string): BiliMessage => ({ id, role: "assistant", contentType: "text", text });
const summary = (n: number, text = `${SUM} ${String(n)}`): BiliMessage => ({ id: `acp_summary_b${String(n)}`, role: "system", contentType: "text", text });
const sidecar = (id: string, text: string): BiliMessage => ({ id, role: "system", contentType: "text", text, rawAnthropicMessage: { role: "system", content: [{ type: "text", text }] } });
const voice = (messages: BiliMessage[]): BiliMessage[] => summariesToAnthropicSystem(messages, (m) => isRenderedSummaryMessage(m));
const wire = (messages: BiliMessage[]): { role: string; content: unknown }[] => coreToAnthropic(voice(messages)) as { role: string; content: unknown }[];
const wc023 = (messages: unknown[]): string[] => VALIDATORS.anthropic({ model: "claude", max_tokens: 10, messages }).filter((v) => v.startsWith("WC-023"));

test("resolveAnthropicSummaryAsSystem: env tier wins, garbage falls through, default off", () => {
    assert.equal(resolveAnthropicSummaryAsSystem({ [ENV]: "1" } as NodeJS.ProcessEnv), true);
    assert.equal(resolveAnthropicSummaryAsSystem({ [ENV]: "ON" } as NodeJS.ProcessEnv), true);
    assert.equal(resolveAnthropicSummaryAsSystem({ [ENV]: " 0 " } as NodeJS.ProcessEnv, true), false, "env 0 beats file true");
    assert.equal(resolveAnthropicSummaryAsSystem({ [ENV]: "bogus" } as NodeJS.ProcessEnv, true), true, "garbage env falls to file");
    assert.equal(resolveAnthropicSummaryAsSystem({} as NodeJS.ProcessEnv), false, "opt-in default off");
    assert.equal(resolveAnthropicSummaryAsSystem({} as NodeJS.ProcessEnv, true), true, "file bool honored");
    assert.equal(resolveAnthropicSummaryAsSystem({} as NodeJS.ProcessEnv, "yes"), false, "non-bool file falls to default off");
});

test("config plumbing: parse, per-field cascade, resolved config, env-only reachability", () => {
    assert.equal(parseCompressSettings({ anthropicSummaryAsSystem: true })?.anthropicSummaryAsSystem, true);
    assert.equal(parseCompressSettings({ anthropicSummaryAsSystem: false })?.anthropicSummaryAsSystem, false, "explicit false kept");
    assert.equal(parseCompressSettings({ anthropicSummaryAsSystem: "yes" }), undefined, "non-boolean rejected");
    assert.equal(mergeCompress({ anthropicSummaryAsSystem: true }, undefined, undefined).anthropicSummaryAsSystem, true, "global level inherited");
    assert.equal(mergeCompress({ anthropicSummaryAsSystem: true }, undefined, { anthropicSummaryAsSystem: false }).anthropicSummaryAsSystem, false, "model level wins");

    const base = defaultConfig(200000);
    withEnv(undefined, () => {
        assert.equal(applyCompressSettings(base, 200000, { anthropicSummaryAsSystem: true }).anthropicSummaryAsSystem, true);
        assert.equal(applyCompressSettings(base, 200000, {}).anthropicSummaryAsSystem, false, "unset resolves to off");
        assert.equal(resolveRequestConfig(base, {}, undefined, "model-x", 200000), base, "nothing set keeps the short-circuit");
    });
    withEnv("0", () => {
        assert.equal(applyCompressSettings(base, 200000, { anthropicSummaryAsSystem: true }).anthropicSummaryAsSystem, false, "env 0 beats file true");
    });
    withEnv("on", () => {
        const resolved = resolveRequestConfig(base, {}, undefined, "model-x", 200000);
        assert.notEqual(resolved, base, "env-only opt-in must not die in the short-circuit");
        assert.equal(resolved.anthropicSummaryAsSystem, true);
    });
});

test("a summary between a user message and an assistant message becomes its own system message", () => {
    const folded = [user("u1", "first request"), summary(1), assistant("a9", "later answer"), user("u10", "next")];
    const out = wire(folded);
    assert.deepEqual(out.map((m) => m.role), ["user", "system", "assistant", "user"]);
    assert.deepEqual(out[1], { role: "system", content: [{ type: "text", text: `${SUM} 1` }] });
    assert.deepEqual(wc023(out), [], "two-sided position rule holds");

    // The head is the SAME bytes the unfolded conversation started with.
    const unfolded = coreToAnthropic([user("u1", "first request"), assistant("a1", "first answer")]);
    assert.equal(JSON.stringify(out[0]), JSON.stringify(unfolded[0]), "first user message untouched by the fold");

    // Without the transform the summary rides inside the first user message.
    const legacy = coreToAnthropic(folded) as { role: string; content: { text?: string }[] }[];
    assert.deepEqual(legacy.map((m) => m.role), ["user", "assistant", "user"]);
    assert.equal(legacy[0]!.content.length, 2, "legacy layout merges the summary into the head message");
});

test("the transform never mutates its input and is a same-array no-op when nothing qualifies", () => {
    const folded = [user("u1", "q"), summary(1), assistant("a9", "a")];
    const snapshot = JSON.stringify(folded);
    const out = voice(folded);
    assert.notEqual(out, folded);
    assert.equal(JSON.stringify(folded), snapshot, "input left intact");
    assert.equal(out[0], folded[0], "untouched messages keep their identity");

    const plain = [user("u1", "q"), assistant("a1", "a")];
    assert.equal(voice(plain), plain, "no summary: same array");
});

test("summaries in any other position keep the user voice", () => {
    const cases: Record<string, BiliMessage[]> = {
        "followed by a user message": [user("u1", "q"), summary(1), user("u9", "q2"), assistant("a9", "a")],
        "preceded by an assistant message": [user("u1", "q"), assistant("a1", "a"), summary(1), assistant("a9", "a2")],
        "at the array start": [summary(1), assistant("a9", "a"), user("u10", "q")],
        "at the array end": [user("u1", "q"), summary(1)],
        "preceded by a client system sidecar": [user("u1", "q"), sidecar("s1", "tools updated"), summary(1), assistant("a9", "a")],
        "followed by a client system sidecar": [user("u1", "q"), summary(1), sidecar("s1", "tools updated"), assistant("a9", "a")],
        "empty summary text": [user("u1", "q"), summary(1, ""), assistant("a9", "a")],
        "a mixed run ending before a user message": [user("u1", "q"), summary(1), summary(2), user("u9", "q2"), assistant("a9", "a")],
    };
    for (const [name, messages] of Object.entries(cases)) {
        assert.equal(voice(messages), messages, `${name}: unchanged`);
        assert.deepEqual(wc023(coreToAnthropic(messages)), [], `${name}: legacy wire stays legal`);
    }
    const clientSystem: BiliMessage = { id: "sys-note", role: "system", contentType: "text", text: "client note" };
    const withClientSystem = [user("u1", "q"), clientSystem, assistant("a1", "a")];
    assert.equal(voice(withClientSystem), withClientSystem, "non-summary system data is not bili's to re-voice");
});

test("a later adjacent fold appends a system message and leaves the earlier prefix byte-identical", () => {
    const first = wire([user("u1", "q"), summary(1), assistant("a9", "a9"), user("u10", "q10"), assistant("a10", "a10")]);
    const second = wire([user("u1", "q"), summary(1), summary(2), assistant("a20", "a20")]);
    assert.deepEqual(second.map((m) => m.role), ["user", "system", "system", "assistant"]);
    assert.equal(JSON.stringify(second.slice(0, 2)), JSON.stringify(first.slice(0, 2)), "head + first summary survive the second fold");
    assert.deepEqual(wc023(second), [], "a run of systems before an assistant is legal");
});

test("a cache mark on the summary rides the emitted system block", () => {
    const mark = { type: "ephemeral" };
    const out = coreToAnthropic(voice([user("u1", "q"), summary(1), assistant("a9", "a")]), new Map<string, unknown>([["acp_summary_b1", mark]]));
    assert.deepEqual(out[1], { role: "system", content: [{ type: "text", text: `${SUM} 1`, cache_control: mark }] });
});

test("round-2 adapter mirrors the steady path: on = system voice, off = legacy bytes", () => {
    const body = { model: "claude", messages: [], stream: true, max_tokens: 10 };
    const core = [user("u1", "q"), summary(1), assistant("a9", "a"), user("u10", "q2")] as CoreMessage[];
    const on = createAnthropicAdapter(body, undefined, undefined, "protocol", undefined, undefined, undefined, true).buildRequest(core, "sys", body) as { messages: { role: string }[] };
    assert.equal(JSON.stringify(on.messages), JSON.stringify(wire(core as BiliMessage[])), "identical to the steady-path rendering");
    assert.deepEqual(VALIDATORS.anthropic(on).filter((v) => v.startsWith("WC-023")), []);

    const off = createAnthropicAdapter(body).buildRequest(core, "sys", body) as { messages: { role: string }[] };
    assert.equal(JSON.stringify(off.messages), JSON.stringify(coreToAnthropic(core as BiliMessage[])), "default off is byte-identical to the pre-feature layout");
});

function makeSession(): Session {
    return {
        id: `sv-${Math.random().toString(36).slice(2)}`,
        meta: {},
        stats: { requests: 0, tokensSaved: 0, inputTokens: 0, cachedTokens: 0, outputTokens: 0, cacheSamples: 0, lastInputTokens: 0, contextTokens: 0, compressCreditTokens: 0, retrieveCalls: 0, retrieveHits: 0, retrieveMisses: 0, retrieveExpired: 0 },
        metadata: {},
        state: createInitialState(),
        pendingRetrievals: [],
        createdAt: Date.now(),
        lastSeen: Date.now(),
        blockContents: new Map(),
        inFlight: 0,
        persisted: false,
    } as unknown as Session;
}

// 40 alternating user/assistant messages; the fold starts at m00002 (the
// first assistant message), i.e. right after the conversation's first user
// message — the geometry that used to rewrite the head.
function foldedSession(endRef: string): { session: Session; core: ReturnType<typeof createCore>; body: AnthropicRequestBody } {
    const session = makeSession();
    const core = createCore();
    const config = defaultConfig(200000);
    const body: AnthropicRequestBody = { model: "claude-test", messages: [] };
    for (let i = 0; i < 40; i++) body.messages.push({ role: i % 2 === 0 ? "user" : "assistant", content: `message ${String(i)} ${"y".repeat(2000)}` });
    const { msgs } = anthropicToCore(body);
    const turn = core.processTurn({ messages: msgs, state: session.state, config, tokenCount: 9999, renderTags: "text-only" });
    session.state = turn.state;
    const res = core.applyCompression({
        ranges: [{ startRef: "m00002", endRef, summary: "early history summary long enough to pass min length check".repeat(3) }],
        state: session.state,
        config,
        messages: turn.messages,
    });
    session.state = res.state;
    assert.equal(res.result.blocksCreated, 1, "compression block should be created");
    return { session, core, body };
}

function countTokensWire(endRef: string, summaryAsSystem: boolean): { role: string; content: unknown }[] {
    const { session, core, body } = foldedSession(endRef);
    const config = { ...defaultConfig(200000), anthropicSummaryAsSystem: summaryAsSystem };
    const prepared = prepareCountTokens(body, core, config, () => undefined, session);
    return (JSON.parse(prepared.body.toString()) as { messages: { role: string; content: unknown }[] }).messages;
}

test("kernel fold from m00002 ending before an assistant: the head survives with the knob on", () => {
    // m00015 is a user message, so the fold's follower (m00016) is an assistant.
    const on = countTokensWire("m00015", true);
    assert.deepEqual(on.slice(0, 3).map((m) => m.role), ["user", "system", "assistant"]);
    assert.match(JSON.stringify(on[1]), /\[Compressed conversation section\]/);
    assert.doesNotMatch(JSON.stringify(on[0]), /\[Compressed conversation section\]/, "first user message carries no summary");
    assert.deepEqual(wc023(on), []);

    const off = countTokensWire("m00015", false);
    assert.deepEqual(off.slice(0, 2).map((m) => m.role), ["user", "assistant"]);
    assert.match(JSON.stringify(off[0]), /\[Compressed conversation section\]/, "default off: summary merged into the head message");
});

test("kernel fold from m00002 ending before a user message: layout is unchanged by the knob", () => {
    // m00014 is an assistant message, so the fold's follower (m00015) is a user.
    assert.equal(JSON.stringify(countTokensWire("m00014", true)), JSON.stringify(countTokensWire("m00014", false)));
});
