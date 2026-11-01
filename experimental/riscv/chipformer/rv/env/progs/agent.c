/* agent - the Hermes-gadget-style harness as a bare-metal program for the
 * Qwenputer machine: a pre-configured system prompt and a small tool set
 * baked into the binary, talking to the model through a mailbox in RAM.
 *
 * This is the in-weights shape of the /dev/qwen idea. On the reference
 * machine (and one day in the compiled circuit) the guest has no host OS
 * to hand it a model connection, so the "device" is a mailbox: the guest
 * writes a prompt into mb_out, sets mb_req, and spins; whoever runs the
 * machine - the session runner today, the model's own language layers in
 * the endgame - writes the answer into mb_in and clears mb_req. The
 * protocol above the mailbox is the same one the Linux guest's agent.sh
 * speaks: one line each way, TOOL <call> or SAY: <answer>.
 *
 * Tools: store <key> <value> / recall <key> / mul <a> <b> (exact).
 * The task is baked in (pre-configured): remember a word, do an exact
 * multiplication no language model does reliably in its head,
 * and report both.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- the mailbox (symbols read by the runner; see bareagent.py) ---- */
/* MB_SIZE is a build-time knob (build_agent.sh -DMB_SIZE): 896 on the
 * stock 32 KB map; the grown 2 MB map build uses 8192 so long model
 * answers fit whole instead of truncating at the old cap. */
#ifndef MB_SIZE
#define MB_SIZE 896
#endif
volatile unsigned int mb_req;
volatile unsigned int mb_outlen;
volatile unsigned int mb_inlen;
volatile char mb_out[MB_SIZE];
volatile char mb_in[MB_SIZE];

static const char *SYS =
    "Answer with exactly one line, no explanation. Tools: "
    "'TOOL store <key> <value>' saves a value under a key, "
    "'TOOL recall <key>' reads it back, "
    "'TOOL mul <a> <b>' multiplies two integers exactly. "
    "After each tool call you get its RESULT. "
    "When the task is finished reply 'SAY: <final answer>'. "
    "Example: TOOL mul 6 7, then after RESULT 42 you reply "
    "SAY: the product is 42.";

/* Operand note (2026-10-31): the pair was 123456789 x 987654321 until
 * the built-in brain (Qwen 3.5 4B) was measured falling into an
 * endless reasoning loop on the 9-digit pair - it tries to work the
 * product out in its head and never emits an answer (bisected live:
 * the store-only task answers instantly, the 9-digit mul task never
 * does, at any token budget). 6 digits x 6 digits (an 11-digit
 * product) stays far beyond in-head reliability and answers cleanly. */
static const char *TASK =
    "remember the word banana under the key fruit, compute "
    "123456 * 654321 with the mul tool, then tell me the word "
    "under fruit and the product";

/* ---- tiny key-value store ---- */
static char kv_key[4][24];
static char kv_val[4][96];
static int kv_n;

static const char *kv_get(const char *key) {
    for (int i = 0; i < kv_n; i++)
        if (strcmp(kv_key[i], key) == 0) return kv_val[i];
    return NULL;
}

static void kv_put(const char *key, const char *val) {
    /* strncpy leaves the destination unterminated when the source is
     * longer than the buffer - and the value is model-controlled text
     * of mailbox length, so terminate explicitly every time */
    for (int i = 0; i < kv_n; i++)
        if (strcmp(kv_key[i], key) == 0) {
            strncpy(kv_val[i], val, sizeof(kv_val[i]) - 1);
            kv_val[i][sizeof(kv_val[i]) - 1] = 0;
            return;
        }
    if (kv_n < 4) {
        strncpy(kv_key[kv_n], key, sizeof(kv_key[0]) - 1);
        kv_key[kv_n][sizeof(kv_key[0]) - 1] = 0;
        strncpy(kv_val[kv_n], val, sizeof(kv_val[0]) - 1);
        kv_val[kv_n][sizeof(kv_val[0]) - 1] = 0;
        kv_n++;
    }
}

/* ---- helpers ---- */
static void u64_dec(unsigned long long v, char *out) {
    char tmp[24];
    int i = 0, j = 0;
    if (v == 0) { out[0] = '0'; out[1] = 0; return; }
    while (v) { tmp[i++] = (char)('0' + v % 10); v /= 10; }
    while (i) out[j++] = tmp[--i];
    out[j] = 0;
}

static unsigned long long parse_u64(const char **p) {
    unsigned long long v = 0;
    while (**p == ' ') (*p)++;
    while (**p >= '0' && **p <= '9') v = v * 10 + (unsigned)(**p - '0'), (*p)++;
    return v;
}

static void ask_model(const char *prompt, char *answer, int cap) {
    int n = (int)strlen(prompt);
    if (n > (int)sizeof(mb_out) - 1) n = (int)sizeof(mb_out) - 1;
    memcpy((void *)mb_out, prompt, (size_t)n);
    mb_out[n] = 0;
    mb_outlen = (unsigned)n;
    mb_inlen = 0;
    mb_req = 1;
    while (mb_req) { /* the machine's other side answers */ }
    int m = (int)mb_inlen;
    if (m > cap - 1) m = cap - 1;
    memcpy(answer, (const void *)mb_in, (size_t)m);
    answer[m] = 0;
}

/* run one TOOL call; returns a static result string */
static char result_buf[256];
static const char *run_tool(const char *call) {
    if (strncmp(call, "store ", 6) == 0) {
        char key[24];
        const char *p = call + 6;
        int i = 0;
        while (*p && *p != ' ' && i < (int)sizeof(key) - 1) key[i++] = *p++;
        key[i] = 0;
        while (*p == ' ') p++;
        /* a model that simulates ahead can glue the next call onto the
         * value ("banana TOOL mul ...") - the value ends there */
        const char *glue = strstr(p, " TOOL ");
        if (glue) {
            static char vbuf[96];
            snprintf(vbuf, sizeof(vbuf), "%.*s", (int)(glue - p), p);
            kv_put(key, vbuf);
            snprintf(result_buf, sizeof(result_buf), "stored %s = %s",
                     key, vbuf);
            return result_buf;
        }
        kv_put(key, p);
        snprintf(result_buf, sizeof(result_buf), "stored %s = %s", key, p);
        return result_buf;
    }
    if (strncmp(call, "recall ", 7) == 0) {
        char key[24];
        const char *p = call + 7;
        int i = 0;
        while (*p && *p != ' ' && i < (int)sizeof(key) - 1) key[i++] = *p++;
        key[i] = 0;
        const char *v = kv_get(key);
        if (v) snprintf(result_buf, sizeof(result_buf), "%s = %s", key, v);
        else snprintf(result_buf, sizeof(result_buf),
                      "(nothing stored under %s)", key);
        return result_buf;
    }
    if (strncmp(call, "mul ", 4) == 0) {
        const char *p = call + 4;
        unsigned long long a = parse_u64(&p);
        unsigned long long b = parse_u64(&p);
        char dec[24];
        u64_dec(a * b, dec);
        snprintf(result_buf, sizeof(result_buf), "%s", dec);
        return result_buf;
    }
    snprintf(result_buf, sizeof(result_buf), "(unknown tool)");
    return result_buf;
}

int main(void) {
    static char prompt[1200];
    static char answer[MB_SIZE];
    static char prev_call[512];
    static char prev_result[256];
    const char *result = NULL;

    printf("[agent] bare-metal harness up; task: %s\n", TASK);
    for (int round = 1; round <= 9; round++) {
        if (round == 1)
            snprintf(prompt, sizeof(prompt), "%s TASK: %s", SYS, TASK);
        else
            snprintf(prompt, sizeof(prompt), "%s TASK: %s RESULT: %s",
                     SYS, TASK, result);
        ask_model(prompt, answer, sizeof(answer));
        printf("[agent] model: %s\n", answer);

        char *say = strstr(answer, "SAY:");
        char *tool = strstr(answer, "TOOL ");
        if (tool && (!say || tool < say)) {
            char *call = tool + 5;
            /* one call per reply: a model that simulates ahead glues
             * the whole plan onto its first call ("store a b TOOL mul
             * ... RESULT ... SAY: ..."). Cut the call at the next call
             * or at a simulated RESULT/SAY so exactly the first real
             * call runs - and so the repeat check below compares the
             * call itself, not the varying simulated tail. */
            for (char *cut = call; (cut = strstr(cut, " TOOL ")) != NULL; ) {
                char *after = cut + 6;
                if (strncmp(after, "store ", 6) == 0 ||
                    strncmp(after, "recall ", 7) == 0 ||
                    strncmp(after, "mul ", 4) == 0) { *cut = 0; break; }
                cut = after;
            }
            char *rc = strstr(call, " RESULT");
            if (rc) *rc = 0;
            char *sc = strstr(call, " SAY:");
            if (sc) *sc = 0;
            if (strcmp(call, prev_call) == 0) {
                static char nudge[384];
                printf("[agent] repeat of the last call - not running it again\n");
                snprintf(nudge, sizeof(nudge),
                         "you already ran that tool call (result: %s). "
                         "Do the NEXT step or SAY the final answer",
                         prev_result);
                result = nudge;
                continue;
            }
            const char *r = run_tool(call);
            printf("[agent] ran: TOOL %s\n", call);
            printf("[agent] result: %s\n", r);
            strncpy(prev_call, call, sizeof(prev_call) - 1);
            prev_call[sizeof(prev_call) - 1] = 0;
            strncpy(prev_result, r, sizeof(prev_result) - 1);
            prev_result[sizeof(prev_result) - 1] = 0;
            result = prev_result;
            continue;
        }
        if (say) {
            printf("AGENT_ANSWER: %s\n", say + 4);
            return 0;
        }
        if (strncmp(answer, "[qwen:", 6) == 0) {
            printf("[agent] empty or device reply - nudging the model\n");
            result = "(your last reply was empty)";
            continue;
        }
        if (strlen(answer) > 200) {
            /* a long non-tool reply is the model thinking out loud,
             * not an answer - nudge it back to the one-line protocol
             * (a real final answer on these tasks is short) */
            printf("[agent] long non-tool reply - nudging the model\n");
            result = "(that was not a tool call or a final answer) "
                     "Reply with exactly one line: TOOL ... or SAY: ...";
            continue;
        }
        printf("AGENT_ANSWER: %s\n", answer);
        return 0;
    }
    printf("AGENT_ANSWER: (round limit reached) last tool result: %s\n",
           prev_result);
    printf("[agent] stopped: tool round limit (9) reached\n");
    return 0;
}
