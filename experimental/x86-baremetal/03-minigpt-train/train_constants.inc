; ============================================================================
; train_constants.inc — Phase 3 Pure x86-64 Mini-GPT Training Engine Constants
; Zero libc, zero libm, zero external dependencies.
; ============================================================================

%include "constants.inc"

; Additional Linux x86-64 syscalls not in Phase 1 constants.inc
%define SYS_mkdir           83

; SGEMM flags
%define SGEMM_BETA_ZERO     0
%define SGEMM_BETA_ONE      1
%define SGEMM_TRANS_A       2

; Checkpoint binary format ("AGPT", version 1)
%define CKPT_MAGIC          0x54504741      ; "AGPT"
%define CKPT_VERSION        1
%define CKPT_HDR_SIZE       64
%define CKPT_VOCAB_SIZE_B   512             ; 256 id_to_byte + 256 byte_to_id

; Checkpoint header offsets
%define CHDR_MAGIC          0               ; u32
%define CHDR_VERSION        4               ; u32
%define CHDR_VOCAB          8               ; u32
%define CHDR_DIM            12              ; u32
%define CHDR_LAYERS         16              ; u32
%define CHDR_HEADS          20              ; u32
%define CHDR_HIDDEN         24              ; u32
%define CHDR_SEQ            28              ; u32
%define CHDR_STEP           32              ; u64
%define CHDR_N_PARAMS       40              ; u64
%define CHDR_RMS_EPS        48              ; f32
%define CHDR_ROPE_THETA     52              ; f32

; ============================================================================
; Global Training Context Struct (g_trn in .bss, 64-byte aligned)
; ============================================================================
; Hyperparameters & derived dimensions
%define TRN_VOCAB           0               ; u64
%define TRN_DIM             8               ; u64
%define TRN_LAYERS          16              ; u64
%define TRN_HEADS           24              ; u64
%define TRN_HEAD_DIM        32              ; u64 (dim / heads)
%define TRN_HIDDEN          40              ; u64 (SwiGLU FF dim)
%define TRN_SEQ             48              ; u64 (T)
%define TRN_BATCH           56              ; u64 (B)
%define TRN_BT              64              ; u64 (B * T)
%define TRN_N_PARAMS        72              ; u64 (total FP32 params)
%define TRN_LAYER_PARAMS    80              ; u64 (params per layer)
%define TRN_RMS_EPS         88              ; f32
%define TRN_ROPE_THETA      92              ; f32
%define TRN_ATT_SCALE       96              ; f32 (1.0 / sqrt(head_dim))
%define TRN_GRAD_CLIP       100             ; f32
%define TRN_LR_MAX          104             ; f32
%define TRN_WD              108             ; f32
%define TRN_STEPS           112             ; u64
%define TRN_WARMUP          120             ; u64
%define TRN_CUR_STEP        128             ; u64 (0-indexed)
%define TRN_SAMPLE_ONLY     136             ; u64 (0=train, 1=sample-only)

; Parameter arena & optimizer state
%define TRN_PARAM_ARENA     144             ; void* (mmap base)
%define TRN_PARAM_ARENA_SZ  152             ; u64
%define TRN_PARAMS          160             ; float* [n_params]
%define TRN_GRADS           168             ; float* [n_params]
%define TRN_OPT_M           176             ; float* [n_params]
%define TRN_OPT_V           184             ; float* [n_params]

; Running FP64 AdamW bias-correction powers (beta1^t, beta2^t)
%define TRN_B1_POW          192             ; f64 (init 1.0)
%define TRN_B2_POW          200             ; f64 (init 1.0)

; Activation & scratch arena
%define TRN_ACT_ARENA       208             ; void* (mmap base)
%define TRN_ACT_ARENA_SZ    216             ; u64

; Per-layer saved activation base pointers (indexed by layer l * layer_stride)
; Layout of layer_x: [L+1, BT, D]
%define TRN_BUF_LAYER_X     224             ; float* [(L+1) * BT * D]
; Per-layer contiguous activation block (stride = TRN_LACT_STRIDE bytes per layer)
%define TRN_BUF_LACT        232             ; void*  [L * lact_stride]
%define TRN_LACT_STRIDE     240             ; u64    bytes per layer in LACT
%define TRN_OFF_LN1_OUT     248             ; u64    offset inside layer block [BT, D]
%define TRN_OFF_LN1_INV     256             ; u64    [BT]
%define TRN_OFF_Q           264             ; u64    [BT, D] (post-RoPE)
%define TRN_OFF_K           272             ; u64    [BT, D] (post-RoPE)
%define TRN_OFF_V           280             ; u64    [BT, D]
%define TRN_OFF_PROBS       288             ; u64    [B, H, T, T]
%define TRN_OFF_ATT_OUT     296             ; u64    [BT, D]
%define TRN_OFF_X_MID       304             ; u64    [BT, D]
%define TRN_OFF_LN2_OUT     312             ; u64    [BT, D]
%define TRN_OFF_LN2_INV     320             ; u64    [BT]
%define TRN_OFF_GATE        328             ; u64    [BT, FF]
%define TRN_OFF_UP          336             ; u64    [BT, FF]
%define TRN_OFF_ACT         344             ; u64    [BT, FF]

; Final norm & logits & loss buffers
%define TRN_BUF_FINAL_OUT   352             ; float* [BT, D]
%define TRN_BUF_FINAL_INV   360             ; float* [BT]
%define TRN_BUF_LOGITS      368             ; float* [BT, V] (in-place -> d_logits in train)
%define TRN_BUF_LOSSES      376             ; float* [BT]

; Scratch buffers (reused across layers during forward/backward)
%define TRN_BUF_WT          384             ; float* [max(D*D, D*FF, V*D)] for W^T
%define TRN_BUF_DX          392             ; float* [BT, D]
%define TRN_BUF_DTMP        400             ; float* [BT, D]
%define TRN_BUF_DQ          408             ; float* [BT, D]
%define TRN_BUF_DK          416             ; float* [BT, D]
%define TRN_BUF_DV          424             ; float* [BT, D]
%define TRN_BUF_DGATE       432             ; float* [BT, FF]
%define TRN_BUF_DUP         440             ; float* [BT, FF]
%define TRN_BUF_DACT        448             ; float* [BT, FF]

; RoPE precomputed tables: cos[T, hd/2], sin[T, hd/2]
%define TRN_BUF_INV_FREQ    456             ; double* [hd/2]
%define TRN_BUF_ROPE_COS    464             ; float* [T, hd/2]
%define TRN_BUF_ROPE_SIN    472             ; float* [T, hd/2]

; Current batch token buffers & sampling scratch
%define TRN_BUF_BATCH_X     480             ; u32* [BT]
%define TRN_BUF_BATCH_Y     488             ; u32* [BT]
%define TRN_BUF_SAMPLE_P    496             ; float* [V] (probs for sample_top_p)

; Parameter sub-offsets within a single layer (in FLOAT counts, not bytes!)
; Layer order: ln1[D], wq[D*D], wk[D*D], wv[D*D], wo[D*D],
;              ln2[D], wgate[FF*D], wup[FF*D], wdown[D*FF]
%define TRN_POFF_LAYERS     504             ; u64 float offset of layer 0 (= V*D)
%define TRN_POFF_FINAL_LN   512             ; u64 float offset of final_ln
%define TRN_POFF_LM_HEAD    520             ; u64 float offset of lm_head
%define TRN_LOFF_LN1        528             ; u64 (= 0)
%define TRN_LOFF_WQ         536             ; u64 (= D)
%define TRN_LOFF_WK         544             ; u64 (= D + D*D)
%define TRN_LOFF_WV         552             ; u64 (= D + 2*D*D)
%define TRN_LOFF_WO         560             ; u64 (= D + 3*D*D)
%define TRN_LOFF_LN2        568             ; u64 (= D + 4*D*D)
%define TRN_LOFF_WGATE      576             ; u64 (= 2*D + 4*D*D)
%define TRN_LOFF_WUP        584             ; u64 (= 2*D + 4*D*D + FF*D)
%define TRN_LOFF_WDOWN      592             ; u64 (= 2*D + 4*D*D + 2*FF*D)

; Dataset mmap & train/val encoded token arrays
%define TRN_DATA_MMAP       600             ; void*
%define TRN_DATA_SIZE       608             ; u64 (raw file bytes)
%define TRN_TOK_IDS         616             ; u8* (encoded dataset, same len as DATA_SIZE)
%define TRN_TRAIN_LEN       624             ; u64 (90% split)
%define TRN_VAL_LEN         632             ; u64 (10% split)

; RNG state (xorshift64*) & latest metrics
%define TRN_RNG             640             ; u64
%define TRN_LAST_LOSS       648             ; f32
%define TRN_LAST_GNORM      652             ; f32
%define TRN_LAST_LR         656             ; f32

; Vocabulary tables (embedded directly in g_trn)
%define TRN_ID_TO_BYTE      672             ; u8[256]
%define TRN_BYTE_TO_ID      928             ; u8[256]

%define TRN_SIZE            1216            ; total struct size (64B multiple)
