; ============================================================================
; constants.inc — Linux x86-64 syscalls, .asmbin header layout, Runtime struct
; Pure x86-64 NASM assembly (zero libc, zero external dependencies)
; ============================================================================

; --- Linux x86-64 syscall numbers ---
%define SYS_read            0
%define SYS_write           1
%define SYS_close           3
%define SYS_fstat           5
%define SYS_mmap            9
%define SYS_munmap          11
%define SYS_clone           56
%define SYS_exit            60
%define SYS_futex           202
%define SYS_clock_gettime   228
%define SYS_exit_group      231
%define SYS_openat          257

; --- Linux file / mmap / clone / futex constants ---
%define AT_FDCWD            -100
%define O_RDONLY            0
%define O_WRONLY_CREAT_TR   0x241       ; O_WRONLY | O_CREAT | O_TRUNC
%define MODE_0644           0o644
%define PROT_READ           1
%define PROT_RW             3           ; PROT_READ | PROT_WRITE
%define MAP_PRIVATE         2
%define MAP_PRIV_ANON       0x22        ; MAP_PRIVATE | MAP_ANONYMOUS
%define CLOCK_MONOTONIC     1

; CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD
%define CLONE_THREAD_FLAGS  0x00010F00
%define FUTEX_WAIT_PRIVATE  128
%define FUTEX_WAKE_PRIVATE  129
%define FUTEX_WAKE_ALL      0x7FFFFFFF

; struct stat (x86-64 Linux): st_size at offset 48, total size 144
%define STAT_SIZE           144
%define STAT_ST_SIZE        48

; --- .asmbin Header Offsets (128 bytes) ---
%define ASMBIN_MAGIC        0x4C4D5341  ; "ASML"
%define ASMBIN_VERSION      1
%define ASMBIN_VERSION_V2   2
%define HDR_FLAG_TIED_EMB   1
%define HDR_FLAG_QK_NORM    2
%define HDR_FLAG_FP16_SCALES 4

%define HDR_MAGIC           0
%define HDR_VERSION         4
%define HDR_DIM             8
%define HDR_HIDDEN_DIM      12
%define HDR_N_LAYERS        16
%define HDR_N_HEADS         20
%define HDR_N_KV_HEADS      24
%define HDR_HEAD_DIM        28
%define HDR_VOCAB_SIZE      32
%define HDR_MAX_SEQ_LEN     36
%define HDR_RMS_EPS         40
%define HDR_ROPE_THETA      44
%define HDR_EOS_1           48
%define HDR_EOS_2           52
%define HDR_HT_MASK         56
%define HDR_FLAGS           60
%define HDR_OFF_TOKENIZER   64
%define HDR_OFF_EMBED       72
%define HDR_OFF_LAYERS      80
%define HDR_LAYER_STRIDE    88
%define HDR_OFF_FINAL_NORM  96
%define HDR_FILE_SIZE       104

; --- Tokenizer Section Header Offsets (relative to base + off_tokenizer) ---
%define TOKH_OFF_B2T        0
%define TOKH_OFF_OFFSETS    8
%define TOKH_OFF_BYTES      16
%define TOKH_OFF_HT         24

; Merge hash table entry (16 bytes): { uint64 key, uint32 merged_id, uint32 rank }
%define HT_KEY              0
%define HT_MERGED           8
%define HT_RANK             12
%define HT_EMPTY_RANK       0xFFFFFFFF
%define HT_SHIFT            45          ; 64 - 19 (524,288 slots)
%define FIB_MULT_64         0x9E3779B97F4A7C15

; --- Qwen3 Chat Template Special Token IDs ---
%define TOK_ENDOFTEXT       151643
%define TOK_IM_START        151644
%define TOK_IM_END          151645
%define TOK_THINK_START     151667
%define TOK_THINK_END       151668
%define TOK_USER            872
%define TOK_ASSISTANT       77091
%define TOK_NL              198
%define TOK_NL2             271

; --- Limits ---
%define MAX_THREADS         64
%define WORKER_STACK_SIZE   65536
%define MAX_PROMPT_TOKENS   8192
%define MAX_GEN_TOKENS      8192

; --- Runtime Context Struct (g_ctx, 256 bytes in .bss) ---
%define CTX_MMAP_BASE       0
%define CTX_MMAP_SIZE       8
%define CTX_ARENA_BASE      16
%define CTX_ARENA_SIZE      24

%define CTX_B2T_PTR         32
%define CTX_TOK_OFFS_PTR    40
%define CTX_TOK_BYTES_PTR   48
%define CTX_HT_PTR          56
%define CTX_EMB_PTR         64
%define CTX_LAYERS_PTR      72
%define CTX_FINAL_NORM_PTR  80

%define CTX_DIM             88
%define CTX_HIDDEN_DIM      96
%define CTX_N_LAYERS        104
%define CTX_N_HEADS         112
%define CTX_N_KV_HEADS      120
%define CTX_HEAD_DIM        128
%define CTX_Q_DIM           136
%define CTX_KV_DIM          144
%define CTX_VOCAB_SIZE      152
%define CTX_MAX_SEQ         160
%define CTX_LAYER_STRIDE    168
%define CTX_HT_MASK         176
%define CTX_EOS_1           184
%define CTX_EOS_2           192
%define CTX_RMS_EPS         200         ; float32 (4B)
%define CTX_ROPE_THETA      204         ; float32 (4B)

; Scratch & KV-cache pointers (inside mmap arena)
%define CTX_BUF_X           208
%define CTX_BUF_XB          216
%define CTX_BUF_Q           224
%define CTX_BUF_K           232
%define CTX_BUF_V           240
%define CTX_BUF_ATT_OUT     248
%define CTX_BUF_ATT_SCR     256
%define CTX_BUF_GATE        264
%define CTX_BUF_UP          272
%define CTX_BUF_LOGITS      280
%define CTX_BUF_XQ          288
%define CTX_BUF_XS          296
%define CTX_BUF_INV_FREQ    304
%define CTX_BUF_COS         312
%define CTX_BUF_SIN         320
%define CTX_BUF_K_CACHE     328
%define CTX_BUF_V_CACHE     336
%define CTX_BUF_PROMPT      344
%define CTX_BUF_GEN         352
%define CTX_BUF_PROBS       360
%define CTX_FP16_SCALES     368         ; uint32: 0 = v1 fp32 scales, 1 = v2 fp16 scales
%define CTX_SCALE_SHIFT     372         ; uint32: 3 = fp32 (K>>3), 4 = fp16 (K>>4)
%define CTX_SIZE            384
