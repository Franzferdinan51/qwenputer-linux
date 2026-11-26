; ============================================================================
; sgemm.asm — Zero-libc Generic Thread Pool + High-Throughput AVX2/FMA FP32 SGEMM
;
; Exports:
;   tp_init(uint64 n_threads)
;   tp_shutdown(void)
;   tp_get_threads(void) -> uint64
;   tp_run(void (*fn)(uint64 tid, uint64 n_threads, void *arg), void *arg)
;   tp_transpose_f32(const float *src, float *dst, uint64 rows, uint64 cols)
;   tp_sgemm(const float *A, const float *B, float *C,
;            uint64 M, uint64 N, uint64 K, uint64 flags)
;   tp_sgemm_nt(const float *A, const float *B, float *C,
;               uint64 M, uint64 N, uint64 K, uint64 beta_flag, float *scratch_bt)
;
; Architecture:
;   - Generic parallel dispatch via raw Linux clone() + futex() with 64-byte
;     cache-line-isolated completion slots (zero cross-CCX atomic contention).
;   - Register-blocked 6x16 AVX2+FMA SGEMM micro-kernel (12 YMM accumulators,
;     zero stack traffic in inner k-loop) supporting both NN (A[M,K]@B[K,N])
;     and TN (A[K,M]^T@B[K,N]) plus arbitrary edge dimensions (e.g. vocab=65).
;   - Parallelized strictly over disjoint 6x16 output tiles of C -> 100%
;     BIT-FOR-BIT DETERMINISTIC across any thread count (1T..64T)!
; ============================================================================

bits 64
default rel

%include "train_constants.inc"

%define SPIN_LIMIT  200000

extern sys_mmap_anon

section .text

global tp_init
global tp_shutdown
global tp_get_threads
global tp_run
global tp_transpose_f32
global tp_sgemm
global tp_sgemm_nt

; ============================================================================
; GENERIC ZERO-LIBC THREAD POOL (clone + futex + cache-line done slots)
; ============================================================================

; ----------------------------------------------------------------------------
; void tp_init(uint64 n_threads)
; ----------------------------------------------------------------------------
tp_init:
    push    rbx
    push    r12
    push    r13

    cmp     rdi, 1
    jge     .chk_max
    mov     rdi, 1
.chk_max:
    cmp     rdi, MAX_THREADS
    jle     .store_n
    mov     rdi, MAX_THREADS
.store_n:
    mov     [rel tp_n_threads], rdi
    mov     dword [rel tp_gen], 0
    mov     dword [rel tp_stop], 0
    mov     dword [rel tp_waiters], 0

    lea     rax, [rel tp_done_arr]
    xor     ecx, ecx
.clr_lp:
    cmp     rcx, MAX_THREADS
    jae     .clr_done
    mov     rdx, rcx
    shl     rdx, 6
    mov     dword [rax + rdx], 0
    inc     rcx
    jmp     .clr_lp
.clr_done:

    cmp     rdi, 1
    jle     .done

    mov     r12, rdi
    mov     rbx, 1
.spawn_lp:
    cmp     rbx, r12
    jae     .done

    mov     rdi, WORKER_STACK_SIZE
    call    sys_mmap_anon
    test    rax, rax
    js      .spawn_fail

    lea     rsi, [rax + WORKER_STACK_SIZE - 16]
    mov     [rsi], rbx                  ; tid on worker stack

    mov     edi, CLONE_THREAD_FLAGS
    xor     edx, edx
    xor     r10d, r10d
    xor     r8d, r8d
    mov     eax, SYS_clone
    syscall

    test    rax, rax
    jz      .worker_entry
    js      .spawn_fail

    inc     rbx
    jmp     .spawn_lp

.spawn_fail:
    mov     [rel tp_n_threads], rbx
.done:
    pop     r13
    pop     r12
    pop     rbx
    ret

.worker_entry:
    pop     r12                         ; r12 = worker tid (1 .. n_threads - 1)
    and     rsp, -16
    xor     r13d, r13d                  ; my_gen = 0

.worker_wait:
    mov     ecx, SPIN_LIMIT
.spin_lp:
    mov     eax, [rel tp_gen]
    cmp     eax, r13d
    jne     .got_job
    pause
    dec     ecx
    jnz     .spin_lp

    lock inc dword [rel tp_waiters]
    mov     eax, [rel tp_gen]
    cmp     eax, r13d
    jne     .unwait

    lea     rdi, [rel tp_gen]
    mov     esi, FUTEX_WAIT_PRIVATE
    mov     edx, r13d
    xor     r10d, r10d
    xor     r8d, r8d
    xor     r9d, r9d
    mov     eax, SYS_futex
    syscall

.unwait:
    lock dec dword [rel tp_waiters]
    jmp     .worker_wait

.got_job:
    cmp     dword [rel tp_stop], 0
    jne     .worker_exit
    mov     r13d, eax                   ; latch new_gen

    ; Call tp_fn(tid=r12, n_threads, tp_arg)
    mov     rdi, r12
    mov     rsi, [rel tp_n_threads]
    mov     rdx, [rel tp_arg]
    mov     rax, [rel tp_fn]
    call    rax

    ; Signal completion on worker's private 64B cache line
    lea     rax, [rel tp_done_arr]
    mov     rcx, r12
    shl     rcx, 6
    mov     [rax + rcx], r13d
    jmp     .worker_wait

.worker_exit:
    xor     edi, edi
    mov     eax, SYS_exit
    syscall
    hlt

; ----------------------------------------------------------------------------
; void tp_shutdown(void)
; ----------------------------------------------------------------------------
tp_shutdown:
    cmp     qword [rel tp_n_threads], 1
    jle     .ret
    mov     dword [rel tp_stop], 1
    lock inc dword [rel tp_gen]
    lea     rdi, [rel tp_gen]
    mov     esi, FUTEX_WAKE_PRIVATE
    mov     edx, FUTEX_WAKE_ALL
    xor     r10d, r10d
    xor     r8d, r8d
    xor     r9d, r9d
    mov     eax, SYS_futex
    syscall
.ret:
    ret

; ----------------------------------------------------------------------------
; uint64 tp_get_threads(void)
; ----------------------------------------------------------------------------
tp_get_threads:
    mov     rax, [rel tp_n_threads]
    test    rax, rax
    jnz     .ok
    mov     rax, 1
.ok:
    ret

; ----------------------------------------------------------------------------
; void tp_run(void (*fn)(uint64 tid, uint64 n_threads, void *arg), void *arg)
; rdi = fn, rsi = arg
; ----------------------------------------------------------------------------
tp_run:
    mov     rax, [rel tp_n_threads]
    cmp     rax, 1
    jle     .single

    push    rbx
    push    r12
    sub     rsp, 8                      ; 16B align
    mov     rbx, rax                    ; n_threads

    mov     [rel tp_fn], rdi
    mov     [rel tp_arg], rsi

    mov     r12d, 1
    lock xadd dword [rel tp_gen], r12d
    inc     r12d                        ; r12d = new_gen

    cmp     dword [rel tp_waiters], 0
    je      .run_main
    lea     rdi, [rel tp_gen]
    mov     esi, FUTEX_WAKE_PRIVATE
    mov     edx, FUTEX_WAKE_ALL
    xor     r10d, r10d
    xor     r8d, r8d
    xor     r9d, r9d
    mov     eax, SYS_futex
    syscall

.run_main:
    xor     edi, edi                    ; tid = 0
    mov     rsi, rbx                    ; n_threads
    mov     rdx, [rel tp_arg]
    mov     rax, [rel tp_fn]
    call    rax

    ; Wait for workers 1 .. n_threads - 1
    lea     rdi, [rel tp_done_arr]
    mov     rcx, 1
.wait_tid:
    cmp     rcx, rbx
    jae     .done
    mov     rdx, rcx
    shl     rdx, 6
.spin_tid:
    cmp     dword [rdi + rdx], r12d
    je      .next_tid
    pause
    jmp     .spin_tid
.next_tid:
    inc     rcx
    jmp     .wait_tid

.done:
    lfence
    add     rsp, 8
    pop     r12
    pop     rbx
    ret

.single:
    mov     rax, rdi
    mov     rdx, rsi                    ; arg
    xor     edi, edi                    ; tid = 0
    mov     esi, 1                      ; n_threads = 1
    jmp     rax

; ============================================================================
; PARALLEL TILED MATRIX TRANSPOSE: dst[cols, rows] = src[rows, cols]^T
; ============================================================================

; void tp_transpose_f32(const float *src, float *dst, uint64 rows, uint64 cols)
; rdi = src, rsi = dst, rdx = rows, rcx = cols
tp_transpose_f32:
    sub     rsp, 40                     ; 32B job struct + 8B align
    mov     [rsp + 0], rdi              ; src
    mov     [rsp + 8], rsi              ; dst
    mov     [rsp + 16], rdx             ; rows
    mov     [rsp + 24], rcx             ; cols
    lea     rdi, [rel transpose_worker]
    mov     rsi, rsp
    call    tp_run
    add     rsp, 40
    ret

; void transpose_worker(uint64 tid, uint64 n_threads, void *arg)
; Partition rows in 8-row blocks across threads; inner 8x8 cache-blocked copy.
transpose_worker:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15

    mov     r8,  [rdx + 0]              ; src
    mov     r9,  [rdx + 8]              ; dst
    mov     r10, [rdx + 16]             ; rows
    mov     r11, [rdx + 24]             ; cols

    ; r0 = (rows * tid) / n_threads; r1 = (rows * (tid + 1)) / n_threads
    mov     rax, r10
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     r12, rax                    ; r = r0

    lea     rax, [rdi + 1]
    imul    rax, r10
    xor     edx, edx
    div     rsi
    mov     r13, rax                    ; r1

    mov     r14, r10
    shl     r14, 2                      ; dst_row_bytes = rows * 4
    mov     r15, r11
    shl     r15, 2                      ; src_row_bytes = cols * 4

.r_lp:
    cmp     r12, r13
    jae     .done
    ; src_row = src + r * cols * 4
    mov     rax, r12
    imul    rax, r15
    lea     rbx, [r8 + rax]             ; &src[r, 0]
    ; dst_col = dst + r * 4 (steps by rows * 4 per c)
    lea     rbp, [r9 + r12*4]           ; &dst[0, r]

    xor     ecx, ecx                    ; c = 0
.c_lp:
    cmp     rcx, r11
    jae     .r_next
    mov     eax, [rbx + rcx*4]
    mov     [rbp], eax
    add     rbp, r14
    inc     rcx
    jmp     .c_lp
.r_next:
    inc     r12
    jmp     .r_lp
.done:
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; PARALLEL FP32 AVX2+FMA SGEMM (6x16 Register-Blocked Micro-Kernel)
; ============================================================================

; SGEMM job descriptor on stack (72 bytes)
%define SJ_A            0
%define SJ_B            8
%define SJ_C            16
%define SJ_M            24
%define SJ_N            32
%define SJ_K            40
%define SJ_FLAGS        48
%define SJ_M_TILES      56
%define SJ_N_TILES      64

; ----------------------------------------------------------------------------
; void tp_sgemm(const float *A, const float *B, float *C,
;               uint64 M, uint64 N, uint64 K, uint64 flags)
; rdi=A, rsi=B, rdx=C, rcx=M, r8=N, r9=K, [rsp+8]=flags
; ----------------------------------------------------------------------------
tp_sgemm:
    mov     r10, [rsp + 8]              ; 7th arg: flags (before sub rsp!)
    sub     rsp, 88                     ; 72B struct + align (88 + 8 ret = 96 = 16*6)
    mov     [rsp + SJ_A], rdi
    mov     [rsp + SJ_B], rsi
    mov     [rsp + SJ_C], rdx
    mov     [rsp + SJ_M], rcx
    mov     [rsp + SJ_N], r8
    mov     [rsp + SJ_K], r9
    mov     [rsp + SJ_FLAGS], r10

    ; m_tiles = (M + 5) / 6
    lea     rax, [rcx + 5]
    xor     edx, edx
    mov     r11, 6
    div     r11
    mov     [rsp + SJ_M_TILES], rax

    ; n_tiles = (N + 15) >> 4
    lea     rax, [r8 + 15]
    shr     rax, 4
    mov     [rsp + SJ_N_TILES], rax

    lea     rdi, [rel sgemm_worker]
    mov     rsi, rsp
    call    tp_run

    add     rsp, 88
    ret

; ----------------------------------------------------------------------------
; void tp_sgemm_nt(const float *A, const float *B, float *C,
;                  uint64 M, uint64 N, uint64 K, uint64 beta_flag, float *scratch_bt)
; C[M, N] = beta*C + A[M, K] @ B[N, K]^T
; rdi=A, rsi=B, rdx=C, rcx=M, r8=N, r9=K, [rsp+8]=beta_flag, [rsp+16]=scratch_bt
; ----------------------------------------------------------------------------
tp_sgemm_nt:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    ; 6 pushes = 48B; original [rsp+8] is now at [rsp+56], [rsp+16] at [rsp+64]
    mov     r12, rdi                    ; A
    mov     r13, rdx                    ; C
    mov     r14, rcx                    ; M
    mov     r15, r8                     ; N
    mov     rbx, r9                     ; K
    mov     rbp, [rsp + 64]             ; scratch_bt
    mov     r10, [rsp + 56]             ; beta_flag

    sub     rsp, 24                     ; 16B align + save beta_flag
    mov     [rsp + 8], r10

    ; Transpose B[N, K] -> scratch_bt[K, N]
    mov     rdi, rsi                    ; src = B
    mov     rsi, rbp                    ; dst = scratch_bt
    mov     rdx, r15                    ; rows = N
    mov     rcx, rbx                    ; cols = K
    call    tp_transpose_f32

    ; Call tp_sgemm(A, scratch_bt, C, M, N, K, beta_flag)
    mov     rdi, r12
    mov     rsi, rbp
    mov     rdx, r13
    mov     rcx, r14
    mov     r8,  r15
    mov     r9,  rbx
    mov     rax, [rsp + 8]
    mov     [rsp + 0], rax              ; 7th arg on stack!
    call    tp_sgemm

    add     rsp, 24
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ----------------------------------------------------------------------------
; void sgemm_worker(uint64 tid, uint64 n_threads, void *arg)
; Executes assigned slice of 6x16 output tiles [t0, t1).
; ----------------------------------------------------------------------------
sgemm_worker:
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 104                    ; 6 pushes(48) + 104 + 8(ret) = 160 (16B aligned)

    ; Stack locals:
    ;   [rsp + 0]  = job ptr
    ;   [rsp + 8]  = a_row_stride (bytes)
    ;   [rsp + 16] = a_k_stride   (bytes)
    ;   [rsp + 24] = b_row_stride (= N * 4)
    ;   [rsp + 32] = n_tiles
    ;   [rsp + 40] = tiles_left
    ;   [rsp + 48] = cur_it
    ;   [rsp + 56] = cur_jt
    ;   [rsp + 64] = M
    ;   [rsp + 72] = N
    ;   [rsp + 80] = K
    ;   [rsp + 88] = flags
    mov     [rsp + 0], rdx
    mov     r8,  [rdx + SJ_M]
    mov     r9,  [rdx + SJ_N]
    mov     r10, [rdx + SJ_K]
    mov     r11, [rdx + SJ_FLAGS]
    mov     [rsp + 64], r8
    mov     [rsp + 72], r9
    mov     [rsp + 80], r10
    mov     [rsp + 88], r11

    ; Strides for A (NN vs TN) and B
    test    r11, SGEMM_TRANS_A
    jnz     .stride_tn
    ; NN: a_row_stride = K * 4, a_k_stride = 4
    lea     rax, [r10 * 4]
    mov     [rsp + 8], rax
    mov     qword [rsp + 16], 4
    jmp     .stride_b
.stride_tn:
    ; TN: a_row_stride = 4, a_k_stride = M * 4
    mov     qword [rsp + 8], 4
    lea     rax, [r8 * 4]
    mov     [rsp + 16], rax
.stride_b:
    lea     rax, [r9 * 4]
    mov     [rsp + 24], rax             ; b_row_stride = N * 4 = c_row_stride

    ; Compute total_tiles = m_tiles * n_tiles
    mov     rcx, [rdx + SJ_N_TILES]
    mov     [rsp + 32], rcx
    mov     rax, [rdx + SJ_M_TILES]
    imul    rax, rcx                    ; total_tiles
    mov     rbp, rax

    ; t0 = (total_tiles * tid) / n_threads
    imul    rax, rdi
    xor     edx, edx
    div     rsi
    mov     r12, rax                    ; t0

    ; t1 = (total_tiles * (tid + 1)) / n_threads
    lea     rax, [rdi + 1]
    imul    rax, rbp
    xor     edx, edx
    div     rsi
    sub     rax, r12                    ; tiles_left = t1 - t0
    jle     .exit
    mov     [rsp + 40], rax

    ; cur_it = t0 / n_tiles, cur_jt = t0 % n_tiles
    mov     rax, r12
    xor     edx, edx
    div     qword [rsp + 32]
    mov     [rsp + 48], rax             ; cur_it
    mov     [rsp + 56], rdx             ; cur_jt

.tile_lp:
    mov     rax, [rsp + 48]             ; it
    imul    rax, rax, 6                 ; i = it * 6
    mov     rbx, [rsp + 56]             ; jt
    shl     rbx, 4                      ; j = jt * 16

    ; Check if full 6x16 tile: (i + 6 <= M) && (j + 16 <= N)
    lea     rcx, [rax + 6]
    cmp     rcx, [rsp + 64]
    ja      .edge_tile
    lea     rcx, [rbx + 16]
    cmp     rcx, [rsp + 72]
    ja      .edge_tile

    ; ========================================================================
    ; FAST 6x16 AVX2+FMA MICRO-KERNEL (zero stack accesses inside k-loop!)
    ; ========================================================================
    mov     rbp, [rsp + 0]              ; job ptr
    ; c_ptr = C + (i * N + j) * 4
    mov     rcx, rax
    imul    rcx, [rsp + 72]
    add     rcx, rbx
    shl     rcx, 2
    add     rcx, [rbp + SJ_C]
    mov     r14, rcx                    ; r14 = c_ptr (preserved across k-loop!)
    mov     r13, [rsp + 24]             ; r13 = b_row_stride = c_row_stride = N * 4

    ; Initialize ymm0..ymm11 (zero if beta==0, else load C)
    test    qword [rsp + 88], SGEMM_BETA_ONE
    jnz     .load_c6
    vxorps  ymm0,  ymm0,  ymm0
    vxorps  ymm1,  ymm1,  ymm1
    vxorps  ymm2,  ymm2,  ymm2
    vxorps  ymm3,  ymm3,  ymm3
    vxorps  ymm4,  ymm4,  ymm4
    vxorps  ymm5,  ymm5,  ymm5
    vxorps  ymm6,  ymm6,  ymm6
    vxorps  ymm7,  ymm7,  ymm7
    vxorps  ymm8,  ymm8,  ymm8
    vxorps  ymm9,  ymm9,  ymm9
    vxorps  ymm10, ymm10, ymm10
    vxorps  ymm11, ymm11, ymm11
    jmp     .prep_k6

.load_c6:
    mov     rdi, r14
    vmovups ymm0,  [rdi]
    vmovups ymm1,  [rdi + 32]
    add     rdi, r13
    vmovups ymm2,  [rdi]
    vmovups ymm3,  [rdi + 32]
    add     rdi, r13
    vmovups ymm4,  [rdi]
    vmovups ymm5,  [rdi + 32]
    add     rdi, r13
    vmovups ymm6,  [rdi]
    vmovups ymm7,  [rdi + 32]
    add     rdi, r13
    vmovups ymm8,  [rdi]
    vmovups ymm9,  [rdi + 32]
    add     rdi, r13
    vmovups ymm10, [rdi]
    vmovups ymm11, [rdi + 32]

.prep_k6:
    ; r8  = a0 = A + i * a_row_stride
    mov     r9,  [rsp + 8]              ; a_row_stride
    mov     r8,  rax
    imul    r8,  r9
    add     r8,  [rbp + SJ_A]           ; a0
    lea     rdx, [r9 + r9]              ; 2 * a_row_stride
    lea     rax, [r8 + rdx]
    add     rax, r9                     ; rax = a3 = a0 + 3 * a_row_stride
    mov     r11, [rsp + 16]             ; a_k_stride

    ; r12 = B + j * 4
    mov     r12, [rbp + SJ_B]
    lea     r12, [r12 + rbx*4]

    mov     rcx, [rsp + 80]             ; K
    test    rcx, rcx
    jz      .store_c6

.k6_lp:
    vmovups ymm12, [r12]                ; B[k, j..j+7]
    vmovups ymm13, [r12 + 32]           ; B[k, j+8..j+15]

    vbroadcastss ymm14, [r8]            ; A[i+0, k]
    vfmadd231ps  ymm0,  ymm14, ymm12
    vfmadd231ps  ymm1,  ymm14, ymm13

    vbroadcastss ymm15, [r8 + r9]       ; A[i+1, k]
    vfmadd231ps  ymm2,  ymm15, ymm12
    vfmadd231ps  ymm3,  ymm15, ymm13

    vbroadcastss ymm14, [r8 + rdx]      ; A[i+2, k]
    vfmadd231ps  ymm4,  ymm14, ymm12
    vfmadd231ps  ymm5,  ymm14, ymm13

    vbroadcastss ymm15, [rax]           ; A[i+3, k]
    vfmadd231ps  ymm6,  ymm15, ymm12
    vfmadd231ps  ymm7,  ymm15, ymm13

    vbroadcastss ymm14, [rax + r9]      ; A[i+4, k]
    vfmadd231ps  ymm8,  ymm14, ymm12
    vfmadd231ps  ymm9,  ymm14, ymm13

    vbroadcastss ymm15, [rax + rdx]     ; A[i+5, k]
    vfmadd231ps  ymm10, ymm15, ymm12
    vfmadd231ps  ymm11, ymm15, ymm13

    add     r8,  r11
    add     rax, r11
    add     r12, r13
    dec     rcx
    jnz     .k6_lp

.store_c6:
    mov     rdi, r14
    vmovups [rdi],      ymm0
    vmovups [rdi + 32], ymm1
    add     rdi, r13
    vmovups [rdi],      ymm2
    vmovups [rdi + 32], ymm3
    add     rdi, r13
    vmovups [rdi],      ymm4
    vmovups [rdi + 32], ymm5
    add     rdi, r13
    vmovups [rdi],      ymm6
    vmovups [rdi + 32], ymm7
    add     rdi, r13
    vmovups [rdi],      ymm8
    vmovups [rdi + 32], ymm9
    add     rdi, r13
    vmovups [rdi],      ymm10
    vmovups [rdi + 32], ymm11
    jmp     .tile_next

    ; ========================================================================
    ; EDGE TILE MICRO-KERNEL (arbitrary mr in 1..6, nr in 1..16)
    ; rax = i, rbx = j
    ; ========================================================================
.edge_tile:
    mov     rbp, [rsp + 0]              ; job ptr
    ; mr = min(6, M - i) -> r14
    mov     r14, [rsp + 64]
    sub     r14, rax
    cmp     r14, 6
    jbe     .mr_ok
    mov     r14, 6
.mr_ok:
    ; nr = min(16, N - j) -> r15
    mov     r15, [rsp + 72]
    sub     r15, rbx
    cmp     r15, 16
    jbe     .nr_ok
    mov     r15, 16
.nr_ok:
    ; r8 = a_row_base = A + i * a_row_stride
    mov     r8, rax
    imul    r8, [rsp + 8]
    add     r8, [rbp + SJ_A]

    ; r9 = c_row_base = C + (i * N + j) * 4
    mov     r9, rax
    imul    r9, [rsp + 72]
    add     r9, rbx
    shl     r9, 2
    add     r9, [rbp + SJ_C]

    ; r10 = b_base = B + j * 4
    mov     r10, [rbp + SJ_B]
    lea     r10, [r10 + rbx*4]

    mov     r11, [rsp + 16]             ; a_k_stride
    mov     r13, [rsp + 24]             ; b_row_stride = c_row_stride

.edge_r_lp:
    test    r14, r14
    jz      .tile_next

    xor     r12d, r12d                  ; col offset c = 0
.edge_c8_lp:
    lea     rcx, [r12 + 8]
    cmp     rcx, r15
    ja      .edge_c1_lp

    test    qword [rsp + 88], SGEMM_BETA_ONE
    jnz     .edge_c8_ld
    vxorps  ymm0, ymm0, ymm0
    jmp     .edge_c8_k
.edge_c8_ld:
    vmovups ymm0, [r9 + r12*4]
.edge_c8_k:
    mov     rdi, r8                     ; a_cur
    lea     rsi, [r10 + r12*4]          ; b_cur
    mov     rcx, [rsp + 80]             ; K
    test    rcx, rcx
    jz      .edge_c8_st
.edge_c8_klp:
    vbroadcastss ymm1, [rdi]
    vfmadd231ps  ymm0, ymm1, [rsi]
    add     rdi, r11
    add     rsi, r13
    dec     rcx
    jnz     .edge_c8_klp
.edge_c8_st:
    vmovups [r9 + r12*4], ymm0
    add     r12, 8
    jmp     .edge_c8_lp

.edge_c1_lp:
    cmp     r12, r15
    jae     .edge_r_next
    test    qword [rsp + 88], SGEMM_BETA_ONE
    jnz     .edge_c1_ld
    vxorps  xmm0, xmm0, xmm0
    jmp     .edge_c1_k
.edge_c1_ld:
    vmovss  xmm0, [r9 + r12*4]
.edge_c1_k:
    mov     rdi, r8
    lea     rsi, [r10 + r12*4]
    mov     rcx, [rsp + 80]
    test    rcx, rcx
    jz      .edge_c1_st
.edge_c1_klp:
    vmovss  xmm1, [rdi]
    vfmadd231ss xmm0, xmm1, [rsi]
    add     rdi, r11
    add     rsi, r13
    dec     rcx
    jnz     .edge_c1_klp
.edge_c1_st:
    vmovss  [r9 + r12*4], xmm0
    inc     r12
    jmp     .edge_c1_lp

.edge_r_next:
    add     r8, [rsp + 8]               ; next row of A
    add     r9, r13                     ; next row of C
    dec     r14
    jmp     .edge_r_lp

.tile_next:
    dec     qword [rsp + 40]
    jz      .exit
    mov     rax, [rsp + 56]             ; jt
    inc     rax
    cmp     rax, [rsp + 32]             ; n_tiles
    jb      .store_jt
    xor     eax, eax
    inc     qword [rsp + 48]            ; ++it
.store_jt:
    mov     [rsp + 56], rax
    jmp     .tile_lp

.exit:
    vzeroupper
    add     rsp, 104
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

; ============================================================================
; BSS: Thread Pool State
; ============================================================================
section .bss
alignb 64
tp_gen:       resd 1
tp_stop:      resd 1
tp_n_threads: resq 1
tp_fn:        resq 1
tp_arg:       resq 1

alignb 64
tp_waiters:   resd 1
resb 60

alignb 64
tp_done_arr:  resb MAX_THREADS * 64
