#!/usr/bin/env bash
# ============================================================================
# scripts/mkiso.sh — Build bootable GRUB Multiboot 1 ISO for asm-llm OS
#
# Supports:
#   - Path A: BIOS + UEFI hybrid ISO via grub-mkrescue (when i386-pc exists)
#   - Path B: Pure UEFI GRUB Multiboot 1 ISO (zero mtools / zero root) via
#             grub-mkstandalone + pure-Python FAT16 EFI System Partition + xorriso
# ============================================================================
set -euo pipefail

KERNEL_ELF="${1:?usage: mkiso.sh <asm-llm-os.elf> <model.asmbin> <out.iso> [cmdline]}"
MODEL_BIN="${2:?missing model.asmbin}"
OUT_ISO="${3:?missing output .iso path}"
CMDLINE="${4:-}"

if [[ ! -f "$KERNEL_ELF" ]]; then
    echo "error: kernel not found: $KERNEL_ELF" >&2
    exit 1
fi
if [[ ! -f "$MODEL_BIN" ]]; then
    echo "error: model not found: $MODEL_BIN" >&2
    exit 1
fi

OUT_DIR="$(dirname "$OUT_ISO")"
mkdir -p "$OUT_DIR"
WORK_BASE="${TMPDIR:-$(cd "$OUT_DIR" && pwd)}"
WORK_DIR="$(mktemp -d "${WORK_BASE}/asm-llm-iso.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

ISO_ROOT="$WORK_DIR/iso"
mkdir -p "$ISO_ROOT/boot/grub"
cp "$KERNEL_ELF" "$ISO_ROOT/boot/asm-llm-os.elf"

# Note: quote cmdline carefully for GRUB2 multiboot command
cat > "$ISO_ROOT/boot/grub/grub.cfg" <<EOF
serial --unit=0 --speed=115200 --word=8 --parity=no --stop=1
terminal_input console serial
terminal_output console serial
set timeout=0
set default=0

menuentry "asm-llm OS v2.0 (Bare-Metal Qwen3-0.6B in Pure x86-64 Assembly)" {
    multiboot /boot/asm-llm-os.elf ${CMDLINE}
    module /boot/model.asmbin
    boot
}
EOF

# Path A: BIOS+UEFI hybrid ISO via grub-mkrescue
if [[ -d /usr/lib/grub/i386-pc ]] && command -v grub-mkrescue >/dev/null 2>&1; then
    echo "[mkiso] Building BIOS+UEFI hybrid ISO via grub-mkrescue..."
    cp --reflink=auto "$MODEL_BIN" "$ISO_ROOT/boot/model.asmbin" 2>/dev/null \
        || cp "$MODEL_BIN" "$ISO_ROOT/boot/model.asmbin"
    grub-mkrescue -o "$OUT_ISO" "$ISO_ROOT"
    ls -lh "$OUT_ISO"
    exit 0
fi

# Path B: UEFI GRUB Multiboot 1 ISO via grub-mkstandalone + xorriso (zero mtools/root)
echo "[mkiso] Building UEFI GRUB Multiboot 1 ISO via grub-mkstandalone + xorriso..."
ln -s "$(realpath "$MODEL_BIN")" "$ISO_ROOT/boot/model.asmbin"
mkdir -p "$WORK_DIR/efi_tree/EFI/BOOT"

cat > "$WORK_DIR/embed.cfg" <<'EOF'
search --no-floppy --file --set=root /boot/asm-llm-os.elf
set prefix=($root)/boot/grub
configfile ($root)/boot/grub/grub.cfg
EOF

grub-mkstandalone \
    -O x86_64-efi \
    -o "$WORK_DIR/efi_tree/EFI/BOOT/BOOTX64.EFI" \
    --modules="part_gpt part_msdos iso9660 fat multiboot normal serial terminal search search_fs_file configfile" \
    "boot/grub/grub.cfg=$WORK_DIR/embed.cfg"

# Create minimal FAT16 EFI System Partition image (zero root, zero mtools needed)
EFI_IMG="$ISO_ROOT/boot/efiboot.img"
python3 - "$WORK_DIR/efi_tree/EFI/BOOT/BOOTX64.EFI" "$EFI_IMG" <<'PYEOF'
import math, os, struct, sys

efi_src, out_img = sys.argv[1], sys.argv[2]
data = open(efi_src, "rb").read()

BPS = 512
SPC = 4                         # 2 KiB clusters
CLUST_BYTES = BPS * SPC
RESV = 4
NFATS = 2
ROOT_ENTRIES = 512
ROOT_SECTORS = (ROOT_ENTRIES * 32) // BPS
data_clusters = max(16, math.ceil(len(data) / CLUST_BYTES) + 32)
# FAT16 requires >= 4085 clusters
total_clusters = max(4120, data_clusters)
fat_sectors = math.ceil((total_clusters + 2) * 2 / BPS)
data_start_sec = RESV + NFATS * fat_sectors + ROOT_SECTORS
total_sectors = data_start_sec + total_clusters * SPC

img = bytearray(total_sectors * BPS)

# Boot sector (BPB)
bpb = struct.pack(
    "<3s8sHBHBHHBHHHIIBBBI11s8s",
    b"\xEB\x3C\x90", b"ASMLLMOS",
    BPS, SPC, RESV, NFATS, ROOT_ENTRIES,
    total_sectors if total_sectors < 65536 else 0,
    0xF8, fat_sectors, 32, 64, 0,
    0 if total_sectors < 65536 else total_sectors,
    0x80, 0, 0x29, 0x20261125, b"ASMLLM_EFI ", b"FAT16   ",
)
img[0:len(bpb)] = bpb
img[510:512] = b"\x55\xAA"

fat = bytearray(fat_sectors * BPS)
struct.pack_into("<HH", fat, 0, 0xFFF8, 0xFFFF)

def clust_off(c):
    return (data_start_sec + (c - 2) * SPC) * BPS

def dir_entry(name83, attr, first_clust, size):
    return struct.pack(
        "<11sBBBHHHHHHHL",
        name83, attr, 0, 0, 0, 0, 0, 0, 0, 0, first_clust, size
    )

# Cluster 2 = /EFI dir, Cluster 3 = /EFI/BOOT dir, Cluster 4.. = BOOTX64.EFI
struct.pack_into("<HH", fat, 2 * 2, 0xFFFF, 0xFFFF)
n_file_clust = math.ceil(len(data) / CLUST_BYTES)
for i in range(n_file_clust):
    c = 4 + i
    nxt = (c + 1) if (i + 1 < n_file_clust) else 0xFFFF
    struct.pack_into("<H", fat, c * 2, nxt)

# Write both FAT copies
for f in range(NFATS):
    off = (RESV + f * fat_sectors) * BPS
    img[off:off + len(fat)] = fat

# Root directory: EFI
root_off = (RESV + NFATS * fat_sectors) * BPS
img[root_off:root_off + 32] = dir_entry(b"EFI        ", 0x10, 2, 0)

# /EFI directory: . .. BOOT
c2 = clust_off(2)
img[c2 + 0:c2 + 32]  = dir_entry(b".          ", 0x10, 2, 0)
img[c2 + 32:c2 + 64] = dir_entry(b"..         ", 0x10, 0, 0)
img[c2 + 64:c2 + 96] = dir_entry(b"BOOT       ", 0x10, 3, 0)

# /EFI/BOOT directory: . .. BOOTX64.EFI
c3 = clust_off(3)
img[c3 + 0:c3 + 32]  = dir_entry(b".          ", 0x10, 3, 0)
img[c3 + 32:c3 + 64] = dir_entry(b"..         ", 0x10, 2, 0)
img[c3 + 64:c3 + 96] = dir_entry(b"BOOTX64 EFI", 0x20, 4, len(data))

# File data starting at cluster 4
c4 = clust_off(4)
img[c4:c4 + len(data)] = data

with open(out_img, "wb") as f:
    f.write(img)
PYEOF

xorriso -as mkisofs \
    -follow-links \
    -R -J -V "ASM_LLM_OS" \
    -e boot/efiboot.img \
    -no-emul-boot \
    -isohybrid-gpt-basdat \
    -o "$OUT_ISO" \
    "$ISO_ROOT"

ls -lh "$OUT_ISO"
