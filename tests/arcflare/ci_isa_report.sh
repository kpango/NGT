#!/usr/bin/env bash
# ci_isa_report.sh — dump the CI runner's CPU model + SIMD ISA flags.
#
# The whole point of running the ArcFlare-vs-QG benchmark on a GitHub-hosted runner
# is to exercise the AVX-512/VNNI distance kernels, which the local dev box
# (AVX2-only Zen 2) cannot. The ArcFlare SIMD kernels are compile-time gated on
# __AVX512F__ / __AVX512DQ__ (lib/NGT/ArcFlare/SIMDUtils.h, ADCTable.h) and selected
# at run time by the multi-versioned gpq4 kernel (CPUID dispatch). This script
# verifies the runner actually advertises those flags so we know the AVX-512 path
# is both compiled in (-march=native) and dispatched.
#
# Writes a markdown block to $GITHUB_STEP_SUMMARY when set, and always echoes to
# stdout. Exit code is always 0 (informational); the workflow inspects the
# emitted has_avx512 marker line if it wants to gate.
set -u

emit() { printf '%s\n' "$*"; [ -n "${GITHUB_STEP_SUMMARY:-}" ] && printf '%s\n' "$*" >> "$GITHUB_STEP_SUMMARY"; }

cpu_model="$(LC_ALL=C lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -1)"
[ -z "$cpu_model" ] && cpu_model="$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -1)"
[ -z "$cpu_model" ] && cpu_model="(unknown)"
n_cpu="$(nproc 2>/dev/null || echo '?')"
mem_kb="$(sed -n 's/^MemTotal:[[:space:]]*\([0-9]*\).*/\1/p' /proc/meminfo 2>/dev/null | head -1)"
mem_gb="?"; [ -n "$mem_kb" ] && mem_gb="$(awk "BEGIN{printf \"%.1f\", $mem_kb/1024/1024}")"

# Single space-delimited flags line from /proc/cpuinfo.
flags_line="$(sed -n 's/^flags[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -1)"

# ISA flags we care about for the ArcFlare kernels.
WATCH="avx2 f16c fma avx512f avx512dq avx512bw avx512vl avx512cd avx512vnni avx512_vnni avx_vnni avx512vpopcntdq avx512_bf16"

present=""
absent=""
for f in $WATCH; do
  if printf ' %s ' "$flags_line" | grep -q " $f "; then
    present="$present $f"
  else
    absent="$absent $f"
  fi
done

# AVX-512 foundation = avx512f + avx512dq (matches defines.h.in NGT_AVX512 guard).
has_avx512=no
printf ' %s ' "$flags_line" | grep -q ' avx512f ' && printf ' %s ' "$flags_line" | grep -q ' avx512dq ' && has_avx512=yes
# VNNI (the int8 dot-product accel the new kernel targets): either AVX-512 VNNI or AVX-VNNI.
has_vnni=no
printf ' %s ' "$flags_line" | grep -qE ' (avx512vnni|avx512_vnni|avx_vnni) ' && has_vnni=yes

emit "## Runner CPU / SIMD ISA report"
emit ""
emit "| Property | Value |"
emit "|---|---|"
emit "| CPU model | \`${cpu_model}\` |"
emit "| Logical CPUs | ${n_cpu} |"
emit "| RAM | ${mem_gb} GiB |"
emit "| AVX-512 foundation (F+DQ) | **${has_avx512}** |"
emit "| VNNI (int8 dot-product) | **${has_vnni}** |"
emit ""
emit "**SIMD flags present:** \`$(echo "$present" | xargs echo)\`"
emit ""
emit "**SIMD flags absent:** \`$(echo "$absent" | xargs echo)\`"
emit ""

if [ "$has_avx512" = yes ]; then
  emit "> ✅ Runner advertises AVX-512. Built with \`-march=native\`, the ArcFlare"
  emit "> AVX-512 distance kernels (\`SIMDUtils.h\`, \`ADCTable.h\`) are compiled in and"
  emit "> the multi-versioned gpq4 kernel will dispatch to them at run time."
else
  emit "> ⚠️ Runner does **not** advertise AVX-512 (F+DQ)."
  emit ">"
  emit "> Standard GitHub-hosted \`ubuntu-latest\` runners are virtualized and the"
  emit "> exposed CPU varies (some Intel Skylake-SP/Cascade Lake hosts DO carry"
  emit "> AVX-512; many do not, and AMD EPYC hosts up to Zen 3 lack it entirely)."
  emit "> When AVX-512 is absent the build silently falls back to the AVX2 path, so"
  emit "> **these numbers do not exercise the AVX-512/VNNI kernels**."
  emit ">"
  emit "> To guarantee AVX-512/VNNI, use one of:"
  emit "> - a larger GitHub-hosted runner (\`ubuntu-latest-4-cores\` / 8-core) — still"
  emit ">   not guaranteed, but pulls from a newer host pool;"
  emit "> - a **self-hosted runner** on a Sapphire Rapids / Ice Lake / Cascade Lake"
  emit ">   box (label it e.g. \`[self-hosted, avx512]\` and set \`runs-on\` to match);"
  emit "> - GitHub's larger hosted runners with a pinned Intel SKU if available to the"
  emit ">   org."
fi

# Machine-readable markers for the workflow / later steps.
echo "has_avx512=${has_avx512}"
echo "has_vnni=${has_vnni}"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "has_avx512=${has_avx512}" >> "$GITHUB_OUTPUT"
  echo "has_vnni=${has_vnni}"     >> "$GITHUB_OUTPUT"
fi
exit 0
