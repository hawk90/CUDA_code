#!/usr/bin/env bash
# 모든 알고리즘을 Nsight Compute(ncu)로 한 번에 프로파일.
#
# ncu 는 (1) root 권한, (2) 올바른 환경변수가 필요하다. sudo 는 PATH/LD_LIBRARY_PATH 를
# 초기화하므로 그냥 `sudo ncu` 하면 ncu 를 못 찾거나 앱이 libcudart 로드에 실패할 수 있다.
# → 이 스크립트가 CUDA 경로를 스스로 탐지해 설정하므로 다음 중 하나로 실행하면 된다:
#       ./build_all.sh                 # (먼저 전체 빌드)
#       sudo -E ./ncu_all.sh           # -E: 현재 환경 보존(권장)
#       sudo ./ncu_all.sh              # (스크립트가 CUDA env 를 재설정하므로 이것도 동작)
#       sudo ./ncu_all.sh basic        # 빠른 섹션 세트
#   비root로 쓰려면 드라이버 옵션 필요:
#       sudo sh -c 'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
#                    > /etc/modprobe.d/ncu.conf' && 재부팅
#
# 레벨(L0..Ln)은 커널 이름이 서로 달라 ncu 가 자동으로 분리해준다
# → 리포트 하나 안에서 레벨별 메트릭을 나란히 비교(수강생 시연용).
# 결과: ncu_reports/<algo>.ncu-rep (Nsight Compute UI), <algo>.csv (레벨별 메트릭 raw)
set -u

SET="${1:-full}"                 # full(기본) | basic
OUT="ncu_reports"; mkdir -p "$OUT"

# --- CUDA 환경 자동 설정 (sudo 가 env 를 날려도 동작하도록) ---
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}"
NCU="$(command -v ncu || true)"
[[ -z "$NCU" && -x "$CUDA_HOME/bin/ncu" ]] && NCU="$CUDA_HOME/bin/ncu"
if [[ -z "$NCU" ]]; then
  echo "ncu 를 못 찾음. CUDA_HOME 을 지정하라:  CUDA_HOME=/path/to/cuda sudo -E ./ncu_all.sh" >&2
  exit 1
fi
echo "ncu = $NCU   (CUDA_HOME=$CUDA_HOME)"
if [[ "$(id -u)" -ne 0 ]]; then
  echo "주의: ncu 는 보통 root 권한이 필요하다. sudo -E ./ncu_all.sh 로 실행하라." >&2
fi

FAILED=0
prof() {                          # prof <이름> <실행파일> <앱 인자...>
  local name="$1" exe="$2"; shift 2
  if [[ ! -x "$exe" ]]; then echo "skip $name  (빌드 안됨 → ./build_all.sh 먼저)"; return; fi
  echo "=== ncu: $name ==="
  rm -f "$OUT/$name.ncu-rep"
  "$NCU" --set "$SET" --export "$OUT/$name" --force-overwrite --csv --page raw \
         "$exe" "$@" > "$OUT/$name.csv" 2> "$OUT/$name.log" || true
  # ncu 는 권한이 없어도 앱만 실행하고 exit 0 → 실제 수집 여부는 .ncu-rep 생성으로 판정.
  if [[ -s "$OUT/$name.ncu-rep" ]]; then
    echo "  → $OUT/$name.ncu-rep (Nsight Compute UI), $OUT/$name.csv (레벨별 메트릭)"
  else
    FAILED=1
    local hint; hint="$(grep -m1 -iE 'privileg|ERROR|permission' "$OUT/$name.log" 2>/dev/null)"
    echo "  프로파일 데이터 없음 → ${hint:-권한 문제로 보임}"
  fi
}

# 벤치별 작은 크기 + iters 1 (ncu 는 커널을 여러 번 replay 하므로 작게).
prof reduction   reduction/build/reduction_bench     --size 1048576 --iters 1
prof scan        scan/build/scan_bench               --size 1048576 --iters 1
prof histogram   histogram/build/histogram_bench     --size 1048576 --iters 1
prof matmul      matmul/build/matmul_bench           --size 1024    --iters 1
prof convolution convolution/build/convolution_bench --size 1024    --iters 1
prof sort        sort/build/sort_bench               --size 262144  --iters 1
prof spmv        SpMV/build/spmv_bench               --size 32768   --iters 1
prof fft         FFT/build/fft_bench                 --size 1024    --iters 1
prof softmax     softmax/build/softmax_bench         --size 2048    --iters 1
prof attention   attention/build/attention_bench     --size 512     --iters 1

echo
if [[ "$FAILED" -ne 0 ]]; then
  echo "일부/전체 프로파일 실패(데이터 없음). ncu 는 root + CUDA env 가 필요하다:"
  echo "   sudo -E ./ncu_all.sh          # 환경 보존하며 root 로 재실행"
  echo "   (비root 허용하려면 NVreg_RestrictProfilingToAdminUsers=0 후 재부팅)"
else
  echo "완료.  ncu_reports/*.ncu-rep 를 Nsight Compute 로 열어 레벨(커널)별 메트릭을 비교하라."
fi
echo "특정 레벨만 집중: sudo -E $NCU --set full <algo>/build/<bench> --variant L2 --iters 1"
