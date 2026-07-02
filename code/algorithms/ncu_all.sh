#!/usr/bin/env bash
# 모든 알고리즘을 Nsight Compute(ncu)로 한 번에 프로파일.
#   ncu 는 root 권한 필요 → sudo 로 한 번만 돌리면 된다:
#       ./build_all.sh          # (먼저 전체 빌드)
#       sudo ./ncu_all.sh       # 전체 프로파일
#       sudo ./ncu_all.sh basic # 빠른 섹션 세트
#
# 레벨(L0..Ln)은 커널 이름이 서로 달라 ncu 가 자동으로 분리해준다
# → 리포트 하나 안에서 레벨별 메트릭을 나란히 비교할 수 있다(수강생 시연용).
# 결과: ncu_reports/<algo>.ncu-rep  (Nsight Compute UI 로 열기)
#       ncu_reports/<algo>.csv      (레벨별 메트릭 raw)
set -u

SET="${1:-full}"                 # full(기본) | basic
OUT="ncu_reports"; mkdir -p "$OUT"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "주의: ncu 는 보통 root 권한이 필요하다.  sudo ./ncu_all.sh 로 실행하라." >&2
fi

prof() {                          # prof <이름> <실행파일> <앱 인자...>
  local name="$1" exe="$2"; shift 2
  if [[ ! -x "$exe" ]]; then echo "skip $name  (빌드 안됨 → ./build_all.sh 먼저)"; return; fi
  echo "=== ncu: $name ==="
  if ncu --set "$SET" --export "$OUT/$name" --force-overwrite --csv --page raw \
         "$exe" "$@" > "$OUT/$name.csv" 2> "$OUT/$name.log"; then
    echo "  → $OUT/$name.ncu-rep (Nsight Compute UI), $OUT/$name.csv (레벨별 메트릭)"
  else
    echo "  실패 → $OUT/$name.log 확인 (권한: sudo 필요할 수 있음)"
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
prof pitfalls    pitfalls/build/pitfalls_bench

echo
echo "완료.  ncu_reports/*.ncu-rep 를 Nsight Compute 로 열어 레벨(커널)별 메트릭을 비교하라."
echo "특정 레벨만 집중: <algo>/build/<bench> --variant L2  (예: --variant L2 → L2 커널만 실행)"
