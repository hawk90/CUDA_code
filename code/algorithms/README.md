# CUDA 알고리즘 최적화 예제 (Part 5)

각 알고리즘을 **naive 베이스라인 → 누적 최적화 레벨 L0..Ln** 으로 발전시키며,
레벨마다 한 기법을 추가해 성능이 단조 개선되는 과정을 실측(GB/s·GFLOP/s)한다.

## 구조
- **common/** — 재사용 헤더온리 코어 (raii · pool · variant · verify · metrics · occupancy · harness · analysis · bandwidth · cli). 모든 알고리즘이 `../common/` 로 참조.
- **reduction · scan · matmul · convolution · sort · histogram · SpMV · FFT · softmax · attention/** — 알고리즘별 클라이언트 (`*_kernels.cuh` 커널 래더, `main.cu` 벤치, `test_*.cu` GoogleTest, `CMakeLists.txt`).
- **pitfalls/** — 흔한 실수 갤러리 + `algo_traps` (알고리즘별 baseline 함정).
- **debug/** — cuda-gdb 진단 케이스 (oob / deadlock / conditional bp / watchpoint …).
- **tma/** — Hopper TMA 자동 폴백 데모.
- **practice/** — 실습용 `baseline.cu`(시작점) + `solution.cu`(정답), 맨 위 주석에 ncu 단계별 힌트.

## 빌드 (폴더별 독립)
```bash
cd FFT
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=87   # 본인 GPU: A100=80, RTX40=89, H100=90
cmake --build build -j
./build/fft_bench            # 테스트: -DBUILD_TESTING=ON 후 ctest
```
전체 한 번에: `ARCH=87 ./build_all.sh`

## CLI (모든 벤치 공통)
```
--size,    -n N     문제 크기
--iters,   -i N     측정 반복
--variant, -a KEY   특정 레벨/변형만 실행 (예: --variant L2)   ← 레벨별 프로파일용
--list,    -l       변형 목록 (harness 벤치)
--help,    -h
```

## ncu 프로파일 (레벨별, 수강생 시연용)
레벨(L0..Ln)은 커널 이름이 달라 ncu 가 자동 분리 → 리포트 하나서 레벨별 메트릭 비교.
```bash
./build_all.sh          # 먼저 전체 빌드
sudo ./ncu_all.sh       # 모든 알고리즘 한 번에 프로파일 (ncu 는 root 필요)
#   결과: ncu_reports/<algo>.ncu-rep (Nsight Compute UI), <algo>.csv (레벨별 메트릭)
```
특정 레벨만 집중: `sudo ncu --set full FFT/build/fft_bench --variant L2 --iters 1`

> roofline 실링은 GPU마다 다르다 — 성능 수치는 실행한 GPU에서 측정해 비교하라.
