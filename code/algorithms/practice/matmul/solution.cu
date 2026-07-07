// 실습: Matrix Multiply 최적화 — SOLUTION (완전 최적화 참고답안)
// baseline.cu 의 STEP 1~2 적용: shared tiling + register tiling(ILP).
//   l1tex__t_sector_hit_rate.pct → 상승 (shared 타일 재사용)
//   dram__bytes.sum → 급감 (전역 반복 로드 제거)
//   sm__throughput.avg.pct_of_peak_sustained → 상승 (register-tiling ILP)
//   sm__pipe_fp32_cycles_active.avg.pct → 높음 (compute-bound 상한)
// 적용 기법: 64×64 타일을 shared로, 스레드당 4×4 출력을 레지스터 누산(acc[4][4]=ILP).
// (chapter34 의 sgemmRegTiled<64,64,8,4,4> 와 동일)
//
// STEP 4  SHARED BANK CONFLICT 제거 (sgemmRegTiledNBC):
//   원본 sgemmRegTiled 은 안쪽 루프의 shared 로드에 2-way bank conflict 가 있다.
//     매핑 tRow=tid/16, tCol=tid%16 → 워프(32스레드)가 tCol 0~15(×2), tRow 0/1 을 걸침.
//     Bs[kk*BN + tCol*TN + j] : tCol 16개 × TN(4) = 64 word > 32 bank → tCol, tCol+8 충돌(2-way).
//     As[(tRow*TM+i)*BK + kk] : stride BK=8, tRow 0/1 이 32 word 간격 → 같은 bank(2-way).
//   측정:  ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum ./solution
//   FIX 1  워프 리매핑 4×8 : 워프가 tCol 8개만 걸치게 → Bs 8×TN=32 word = 32 bank 정확히 1:1 (충돌 0)
//   FIX 2  As leading dim 패딩 BK→BK+1 : tRow stride 4*(BK+1)=36%32=4 ≠0 → tRow 별 bank 분산 (충돌 0)
#include "../../common/raii.cuh"
#include <cstdio>
#include <vector>

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
sgemmRegTiled(const float* __restrict__ A, const float* __restrict__ B,
              float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];
    int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;
    int tRow = threadIdx.x / (BN / TN);
    int tCol = threadIdx.x % (BN / TN);
    const int numThreads = (BM / TM) * (BN / TN);

    float acc[TM][TN];                              // 독립 누산기 = ILP
    #pragma unroll
    for (int i = 0; i < TM; ++i) for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    float regA[TM], regB[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = threadIdx.x; i < BM * BK; i += numThreads) {   // A 타일 → shared
            int r = i / BK, c = i % BK;
            As[i] = (blockRow + r < M && k0 + c < K) ? A[(blockRow + r) * K + k0 + c] : 0.0f;
        }
        for (int i = threadIdx.x; i < BK * BN; i += numThreads) {   // B 타일 → shared
            int r = i / BN, c = i % BN;
            Bs[i] = (k0 + r < K && blockCol + c < N) ? B[(k0 + r) * N + blockCol + c] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regA[i] = As[(tRow * TM + i) * BK + kk];
            #pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = Bs[kk * BN + tCol * TN + j];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];   // TM×TN 독립 FMA
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            int r = blockRow + tRow * TM + i, c = blockCol + tCol * TN + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

// STEP 4: BANK-CONFLICT-FREE 버전.
//   FIX 1 워프 리매핑(4행×8열): 워프가 tCol 을 8개(=32bank)만 걸치도록.
//   FIX 2 As leading dim 을 BK+1 로 패딩: tRow 별 bank 를 분산.
// 나머지 로직(tiling, register ILP)은 sgemmRegTiled 와 동일 → 순수하게 conflict 만 제거해 A/B 비교 가능.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
sgemmRegTiledNBC(const float* __restrict__ A, const float* __restrict__ B,
                 float* __restrict__ C, int M, int N, int K) {
    constexpr int AS_LD = BK + 1;                   // FIX 2: 패딩된 As leading dim
    __shared__ float As[BM * AS_LD];
    __shared__ float Bs[BK * BN];
    int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;
    const int numThreads = (BM / TM) * (BN / TN);   // 256

    // FIX 1: 워프를 (WM 행 × WN 열) 스레드로 배치해 워프가 tCol 을 WN(=8)개만 걸치게 한다.
    constexpr int WN = 8, WM = 4;                   // WN*WM = 32 = warpSize
    constexpr int TCOLS = BN / TN;                  // 16 thread-cols
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int warpCol = warp % (TCOLS / WN);              // 0..1
    int warpRow = warp / (TCOLS / WN);              // 0..3
    int tCol = warpCol * WN + (lane % WN);          // 0..15  (워프 내 8개 연속)
    int tRow = warpRow * WM + (lane / WN);          // 0..15  (워프 내 4개)

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;
    float regA[TM], regB[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = threadIdx.x; i < BM * BK; i += numThreads) {   // A 타일 → shared (패딩 저장)
            int r = i / BK, c = i % BK;
            As[r * AS_LD + c] = (blockRow + r < M && k0 + c < K) ? A[(blockRow + r) * K + k0 + c] : 0.0f;
        }
        for (int i = threadIdx.x; i < BK * BN; i += numThreads) {   // B 타일 → shared
            int r = i / BN, c = i % BN;
            Bs[i] = (k0 + r < K && blockCol + c < N) ? B[(k0 + r) * N + blockCol + c] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) regA[i] = As[(tRow * TM + i) * AS_LD + kk];  // 충돌 0
            #pragma unroll
            for (int j = 0; j < TN; ++j) regB[j] = Bs[kk * BN + tCol * TN + j];        // 충돌 0
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += regA[i] * regB[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            int r = blockRow + tRow * TM + i, c = blockCol + tCol * TN + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    std::vector<float> A(N * N), B(N * N), C(N * N);
    for (int i = 0; i < N * N; ++i) { A[i] = (i % 7) * 0.1f; B[i] = (i % 5) * 0.1f; }

    DeviceBuffer<float> dA(N * N), dB(N * N), dC(N * N);
    dA.copyFromHost(A.data()); dB.copyFromHost(B.data());

    dim3 grid((N + 63) / 64, (N + 63) / 64);
    const int iters = 30;
    int rr = N / 3, cc = N / 2; double ref = 0;
    for (int k = 0; k < N; ++k) ref += (double)A[rr * N + k] * B[k * N + cc];

    // 두 커널을 나란히 측정 → ncu 로 bank conflict / 성능 A/B 비교
    //   ncu --set full -k "regex:sgemm" ./solution
    auto run = [&](const char* tag, void (*launch)(dim3, DeviceBuffer<float>&, DeviceBuffer<float>&,
                                                   DeviceBuffer<float>&, int)) {
        launch(grid, dA, dB, dC, N);                       // warm-up
        CHECK_CUDA(cudaDeviceSynchronize());
        GpuTimer t; t.start();
        for (int k = 0; k < iters; ++k) launch(grid, dA, dB, dC, N);
        double ms = t.stop() / iters;
        dC.copyToHost(C.data());
        bool ok = std::abs(C[rr * N + cc] - ref) < 1e-2 * (1 + std::abs(ref));
        double gflops = 2.0 * N * N * N / (ms / 1e3) / 1e9;
        std::printf("%-12s %.4f ms   %.1f GFLOP/s   check=%s\n", tag, ms, gflops, ok ? "OK" : "FAIL");
    };

    run("regtiled:", [](dim3 g, DeviceBuffer<float>& a, DeviceBuffer<float>& b,
                        DeviceBuffer<float>& c, int n) {
        sgemmRegTiled<64, 64, 8, 4, 4><<<g, 256>>>(a.data(), b.data(), c.data(), n, n, n);
    });
    run("nbc(fixed):", [](dim3 g, DeviceBuffer<float>& a, DeviceBuffer<float>& b,
                          DeviceBuffer<float>& c, int n) {
        sgemmRegTiledNBC<64, 64, 8, 4, 4><<<g, 256>>>(a.data(), b.data(), c.data(), n, n, n);
    });
    return 0;
}
