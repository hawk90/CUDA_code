// common/cli.cuh
// 재사용 코어: CLI 파싱. 긴 이름(쓰기 쉬움) + 짧은 별칭.
//   --size/-n N   --iters/-i N   --variant/-a KEY(반복)   --list/-l   --help/-h
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct BenchOptions {
    long n = (1L << 24);               // --size:    문제 크기
    int  iters = 100;                  // --iters:   측정 반복
    std::vector<std::string> variants; // --variant: 실행할 변형/레벨 key (비면 전체)
    bool listOnly = false;             // --list
    bool showHelp = false;             // --help
    bool valid = true;
};

class CliParser {
public:
    static void printUsage(const char* prog) {
        std::printf("Usage: %s [--size N] [--iters N] [--variant KEY]... [--list] [--help]\n", prog);
        std::printf("  --size,    -n N     문제 크기 (기본 1<<24)\n");
        std::printf("  --iters,   -i N     측정 반복 (기본 100)\n");
        std::printf("  --variant, -a KEY   실행할 변형/레벨 key (반복 지정 가능; 미지정 시 전체)\n");
        std::printf("                      예: --variant L2  → L2 레벨만 (레벨별 ncu 프로파일용)\n");
        std::printf("  --list,    -l       변형 목록 출력 후 종료\n");
        std::printf("  --help,    -h       도움말\n");
    }

    static BenchOptions parse(int argc, char** argv) {
        BenchOptions o;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if      (a == "--size"    || a == "-n") o.n     = std::atol(value(argc, argv, i, o));
            else if (a == "--iters"   || a == "-i") o.iters = std::atoi(value(argc, argv, i, o));
            else if (a == "--variant" || a == "-a") o.variants.push_back(value(argc, argv, i, o));
            else if (a == "--list"    || a == "-l") o.listOnly = true;
            else if (a == "--help"    || a == "-h") o.showHelp = true;
            else { std::fprintf(stderr, "unknown arg: %s (use --help)\n", a.c_str()); o.valid = false; }
            if (!o.valid) break;
        }
        return o;
    }

private:
    static const char* value(int argc, char** argv, int& i, BenchOptions& o) {
        if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", argv[i]); o.valid = false; return "0"; }
        return argv[++i];
    }
};

// custom-main 클라이언트용: 라벨 첫 토큰("L2 register-tiled" → "L2")이 --variant 로 선택됐는지.
// variants 비면 전체 실행. → `--variant L2` 로 한 레벨만 실행해 레벨별 ncu 프로파일 가능.
inline bool cliSelected(const BenchOptions& o, const std::string& label) {
    if (o.variants.empty()) return true;
    std::string tok = label.substr(0, label.find(' '));
    for (const auto& k : o.variants)
        if (k == tok) return true;
    return false;
}
