#include "CommandLine.h"
#include <cstdio>
#include <cstdlib>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace
{
#if defined(_WIN32)
// 终端里中文乱码的根因：源码与 printf 走的是 UTF-8 字节，而 Windows 控制台默认按系统
// ANSI 代码页（简中为 936/GBK）解码这些字节，两者不一致就成了"鐢ㄦ硶"这种乱码。
// 把控制台输出代码页切到 UTF-8 让两边对齐；退出时恢复原值，免得影响同一窗口后续的程序。
UINT g_consoleCpToRestore = 0; // 0 表示没改过，不需要恢复

void RestoreConsoleOutputCp()
{
    if (g_consoleCpToRestore != 0)
    {
        SetConsoleOutputCP(g_consoleCpToRestore);
    }
}

// 必须在任何 printf 之前调用：--help 与参数报错都要用中文.
void SetupConsoleOutputUtf8()
{
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    // 输出被重定向到文件或管道时没有控制台代码页这回事：UTF-8 字节本来就是想要的编码.
    if (hOut == nullptr || hOut == INVALID_HANDLE_VALUE || (GetConsoleMode(hOut, &mode) == 0))
    {
        return;
    }

    const UINT previous = GetConsoleOutputCP();
    if (previous == 0 || previous == CP_UTF8 || (SetConsoleOutputCP(CP_UTF8) == 0))
    {
        return;
    }

    g_consoleCpToRestore = previous;
    std::atexit(RestoreConsoleOutputCp);
}
#else
void SetupConsoleOutputUtf8()
{
}
#endif

void PrintUsage()
{
    printf("用法: NumDistinguish [选项]\n");
    printf("  --exp NAME       实验名，输出到 runs/<NAME>_<时间戳>/ （默认 exp）\n");
    printf("  --seed N         随机种子；省略则自动派生并记入 meta.json\n");
    printf("  --steps N        训练步数，1 步 = 1 个 batch （默认 25000）\n");
    printf("  --batch N        每批样本数 （默认 100）\n");
    printf("  --out DIR        遥测输出根目录 （默认 ../runs，即仓库根的 runs/）\n");
    printf("  --probes N       激活快照使用的固定样本数 （默认 16，0 表示关闭）\n");
    printf("  --hist-range R   直方图固定分箱范围 ±R；一个值用于所有层，"
           "或用逗号按层各给一个（默认按层 1,1.5,1.5,2）\n");
    printf("提示: 浏览器查看遥测请单独运行 NumDistinguishWeb（见 web/README.md）\n");
}

bool ParseUint(const char *text, uint32_t &out)
{
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || (end != nullptr && *end != '\0'))
    {
        return false;
    }
    out = static_cast<uint32_t>(value);
    return true;
}
} // namespace

bool ParseArgs(int argc, char **argv, Options &opt)
{
    SetupConsoleOutputUtf8();

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto requireValue = [&](const char *name) -> const char * {
            if (i + 1 >= argc)
            {
                printf("参数 %s 缺少取值\n", name);
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h")
        {
            PrintUsage();
            exit(0);
        }
        else if (arg == "--exp")
        {
            const char *v = requireValue("--exp");
            if (v == nullptr)
            {
                return false;
            }
            opt.expName = v;
        }
        else if (arg == "--out")
        {
            const char *v = requireValue("--out");
            if (v == nullptr)
            {
                return false;
            }
            opt.outRoot = v;
        }
        else if (arg == "--seed")
        {
            const char *v = requireValue("--seed");
            uint32_t value = 0;
            if (v == nullptr || !ParseUint(v, value))
            {
                return false;
            }
            opt.seed = value;
            opt.seedGiven = true;
        }
        else if (arg == "--steps")
        {
            const char *v = requireValue("--steps");
            if (v == nullptr || !ParseUint(v, opt.steps))
            {
                return false;
            }
        }
        else if (arg == "--batch")
        {
            const char *v = requireValue("--batch");
            uint32_t value = 0;
            if (v == nullptr || !ParseUint(v, value))
            {
                return false;
            }
            opt.batchSize = static_cast<int>(value);
        }
        else if (arg == "--probes")
        {
            const char *v = requireValue("--probes");
            if (v == nullptr || !ParseUint(v, opt.probes))
            {
                return false;
            }
        }
        else if (arg == "--hist-range")
        {
            const char *v = requireValue("--hist-range");
            if (v == nullptr)
            {
                return false;
            }
            // 一个值 = 所有层；逗号分隔 = 按层各一个
            const std::string spec = v;
            std::vector<double> ranges;
            size_t start = 0;
            while (start <= spec.size())
            {
                const size_t comma = spec.find(',', start);
                const size_t len = (comma == std::string::npos) ? std::string::npos : comma - start;
                const std::string part = spec.substr(start, len);
                char *end = nullptr;
                const double value = std::strtod(part.c_str(), &end);
                if (part.empty() || end == part.c_str() || *end != '\0' || !(value > 0.0))
                {
                    printf("--hist-range 取值非法: %s（应为正数，或用逗号分隔每层一个）\n",
                           spec.c_str());
                    return false;
                }
                ranges.push_back(value);
                if (comma == std::string::npos)
                {
                    break;
                }
                start = comma + 1;
            }
            opt.histRanges = ranges;
        }
        else
        {
            printf("未知参数: %s\n", arg.c_str());
            PrintUsage();
            return false;
        }
    }
    return true;
}
