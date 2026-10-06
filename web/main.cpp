// NumDistinguishWeb 入口：独立于训练进程的只读遥测服务。
// 它只把 runs/ 下的文件通过 HTTP 暴露给浏览器，不训练、不碰数据库；
// 与 NumDistinguish 之间唯一的接口就是 runs/ 里的文件，不需要共享内存或锁。
#include "DashboardServer.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace
{
struct WebOptions
{
    std::string runsRoot = "../runs";
    std::string webRoot = "../web";
    int port = 5108;
};

std::atomic<bool> g_stopRequested{false};

#if defined(_WIN32)
BOOL WINAPI ConsoleHandler(DWORD type)
{
    if (type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT || type == CTRL_CLOSE_EVENT)
    {
        // 只置标志位：控制台处理函数跑在另一个线程上，在这里做 I/O 有死锁风险.
        g_stopRequested.store(true);
        return TRUE;
    }
    return FALSE;
}

// 与训练入口 CommandLine.cpp 里同样的处理：源码/printf 走 UTF-8 字节，而 Windows 控制台
// 默认按 ANSI 代码页解码，中文会乱码。切到 UTF-8 并在退出时恢复。
UINT g_consoleCpToRestore = 0;

void RestoreConsoleOutputCp()
{
    if (g_consoleCpToRestore != 0)
    {
        SetConsoleOutputCP(g_consoleCpToRestore);
    }
}

void SetupConsoleOutputUtf8()
{
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
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
    printf("用法: NumDistinguishWeb [选项]\n");
    printf("  --out DIR   遥测根目录（只读） （默认 ../runs，即仓库根的 runs/）\n");
    printf("  --web DIR   前端静态文件目录 （默认 ../web）\n");
    printf("  --port N    HTTP 端口；默认 5108，被占用时自动改用空闲端口；0 表示直接自动分配\n");
}

bool ParseUint(const char *text, int &out)
{
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (end == text || (end != nullptr && *end != '\0'))
    {
        return false;
    }
    out = static_cast<int>(value);
    return true;
}

bool ParseArgs(int argc, char **argv, WebOptions &opt)
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
        else if (arg == "--out")
        {
            const char *v = requireValue("--out");
            if (v == nullptr)
            {
                return false;
            }
            opt.runsRoot = v;
        }
        else if (arg == "--web")
        {
            const char *v = requireValue("--web");
            if (v == nullptr)
            {
                return false;
            }
            opt.webRoot = v;
        }
        else if (arg == "--port")
        {
            const char *v = requireValue("--port");
            if (v == nullptr || !ParseUint(v, opt.port))
            {
                return false;
            }
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

// 保持服务存活，方便浏览器继续看。
// 交互式终端按回车退出；非交互式（后台运行/管道）读不到输入就退化为一直存活，
// 靠 Ctrl+C 或外部终止结束。
void HoldServerAlive()
{
    printf("\n按回车退出（非交互式运行则保持存活，Ctrl+C 亦可）...\n");
    std::string line;
    std::getline(std::cin, line);
    if (!std::cin.eof())
    {
        return;
    }
    while (!g_stopRequested.load())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}
} // namespace

int main(int argc, char **argv)
{
    WebOptions opt;
    if (!ParseArgs(argc, argv, opt))
    {
        return 2;
    }

#if defined(_WIN32)
    SetConsoleCtrlHandler(ConsoleHandler, TRUE);
#endif

    DashboardServer server;
    std::string url;
    if (!server.Start(opt.runsRoot, opt.webRoot, opt.port, url))
    {
        printf("HTTP 服务启动失败。\n");
        return 3;
    }
    printf("仪表盘: %s\n", url.c_str());
    HoldServerAlive();
    server.Stop();
    printf("done..\n");
    return 0;
}
