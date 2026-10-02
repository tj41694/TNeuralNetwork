#include "DashboardServer.h"
#include "httplib/httplib.h"
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

namespace
{
const char *const kDataFiles[] = {"meta.json",           "status.json",   "scalars.jsonl",
                                  "histograms.bin",      "activations.bin", "probe_inputs.bin"};

bool ReadFileText(const std::string &path, std::string &out)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return true;
}

uint64_t FileSizeOrZero(const std::string &path)
{
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0u : static_cast<uint64_t>(size);
}

// 文件已经由 Recorder 写成 JSON，直接内联即可；缺失时给 null.
std::string JsonRawOrNull(const std::string &path)
{
    std::string text;
    if (!ReadFileText(path, text))
    {
        return "null";
    }
    return text;
}

std::string QuoteJson(const std::string &s)
{
    std::string out = "\"";
    for (char c : s)
    {
        if (c == '"' || c == '\\')
        {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

// 只接受单层目录名，杜绝 ".."、绝对路径与 Windows ADS 写法.
bool IsSafeRunName(const std::string &name)
{
    if (name.empty() || name.size() > 160 || name == "." || name == "..")
    {
        return false;
    }
    if (name.find("..") != std::string::npos)
    {
        return false;
    }
    for (char c : name)
    {
        const unsigned char u = static_cast<unsigned char>(c);
        if (!(std::isalnum(u) || c == '_' || c == '-' || c == '.'))
        {
            return false;
        }
    }
    return true;
}

// 只允许 web 根目录下的直接子文件，不允许任何路径分隔符.
bool IsSafeAssetName(const std::string &name)
{
    if (name.empty() || name.size() > 128)
    {
        return false;
    }
    if (name.find('/') != std::string::npos || name.find('\\') != std::string::npos)
    {
        return false;
    }
    if (name.find("..") != std::string::npos || name.find(':') != std::string::npos)
    {
        return false;
    }
    return true;
}

const char *MimeForName(const std::string &name)
{
    const auto dot = name.find_last_of('.');
    std::string ext = (dot == std::string::npos) ? std::string() : name.substr(dot + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    if (ext == "html")
    {
        return "text/html; charset=utf-8";
    }
    if (ext == "js" || ext == "mjs")
    {
        return "text/javascript; charset=utf-8";
    }
    if (ext == "css")
    {
        return "text/css; charset=utf-8";
    }
    if (ext == "json")
    {
        return "application/json; charset=utf-8";
    }
    if (ext == "svg")
    {
        return "image/svg+xml";
    }
    if (ext == "png")
    {
        return "image/png";
    }
    if (ext == "ico")
    {
        return "image/x-icon";
    }
    return "application/octet-stream";
}

std::string BuildStateJson(const std::string &runName, const std::string &runDir)
{
    std::string j = "{";
    j += "\"run\":" + QuoteJson(runName) + ",";
    j += "\"meta\":" + JsonRawOrNull(runDir + "/meta.json") + ",";
    j += "\"status\":" + JsonRawOrNull(runDir + "/status.json") + ",";
    j += "\"files\":{";
    for (size_t i = 0; i < std::size(kDataFiles); ++i)
    {
        if (i > 0)
        {
            j += ",";
        }
        j += QuoteJson(kDataFiles[i]) + ":{\"size\":" +
             std::to_string(FileSizeOrZero(runDir + "/" + kDataFiles[i])) + "}";
    }
    j += "}}";
    return j;
}

std::string BuildRunsJson(const std::string &runsRoot)
{
    // 目录名里带时间戳，但实验名前缀会压过时间戳，所以按名字排序得不到时间序；
    // 用目录 mtime 排序才能保证"最新的在前"。
    struct RunEntry
    {
        std::filesystem::path path;
        std::string name;
        int64_t mtimeMs = 0;
    };

    std::vector<RunEntry> entries;
    std::error_code ec;
    for (const auto &entry : std::filesystem::directory_iterator(runsRoot, ec))
    {
        std::error_code isDirEc;
        if (!entry.is_directory(isDirEc))
        {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (!IsSafeRunName(name))
        {
            continue;
        }
        std::error_code mtimeEc;
        const auto mtime = std::filesystem::last_write_time(entry.path(), mtimeEc);
        RunEntry item;
        item.path = entry.path();
        item.name = name;
        item.mtimeMs =
            mtimeEc ? 0
                    : static_cast<int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                               mtime.time_since_epoch())
                                               .count());
        entries.push_back(std::move(item));
    }
    std::sort(entries.begin(), entries.end(),
              [](const RunEntry &a, const RunEntry &b) { return a.mtimeMs > b.mtimeMs; });

    std::string j = "{\"runs\":[";
    for (size_t i = 0; i < entries.size(); ++i)
    {
        if (i > 0)
        {
            j += ",";
        }
        uint64_t bytes = 0;
        for (const char *file : kDataFiles)
        {
            bytes += FileSizeOrZero((entries[i].path / file).string());
        }
        j += "{\"name\":" + QuoteJson(entries[i].name) + ",\"bytes\":" + std::to_string(bytes) +
             ",\"mtimeMs\":" + std::to_string(entries[i].mtimeMs) + ",\"status\":" +
             JsonRawOrNull((entries[i].path / "status.json").string()) + "}";
    }
    j += "]}";
    return j;
}
} // namespace

struct DashboardServer::Impl
{
    httplib::Server server;
    std::thread worker;
    std::atomic<bool> running{false};
    std::string runsRoot;
    std::string webRoot;

    void ServeAsset(const std::string &name, httplib::Response &res) const
    {
        if (!IsSafeAssetName(name))
        {
            res.status = 403;
            res.set_content("{\"error\":\"bad asset name\"}", "application/json; charset=utf-8");
            return;
        }
        std::string body;
        if (!ReadFileText(webRoot + "/" + name, body))
        {
            res.status = 404;
            res.set_content("{\"error\":\"asset not found\"}", "application/json; charset=utf-8");
            return;
        }
        res.set_content(body, MimeForName(name));
    }

    void Setup()
    {
        // 只读：任何写方法直接 405
        server.set_pre_routing_handler([](const httplib::Request &req, httplib::Response &res) {
            if (req.method != "GET" && req.method != "HEAD")
            {
                res.status = 405;
                res.set_content("{\"error\":\"read-only server\"}", "application/json; charset=utf-8");
                return httplib::Server::HandlerResponse::Handled;
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });

        // 对所有响应统一禁止缓存；Range 的字节偏移一旦被浏览器缓存命中就会算错
        server.set_post_routing_handler([](const httplib::Request &, httplib::Response &res) {
            res.set_header("Cache-Control", "no-store");
            res.set_header("Accept-Ranges", "bytes");
        });

        server.set_file_extension_and_mimetype_mapping("jsonl", "text/plain; charset=utf-8");
        server.set_file_extension_and_mimetype_mapping("json", "application/json; charset=utf-8");
        server.set_file_extension_and_mimetype_mapping("bin", "application/octet-stream");

        // 注意：内建文件服务在路由之前执行，所以这里只挂 /runs，
        // 绝不能挂 "/"，否则会吞掉 /api/*。
        server.set_mount_point("/runs", runsRoot, httplib::Headers{{"Cache-Control", "no-store"}});

        server.Get("/", [this](const httplib::Request &, httplib::Response &res) {
            ServeAsset("index.html", res);
        });

        server.Get(R"(/([A-Za-z0-9_.\-]+\.(?:js|mjs|css|html|json|svg|png|ico)))",
                   [this](const httplib::Request &req, httplib::Response &res) {
                       ServeAsset(req.matches[1].str(), res);
                   });

        server.Get("/api/runs", [this](const httplib::Request &, httplib::Response &res) {
            res.set_content(BuildRunsJson(runsRoot), "application/json; charset=utf-8");
        });

        server.Get("/api/state", [this](const httplib::Request &req, httplib::Response &res) {
            const std::string run = req.has_param("run")
                                        ? req.get_param_value("run")
                                        : std::string();
            if (run.empty() || !IsSafeRunName(run))
            {
                res.status = 400;
                res.set_content("{\"error\":\"missing or invalid run\"}",
                                "application/json; charset=utf-8");
                return;
            }
            const std::string runDir = runsRoot + "/" + run;
            std::error_code ec;
            if (!std::filesystem::is_directory(runDir, ec))
            {
                res.status = 404;
                res.set_content("{\"error\":\"run not found\"}", "application/json; charset=utf-8");
                return;
            }
            res.set_content(BuildStateJson(run, runDir), "application/json; charset=utf-8");
        });
    }
};

DashboardServer::DashboardServer() : m_impl(new Impl())
{
}

DashboardServer::~DashboardServer()
{
    Stop();
}

bool DashboardServer::Start(const std::string &runsRoot, const std::string &webRoot, int port,
                            std::string &urlOut)
{
    m_impl->runsRoot = runsRoot;
    m_impl->webRoot = webRoot;

    std::error_code ec;
    std::filesystem::create_directories(runsRoot, ec);

    m_impl->Setup();

    int actualPort = 0;
    if (port > 0)
    {
        if (!m_impl->server.bind_to_port("127.0.0.1", port))
        {
            return false;
        }
        actualPort = port;
    }
    else
    {
        // 只绑 loopback：既避免 Windows 防火墙弹窗，也避免把 runs/ 暴露到局域网
        actualPort = m_impl->server.bind_to_any_port("127.0.0.1");
        if (actualPort <= 0)
        {
            return false;
        }
    }

    m_impl->running = true;
    m_impl->worker = std::thread([this]() {
        m_impl->server.listen_after_bind();
        m_impl->running = false;
    });

    urlOut = "http://127.0.0.1:" + std::to_string(actualPort) + "/";
    return true;
}

void DashboardServer::Stop()
{
    if (!m_impl)
    {
        return;
    }
    m_impl->server.stop();
    if (m_impl->worker.joinable())
    {
        m_impl->worker.join();
    }
    m_impl->running = false;
}

bool DashboardServer::IsRunning() const
{
    return m_impl && m_impl->running.load();
}
