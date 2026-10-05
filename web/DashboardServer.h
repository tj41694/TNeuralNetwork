#pragma once
#include <memory>
#include <string>

// 只读遥测服务：把 runs/ 目录按 HTTP 暴露给浏览器。
// 它只认文件系统，不引用任何训练对象，因此与训练线程之间不需要锁。
class DashboardServer
{
  public:
    DashboardServer();
    ~DashboardServer();
    DashboardServer(const DashboardServer &) = delete;
    DashboardServer &operator=(const DashboardServer &) = delete;

    // port > 0 时优先绑定该端口，被占用则回退到系统分配的空闲端口；port <= 0 直接自动分配。
    // 成功时 urlOut 填成 http://127.0.0.1:<port>/
    bool Start(const std::string &runsRoot, const std::string &webRoot, int port,
               std::string &urlOut);
    void Stop();
    bool IsRunning() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
