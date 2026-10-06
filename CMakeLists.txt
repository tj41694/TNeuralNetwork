cmake_minimum_required(VERSION 3.15)
project(NumDistinguish LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)

# 训练可执行文件：只训练并写遥测，不含任何 HTTP 服务。默认工作目录 build/。
add_executable(NumDistinguish
    TNeuralNetworkEngine/main.cpp
    TNeuralNetworkEngine/CommandLine.cpp
    TNeuralNetworkEngine/NumDistinguish.cpp
    TNeuralNetworkEngine/Sample.cpp
    TNeuralNetworkEngine/Shuffle.cpp
    TNeuralNetworkEngine/TnRandom.cpp
    record/Recorder.cpp
    TNeuralNetworkEngine/sqlite3/sqlite3.c
)

# record/ 只负责把遥测写成文件。
target_include_directories(NumDistinguish PRIVATE
    TNeuralNetworkEngine
    record
)
target_link_libraries(NumDistinguish PRIVATE Threads::Threads)

# Web 服务可执行文件：独立进程，只读 runs/ 提供给浏览器，与训练互不引用。
add_executable(NumDistinguishWeb
    web/main.cpp
    web/DashboardServer.cpp
)
target_include_directories(NumDistinguishWeb PRIVATE web)
target_link_libraries(NumDistinguishWeb PRIVATE Threads::Threads)

if(WIN32)
    # cpp-httplib 的 Winsock 依赖。当前工具链是 GNU 风格的 clang++ 驱动，
    # 不认 #pragma comment(lib, ...)，所以必须在这里显式链接。
    target_link_libraries(NumDistinguishWeb PRIVATE ws2_32)
endif()
