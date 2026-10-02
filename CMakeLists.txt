cmake_minimum_required(VERSION 3.15)
project(NumDistinguish LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(NumDistinguish
    NumDistinguish/main.cpp
    NumDistinguish/NumDistinguish.cpp
    NumDistinguish/Sample.cpp
    NumDistinguish/Shuffle.cpp
    NumDistinguish/TnRandom.cpp
    record/Recorder.cpp
    web/DashboardServer.cpp
    NumDistinguish/sqlite3/sqlite3.c
)

# record/ 与 web/ 各自是一个自洽的功能目录（见各自的 README.md）：
# record/ 只负责把遥测写成文件，web/ 只负责把这些文件只读地提供给浏览器。
target_include_directories(NumDistinguish PRIVATE
    NumDistinguish
    record
    web
)

find_package(Threads REQUIRED)
target_link_libraries(NumDistinguish PRIVATE Threads::Threads)

if(WIN32)
    # cpp-httplib 的 Winsock 依赖。当前工具链是 GNU 风格的 clang++ 驱动，
    # 不认 #pragma comment(lib, ...)，所以必须在这里显式链接。
    target_link_libraries(NumDistinguish PRIVATE ws2_32)
endif()
