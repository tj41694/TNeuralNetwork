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
    NumDistinguish/sqlite3/sqlite3.c
)

target_include_directories(NumDistinguish PRIVATE NumDistinguish)
