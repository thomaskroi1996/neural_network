cmake_minimum_required(VERSION 3.16)
project(myNN LANGUAGES CXX)

# === Build settings ===
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type (if not set)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# === Options ===
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_EXAMPLES "Build example programs" ON)

# === Library target ===
file(GLOB_RECURSE TINNN_SRC CONFIGURE_DEPENDS src/*.cpp)
add_library(myNN STATIC ${TINNN_SRC})
target_include_directories(myNN PUBLIC include)
target_compile_options(myNN PRIVATE -Wall -Wextra -Wpedantic)

# === Example executables ===
if(BUILD_EXAMPLES)
    file(GLOB EXAMPLE_SRC CONFIGURE_DEPENDS examples/*.cpp)
    foreach(example ${EXAMPLE_SRC})
        get_filename_component(example_name ${example} NAME_WE)
        add_executable(${example_name} ${example})
        target_link_libraries(${example_name} PRIVATE myNN)
    endforeach()
endif()

# === Tests ===
if(BUILD_TESTS)
    enable_testing()
    file(GLOB TEST_SRC CONFIGURE_DEPENDS tests/*.cpp)
    add_executable(myNN_tests ${TEST_SRC})
    target_link_libraries(myNN_tests PRIVATE myNN)
    add_test(NAME myNN_RunTests COMMAND myNN_tests)
endif()
