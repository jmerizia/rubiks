cmake_minimum_required(VERSION 3.5.1)

project(rubiks)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(/home/jake/git/dlib/ dlib_build)

set(SOURCES
    src/main.cpp
    src/helpers.cpp
    )

add_executable(rubiks ${SOURCES})

target_link_libraries(rubiks dlib::dlib)

target_include_directories(rubiks PRIVATE ${PROJECT_SOURCE_DIR}/include)

target_compile_options(rubiks PRIVATE -Werror -Wall -O3 -DLOGGING=1)
