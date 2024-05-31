cmake_minimum_required(VERSION 3.10)
project(CUDA_FeatureDetection)

list(APPEND CMAKE_PREFIX_PATH "opencv")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CLI11 CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(demo src/main.cpp)

target_link_libraries(
    demo
        PRIVATE
            ${OpenCV_LIBS}
            CLI11::CLI11
)