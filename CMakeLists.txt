cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ldl")
project(track)

set(CMAKE_CXX_STANDARD 14)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS"${CMAKE_EXE_LINKER_FLAGS}${OpenMP_EXE_LINKER_FLAGS}")
endif()

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(RKNN_RT_LIB /home/toybrick/lrtrack_cv/lib/librknnrt.so)


message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/lib)

add_executable(track 
    src/main.cpp 
    src/mathes.cpp
    )

target_link_libraries(track ${OpenCV_LIBS} ${RKNN_RT_LIB} -fopenmp)

# -lrknn_api -lade -llibjasper -llibjpeg-turbo -llibpng -llibprotobuf -llibtiff -llibwebp -lquirc -ltegra_hal -lzlib -lavformat -lavcodec -lavdevice -lavfilter -lavutil -lswresample -lswscale

