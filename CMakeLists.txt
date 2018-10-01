cmake_minimum_required(VERSION 2.8)
project(Deblurring)

add_definitions(-DNUM_SINGLE=1)
add_definitions(-DTVREG_DECONV=1)
SET(CMAKE_C_FLAGS_DEBUG "-O3")
SET(CMAKE_C_FLAGS_RELEASE "-O3")
SET(CMAKE_CXX_FLAGS_DEBUG "-O3")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ftree-vectorize")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")
add_executable(Deblurring
        main.cpp
        angleSet.cpp
        computeProjectionsAutocorrelation.cpp
        deconvBregman.cpp
        estimateKernel.cpp
        ${CMAKE_CURRENT_LIST_DIR}/tvdeconv_20120607/randmt.c
        ${CMAKE_CURRENT_LIST_DIR}/tvdeconv_20120607/tvreg.c
        ${CMAKE_CURRENT_LIST_DIR}/tvdeconv_20120607/basic.c
        )
target_link_libraries(Deblurring -lfftw3 -lfftw3f -lm)