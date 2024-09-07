cmake_minimum_required(VERSION 3.5)
project(mppi)

find_package(PythonLibs REQUIRED)
find_package (Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)

include_directories(
    mppi/
    model/

    EigenRand/
    ${EIGEN3_INCLUDE_DIRS}
    matplotlibcpp/
    ${PYTHON_INCLUDE_DIRS}
)

set(COMMON_SOURCES
    src/bi_mppi.cpp
    src/mppi.cpp
    src/log_mppi.cpp
    src/cluster_mppi.cpp
)

foreach(SRC ${COMMON_SOURCES})
    get_filename_component(EXE_NAME ${SRC} NAME_WE)
    add_executable(${EXE_NAME} ${SRC})
    target_compile_options(${EXE_NAME} PRIVATE -O3)
    target_link_libraries(${EXE_NAME}     
        ${PYTHON_LIBRARIES}
        OpenMP::OpenMP_CXX
    )
endforeach()