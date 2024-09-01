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

add_executable(${PROJECT_NAME}
    src/log_mppi.cpp
)

# add_compile_options(-march=native)
target_compile_options(${PROJECT_NAME} PRIVATE -O3)

target_link_libraries(${PROJECT_NAME}
    ${PYTHON_LIBRARIES}
    OpenMP::OpenMP_CXX
)