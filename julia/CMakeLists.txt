cmake_minimum_required(VERSION 3.10)
project(exafmm_c_interface C CXX)

# cmake settings
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_COLOR_MAKEFILE on)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../cmake-modules")


set(CMAKE_CXX_STANDARD 11)

# check BLAS and LAPACK
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(FFTW REQUIRED)

# define targets
add_library (ExaFMMCInterface SHARED ExaFMMCInterface.cpp)
add_library (ExaFMMCInterface32 SHARED ExaFMMCInterface.cpp)

target_include_directories(ExaFMMCInterface PUBLIC ${PROJECT_SOURCE_DIR}/../include)
target_include_directories(ExaFMMCInterface32 PUBLIC ${PROJECT_SOURCE_DIR}/../include)

target_compile_definitions(ExaFMMCInterface32 PUBLIC FLOAT)

# check compiler support c++11 standard
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if(COMPILER_SUPPORTS_MARCH_NATIVE)
        target_compile_options(ExaFMMCInterface PUBLIC -march=native)
        target_compile_options(ExaFMMCInterface32 PUBLIC -march=native)
else()
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                target_compile_options(ExaFMMCInterface32 PUBLIC -msse3)
        endif()
endif()

target_link_libraries (ExaFMMCInterface PRIVATE ${BLAS_LIBRARIES}
                                        PRIVATE ${LAPACK_LIBRARIES}
                                        PRIVATE ${FFTW_LIBRARIES}
                                        PRIVATE OpenMP::OpenMP_CXX)

target_link_libraries (ExaFMMCInterface
        PUBLIC OpenMP::OpenMP_CXX
)

target_link_libraries (ExaFMMCInterface32
        PUBLIC OpenMP::OpenMP_CXX
)

target_link_libraries (ExaFMMCInterface32
        PRIVATE ${BLAS_LIBRARIES}
        PRIVATE ${LAPACK_LIBRARIES}
        PRIVATE ${FFTW_LIBRARIES}
        PRIVATE OpenMP::OpenMP_CXX
)

install(TARGETS ExaFMMCInterface 
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

install(TARGETS ExaFMMCInterface32 
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
