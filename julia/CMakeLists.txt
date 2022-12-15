add_library (ExaFMMCInterface SHARED ExaFMMCInterface.cpp)
add_library (ExaFMMCInterface32 SHARED ExaFMMCInterface.cpp)

target_compile_definitions(ExaFMMCInterface32 PUBLIC FLOAT)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(ExaFMMCInterface32 PUBLIC -msse3)
endif()

target_link_libraries (ExaFMMCInterface PRIVATE ${BLAS_LIBRARIES}
                                        PRIVATE ${LAPACK_LIBRARIES}
                                        PRIVATE ${FFTW_LIBRARIES})

target_link_libraries (ExaFMMCInterface32
        PRIVATE ${BLAS_LIBRARIES}
        PRIVATE ${LAPACK_LIBRARIES}
        PRIVATE ${FFTW_LIBRARIES}
)

install(TARGETS ExaFMMCInterface 
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

install(TARGETS ExaFMMCInterface32 
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
