add_library (ExaFMMCInterface SHARED ExaFMMCInterface.cpp)

target_link_libraries (ExaFMMCInterface PRIVATE ${BLAS_LIBRARIES}
                                        PRIVATE ${LAPACK_LIBRARIES}
                                        PRIVATE ${FFTW_LIBRARIES})

install(TARGETS ExaFMMCInterface 
        LIBRARY DESTINATION /home/jt286/Documents/Code/C++/exafmm-t/lib
)
