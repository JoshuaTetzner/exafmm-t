.SUFFIXES: .cpp

WFLAGS = -fmudflap -fno-strict-aliasing -fsanitize=address -fsanitize=leak -fstack-protector -ftrapv -Wall -Warray-bounds -Wcast-align -Wcast-qual -Wextra -Wfatal-errors -Wformat=2 -Wformat-nonliteral -Wformat-security -Winit-self -Wmissing-format-attribute -Wmissing-include-dirs -Wmissing-noreturn -Wno-missing-field-initializers -Wno-overloaded-virtual -Wno-unused-local-typedefs -Wno-unused-parameter -Wno-unused-variable -Wpointer-arith -Wredundant-decls -Wreturn-type -Wshadow -Wstrict-aliasing -Wstrict-overflow=5 -Wswitch-enum -Wuninitialized -Wunreachable-code -Wunused-but-set-variable -Wwrite-strings -Wno-error=missing-field-initializers -Wno-error=overloaded-virtual -Wno-error=unused-local-typedefs -Wno-error=unused-parameter -Wno-error=unused-variable
# -Wsign-compare -Werror

CXX = mpic++ -g -O3 -mavx -fabi-version=6 -std=gnu++11 -fopenmp -I./include
#CXX = g++-mp-5 -g -O3 -msse4 -std=c++11 -fopenmp -I./include
LDFLAGS = -lfftw3 -lfftw3f -llapack -lblas

%.o: %.cpp
	$(CXX) -c $< -o $@

pfloat: main.cpp
	time $(CXX) -c $< -o main.o -DFLOAT -DPOTENTIAL

pdouble: main.cpp
	time $(CXX) -c $< -o main.o -DPOTENTIAL

float: main.cpp
	time $(CXX) -c $< -o main.o -DFLOAT

double: main.cpp
	time $(CXX) -c $< -o main.o

link: main.o
	$(CXX) $? $(LDFLAGS)

clean:
	rm -f *.o *.out

cleandat:
	rm -f $(PVFMM_DIR)/*f.data

tags:
	find . -name "*.cpp" -o -name "*.hpp" | xargs etags -f TAGS

p4:
	./a.out -T 8 -n 1000000 -P 4 -c 64

p16:
	./a.out -T 8 -n 1000000 -P 16 -c 320

y4:
	./a.out -T 32 -n 1000000 -P 4 -c 64

y16:
	./a.out -T 32 -n 1000000 -P 16 -c 320
