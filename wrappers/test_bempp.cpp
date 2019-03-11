#include <iostream>
#include "args.h"
#include "exafmm_t.h"

using namespace exafmm_t;

extern "C" Bodies array_to_bodies(size_t count, real_t* coord, real_t* value, bool is_source=true);
extern "C" void init_FMM(Args& args);
extern "C" Nodes setup_FMM(Bodies& sources, Bodies& targets, NodePtrs& leafs, Args& args);
extern "C" void run_FMM(Nodes& nodes, NodePtrs& leafs);

int main(int argc, char **argv) {
  // initialize global variables
  size_t N = 1000000;
  Args args;
  init_FMM(args);

  // generate random coordinates and charges
  real_t * src_coord = new real_t [3*N];
  real_t * src_q = new real_t [N];
  real_t * trg_coord = new real_t [3*N];
  real_t * trg_p = new real_t [N];
  for(size_t i=0; i<N; ++i) {
    for(int d=0; d<3; ++d) {
      src_coord[3*i+d] = drand48();
      trg_coord[3*i+d] = drand48();
    }
    src_q[i] = drand48();
  }
  Bodies sources = array_to_bodies(N, src_coord, src_q);
  Bodies targets = array_to_bodies(N, trg_coord, trg_p, false);

  // setup FMM
  NodePtrs leafs;
  Nodes nodes = setup_FMM(sources, targets, leafs, args);

  // run FMM
  run_FMM(nodes, leafs);

  // delete arrays
  delete[] src_coord;
  delete[] src_q;
  delete[] trg_coord;
  delete[] trg_p;
  
  return 0;
}
