#include <iostream>
#include<unistd.h>
#include "exafmm_t.h"
#include "dataset.h"
#include "laplace.h"
#if NON_ADAPTIVE
#include "build_non_adaptive_tree.h"
#else
#include "build_tree.h"
#endif
#include "build_list.h"
#include "laplace.h"
#include "helmholtz.h"
#include "modified_helmholtz.h"
using namespace std;

using real_t = exafmm_t::real_t;
using complex_t = exafmm_t::complex_t;
template <typename T> using Body = exafmm_t::Body<T>;
template <typename T> using Bodies = exafmm_t::Bodies<T>;
template <typename T> using Node = exafmm_t::Node<T>;
template <typename T> using Nodes = exafmm_t::Nodes<T>;
template <typename T> using NodePtrs = exafmm_t::NodePtrs<T>;

extern "C" {
  void* init_sources_F64(real_t* coords , real_t* charges, int nsrcs);
  void* init_sources_C64(real_t* coords , complex_t* charges, int nsrcs);
  void* init_targets_F64(real_t* coords, int ntarg);
  void* init_targets_C64(real_t* coords, int ntarg);
  void* LaplaceFMM(int p, int ncrit);
  void* HelmholtzFMM(int p, int ncrit, complex_t wavek);
  void* setup_laplace(void* src, void* trg, void* pfmm);
  void* setup_helmholtz(void* src, void* trg, void* pfmm);
  real_t* evaluate_laplace(void* fmmstruct);
  complex_t* evaluate_helmholtz(void* fmmstruct);
  void update_charges_real(void* fmm, real_t* charges);
  void update_charges_cplx(void* fmm, complex_t* charges);
  void clear_values(void* fmm);
}

void* init_sources_F64(real_t* coords , real_t* charges, int nsrcs) {
  Bodies<real_t>* sources = new Bodies<real_t>(nsrcs);
  #pragma omp parallel for
  for(ssize_t i=0; i<nsrcs; ++i) {
    (*sources)[i].X[0] = coords[i+nsrcs*0];
    (*sources)[i].X[1] = coords[i+nsrcs*1];
    (*sources)[i].X[2] = coords[i+nsrcs*2];
    (*sources)[i].q = charges[i];
    (*sources)[i].ibody = i;
  }

  return reinterpret_cast<void*> (sources);
}

void* init_sources_C64(real_t* coords , complex_t* charges, int nsrcs) {
  Bodies<complex_t>* sources = new Bodies<complex_t>(nsrcs);
  #pragma omp parallel for
  for(ssize_t i=0; i<nsrcs; ++i) {
    (*sources)[i].X[0] = coords[i+nsrcs*0];
    (*sources)[i].X[1] = coords[i+nsrcs*1];
    (*sources)[i].X[2] = coords[i+nsrcs*2];
    (*sources)[i].q = charges[i];
    (*sources)[i].ibody = i;
  }

  return reinterpret_cast<void*> (sources);
}

void* init_targets_F64(real_t* coords, int ntarg) {
  Bodies<real_t>* sources = new Bodies<real_t>(ntarg);
  #pragma omp parallel for
  for(ssize_t i=0; i<ntarg; ++i) {
    (*sources)[i].X[0] = coords[i+ntarg*0];
    (*sources)[i].X[1] = coords[i+ntarg*1];
    (*sources)[i].X[2] = coords[i+ntarg*2];
    (*sources)[i].ibody = i;
  }

  return reinterpret_cast<void*> (sources);
}

void* init_targets_C64(real_t* coords, int ntarg) {
  Bodies<complex_t>* sources = new Bodies<complex_t>(ntarg);
  #pragma omp parallel for
  for(ssize_t i=0; i<ntarg; ++i) {
    (*sources)[i].X[0] = coords[i+ntarg*0];
    (*sources)[i].X[1] = coords[i+ntarg*1];
    (*sources)[i].X[2] = coords[i+ntarg*2];
    (*sources)[i].ibody = i;
  }

  return reinterpret_cast<void*> (sources);
}

template <typename T>
struct Tree {
  Nodes<T> nodes;          //!< Vector of all nodes in the tree
  NodePtrs<T> leafs;       //!< Vector of leaf pointers
  NodePtrs<T> nonleafs;    //!< Vector of nonleaf pointers
};

template <typename T>
Tree<T> build_tree(Bodies<T>& sources, Bodies<T>& targets, exafmm_t::FmmBase<T>& fmm) {
  exafmm_t::get_bounds<T>(sources, targets, fmm.x0, fmm.r0);
  Tree<T> tree;
  tree.nodes = exafmm_t::build_tree<T>(sources, targets, tree.leafs, tree.nonleafs, fmm);
  return tree;
}

template <typename T>
void build_list(Tree<T>& tree, exafmm_t::FmmBase<T>& fmm) {
  exafmm_t::build_list<T>(tree.nodes, fmm);
}

void* LaplaceFMM(int p, int ncrit){
  exafmm_t::LaplaceFmm* fmm = new exafmm_t::LaplaceFmm(p, ncrit);
  return reinterpret_cast<void*> (fmm);
}

void* HelmholtzFMM(int p, int ncrit, complex_t wavek){
  exafmm_t::HelmholtzFmm* fmm = new exafmm_t::HelmholtzFmm(p, ncrit, wavek);
  return reinterpret_cast<void*> (fmm);
}

void* ModifiedHelmholtzFMM(int p, int ncrit, real_t wavek){
  exafmm_t::ModifiedHelmholtzFmm* fmm = new exafmm_t::ModifiedHelmholtzFmm(p, ncrit, wavek);
  return reinterpret_cast<void*> (fmm);
}

struct Laplace {
  exafmm_t::LaplaceFmm* fmm;
  Tree<real_t>* tree;
};

struct Helmholtz {
  exafmm_t::HelmholtzFmm* fmm;
  Tree<complex_t>* tree;
};

struct ModifiedHelmholtz {
  exafmm_t::ModifiedHelmholtzFmm* fmm;
  Tree<real_t>* tree;
};

void* setup_laplace(void* src, void* trg, void* pfmm)
{
  Bodies<real_t>* sources = reinterpret_cast<Bodies<real_t>*> (src);
  Bodies<real_t>* targets = reinterpret_cast<Bodies<real_t>*> (trg);
  exafmm_t::LaplaceFmm* fmm = reinterpret_cast<exafmm_t::LaplaceFmm*> (pfmm);

  Tree<real_t>* tree = new Tree<real_t>(build_tree<real_t>((*sources), (*targets), (*fmm)));
  exafmm_t::init_rel_coord();
  build_list<real_t>((*tree), (*fmm));
  fmm->M2L_setup(tree->nonleafs);
  fmm->precompute();
  Laplace* laplacefmm = new Laplace();
  laplacefmm->fmm = fmm;
  laplacefmm->tree = tree;
  return reinterpret_cast<void*> (laplacefmm);
}

void* setup_helmholtz(void* src, void* trg, void* pfmm)
{
  Bodies<complex_t>* sources = reinterpret_cast<Bodies<complex_t>*> (src);
  Bodies<complex_t>* targets = reinterpret_cast<Bodies<complex_t>*> (trg);
  exafmm_t::HelmholtzFmm* fmm = reinterpret_cast<exafmm_t::HelmholtzFmm*> (pfmm);

  Tree<complex_t>* tree = new Tree<complex_t>(build_tree<complex_t>((*sources), (*targets), (*fmm)));
  exafmm_t::init_rel_coord();
  build_list<complex_t>((*tree), (*fmm));
  fmm->M2L_setup(tree->nonleafs);
  fmm->precompute();
  Helmholtz* helmholtzfmm = new Helmholtz();
  helmholtzfmm->fmm = fmm;
  helmholtzfmm->tree = tree;
  return reinterpret_cast<void*> (helmholtzfmm);
}

void* setup_modifiedhelmholtz(void* src, void* trg, void* pfmm)
{
  Bodies<real_t>* sources = reinterpret_cast<Bodies<real_t>*> (src);
  Bodies<real_t>* targets = reinterpret_cast<Bodies<real_t>*> (trg);
  exafmm_t::ModifiedHelmholtzFmm* fmm = reinterpret_cast<exafmm_t::ModifiedHelmholtzFmm*> (pfmm);

  Tree<real_t>* tree = new Tree<real_t>(build_tree<real_t>((*sources), (*targets), (*fmm)));
  exafmm_t::init_rel_coord();
  build_list<real_t>((*tree), (*fmm));
  fmm->M2L_setup(tree->nonleafs);
  fmm->precompute();
  ModifiedHelmholtz* modifiedhelmholtzfmm = new ModifiedHelmholtz();
  modifiedhelmholtzfmm->fmm = fmm;
  modifiedhelmholtzfmm->tree = tree;
  return reinterpret_cast<void*> (modifiedhelmholtzfmm);
}

real_t* evaluate_laplace(void* laplace) {

  Laplace* laplacefmm = reinterpret_cast<Laplace*> (laplace);
  laplacefmm->fmm->upward_pass(laplacefmm->tree->nodes, laplacefmm->tree->leafs, false);
  laplacefmm->fmm->downward_pass(laplacefmm->tree->nodes, laplacefmm->tree->leafs, false);
  
  int ntrg = (laplacefmm->tree->nodes[0].ntrgs);
  real_t* trg_value = new real_t [4*ntrg];
  
  #pragma omp parallel for
  for (size_t i=0; i<laplacefmm->tree->leafs.size(); ++i) {
    Node<real_t>* leaf = laplacefmm->tree->leafs[i];
    std::vector<int> & itrgs = leaf->itrgs;

    for (size_t j=0; j<itrgs.size(); ++j) {
      trg_value[itrgs[j]] = leaf->trg_value[4*j+0];
      trg_value[ntrg + itrgs[j]] = leaf->trg_value[4*j+1];
      trg_value[2*ntrg + itrgs[j]] = leaf->trg_value[4*j+2];
      trg_value[3*ntrg + itrgs[j]] = leaf->trg_value[4*j+3];
    }
  }

  return trg_value;
}

complex_t* evaluate_helmholtz(void* helmholtz) {

  Helmholtz* helmholtzfmm = reinterpret_cast<Helmholtz*> (helmholtz);
  helmholtzfmm->fmm->upward_pass(helmholtzfmm->tree->nodes, helmholtzfmm->tree->leafs, false);
  helmholtzfmm->fmm->downward_pass(helmholtzfmm->tree->nodes, helmholtzfmm->tree->leafs, false);

  int ntrg = (helmholtzfmm->tree->nodes[0].ntrgs);
  complex_t* trg_value = new complex_t [4*ntrg];

  #pragma omp parallel for
  for (size_t i=0; i<helmholtzfmm->tree->leafs.size(); ++i) {
    Node<complex_t>* leaf = helmholtzfmm->tree->leafs[i];
    std::vector<int> & itrgs = leaf->itrgs;
    
    for (size_t j=0; j<itrgs.size(); ++j) {
      trg_value[itrgs[j]] = leaf->trg_value[4*j+0];
      trg_value[ntrg + itrgs[j]] = leaf->trg_value[4*j+1];
      trg_value[2*ntrg + itrgs[j]] = leaf->trg_value[4*j+2];
      trg_value[3*ntrg + itrgs[j]] = leaf->trg_value[4*j+3];
    }
  }

  return trg_value;
}

real_t* evaluate_modifiedhelmholtzfmm(void* helmholtz) {

  ModifiedHelmholtz* modifiedhelmholtzfmm = reinterpret_cast<ModifiedHelmholtz*> (helmholtz);
  modifiedhelmholtzfmm->fmm->upward_pass(modifiedhelmholtzfmm->tree->nodes, modifiedhelmholtzfmm->tree->leafs, false);
  modifiedhelmholtzfmm->fmm->downward_pass(modifiedhelmholtzfmm->tree->nodes, modifiedhelmholtzfmm->tree->leafs, false);

  int ntrg = (modifiedhelmholtzfmm->tree->nodes[0].ntrgs);
  real_t* trg_value = new real_t [4*ntrg];
  
  #pragma omp parallel for
  for (size_t i=0; i<modifiedhelmholtzfmm->tree->leafs.size(); ++i) {
    Node<real_t>* leaf = modifiedhelmholtzfmm->tree->leafs[i];
    std::vector<int> & itrgs = leaf->itrgs;
    
    for (size_t j=0; j<itrgs.size(); ++j) {
      trg_value[itrgs[j]] = leaf->trg_value[4*j+0];
      trg_value[ntrg + itrgs[j]] = leaf->trg_value[4*j+1];
      trg_value[2*ntrg + itrgs[j]] = leaf->trg_value[4*j+2];
      trg_value[3*ntrg + itrgs[j]] = leaf->trg_value[4*j+3];
    }
  }

  return trg_value;
}

void update_charges_real(void* fmm, real_t* charges) {
  // update charges of sources
  Laplace* laplacefmm = reinterpret_cast<Laplace*> (fmm);
#pragma omp parallel for
  for (size_t i=0; i<laplacefmm->tree->leafs.size(); ++i) {
    auto leaf = laplacefmm->tree->leafs[i];
    std::vector<int>& isrcs = leaf->isrcs;
    for (size_t j=0; j<isrcs.size(); ++j) {
      leaf->src_value[j] = charges[isrcs[j]];
    }
  }
}

void update_charges_cplx(void* fmm, complex_t* charges) {
  // update charges of sources
  Helmholtz* helmholtzfmm = reinterpret_cast<Helmholtz*> (fmm);
#pragma omp parallel for
  for (size_t i=0; i<helmholtzfmm->tree->leafs.size(); ++i) {
    auto leaf = helmholtzfmm->tree->leafs[i];
    std::vector<int>& isrcs = leaf->isrcs;
    for (size_t j=0; j<isrcs.size(); ++j) {
      leaf->src_value[j] = charges[isrcs[j]];
    }
  }
}


void clear_values(void* fmm) {
  Laplace* laplacefmm = reinterpret_cast<Laplace*> (fmm);
#pragma omp parallel for
  for (size_t i=0; i<laplacefmm->tree->nodes.size(); ++i) {
    auto& node = laplacefmm->tree->nodes[i];
    std::fill(node.up_equiv.begin(), node.up_equiv.end(), 0.);
    std::fill(node.dn_equiv.begin(), node.dn_equiv.end(), 0.);
    if (node.is_leaf)
      std::fill(node.trg_value.begin(), node.trg_value.end(), 0.);
  }
}