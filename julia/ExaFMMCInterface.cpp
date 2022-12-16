#include <iostream>
#include <unistd.h>
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
  Bodies<real_t>* init_sources_F(real_t* coords , real_t* charges, int nsrcs);
  Bodies<complex_t>* init_sources_C(real_t* coords , complex_t* charges, int nsrcs);
  Bodies<real_t>* init_targets_F(real_t* coords, int ntarg);
  Bodies<complex_t>* init_targets_C(real_t* coords, int ntarg);
  exafmm_t::LaplaceFmm* LaplaceFMM(int p, int ncrit);
  exafmm_t::HelmholtzFmm* HelmholtzFMM(int p, int ncrit, complex_t wavek);
  exafmm_t::ModifiedHelmholtzFmm* ModifiedHelmholtzFMM(int p, int ncrit, real_t wavek);
  void* setup_laplace(Bodies<real_t>* sources, Bodies<real_t>* targets, exafmm_t::LaplaceFmm* fmm);
  void* setup_helmholtz(Bodies<complex_t>* src, Bodies<complex_t>* trg, exafmm_t::HelmholtzFmm* pfmm);
  void* setup_modifiedhelmholtz(Bodies<real_t>* sources, Bodies<real_t>* targets, exafmm_t::ModifiedHelmholtzFmm* fmm);
  real_t* evaluate_laplace(void* fmmstruct);
  complex_t* evaluate_helmholtz(void* fmmstruct);
  real_t* evaluate_modifiedhelmholtz(void* fmmstruct);
  void update_charges_real(void* fmm, real_t* charges);
  void update_charges_cplx(void* fmm, complex_t* charges);
  void clear_values(void* fmm);
  real_t* verify_laplace(void* fmm);
  real_t* verify_helmholtz(void* fmm);
  real_t* verify_modifiedhelmholtz(void* fmm);
  void freestorage_real (exafmm_t::FmmBase<real_t>* fmm, void* fmmstruct, Bodies<real_t>* src, Bodies<real_t>* trg);
  void freestorage_cplx(exafmm_t::FmmBase<complex_t>* fmm, void* fmmstruct, Bodies<complex_t>* src, Bodies<complex_t>* trg);
}

Bodies<real_t>* init_sources_F(real_t* coords , real_t* charges, int nsrcs) {
  Bodies<real_t>* sources = new Bodies<real_t>(nsrcs);
  #pragma omp parallel for
  for(ssize_t i=0; i<nsrcs; ++i) {
    (*sources)[i].X[0] = coords[i+nsrcs*0];
    (*sources)[i].X[1] = coords[i+nsrcs*1];
    (*sources)[i].X[2] = coords[i+nsrcs*2];
    (*sources)[i].q = charges[i];
    (*sources)[i].ibody = i;
  }

  return sources;
}

Bodies<complex_t>* init_sources_C(real_t* coords , complex_t* charges, int nsrcs) {
  Bodies<complex_t>* sources = new Bodies<complex_t>(nsrcs);
  #pragma omp parallel for
  for(ssize_t i=0; i<nsrcs; ++i) {
    (*sources)[i].X[0] = coords[i+nsrcs*0];
    (*sources)[i].X[1] = coords[i+nsrcs*1];
    (*sources)[i].X[2] = coords[i+nsrcs*2];
    (*sources)[i].q = charges[i];
    (*sources)[i].ibody = i;
  }

  return sources;
}

Bodies<real_t>* init_targets_F(real_t* coords, int ntarg) {
  Bodies<real_t>* sources = new Bodies<real_t>(ntarg);
  #pragma omp parallel for
  for(ssize_t i=0; i<ntarg; ++i) {
    (*sources)[i].X[0] = coords[i+ntarg*0];
    (*sources)[i].X[1] = coords[i+ntarg*1];
    (*sources)[i].X[2] = coords[i+ntarg*2];
    (*sources)[i].ibody = i;
  }

  return sources;
}

Bodies<complex_t>* init_targets_C(real_t* coords, int ntarg) {
  Bodies<complex_t>* sources = new Bodies<complex_t>(ntarg);
  #pragma omp parallel for
  for(ssize_t i=0; i<ntarg; ++i) {
    (*sources)[i].X[0] = coords[i+ntarg*0];
    (*sources)[i].X[1] = coords[i+ntarg*1];
    (*sources)[i].X[2] = coords[i+ntarg*2];
    (*sources)[i].ibody = i;
  }

  return sources;
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

exafmm_t::LaplaceFmm* LaplaceFMM(int p, int ncrit){
  exafmm_t::LaplaceFmm* fmm = new exafmm_t::LaplaceFmm(p, ncrit);
  return fmm;
}

exafmm_t::HelmholtzFmm* HelmholtzFMM(int p, int ncrit, complex_t wavek){
  exafmm_t::HelmholtzFmm* fmm = new exafmm_t::HelmholtzFmm(p, ncrit, wavek);
  return fmm;
}

exafmm_t::ModifiedHelmholtzFmm* ModifiedHelmholtzFMM(int p, int ncrit, real_t wavek){
  exafmm_t::ModifiedHelmholtzFmm* fmm = new exafmm_t::ModifiedHelmholtzFmm(p, ncrit, wavek);
  return fmm;
}

struct LaplaceStruct {
  exafmm_t::LaplaceFmm* fmm;
  Tree<real_t>* tree;
};

struct HelmholtzStruct {
  exafmm_t::HelmholtzFmm* fmm;
  Tree<complex_t>* tree;
};

struct ModifiedHelmholtzStruct {
  exafmm_t::ModifiedHelmholtzFmm* fmm;
  Tree<real_t>* tree;
};

void* setup_laplace(Bodies<real_t>* sources, Bodies<real_t>* targets, exafmm_t::LaplaceFmm* fmm)
{
  Tree<real_t>* tree = new Tree<real_t>(build_tree<real_t>((*sources), (*targets), (*fmm)));
  exafmm_t::init_rel_coord();
  build_list<real_t>((*tree), (*fmm));
  fmm->M2L_setup(tree->nonleafs);
  fmm->precompute();
  LaplaceStruct* laplacefmm = new LaplaceStruct();
  laplacefmm->fmm = fmm;
  laplacefmm->tree = tree;
  return laplacefmm;
}

void* setup_helmholtz(Bodies<complex_t>* sources, Bodies<complex_t>* targets, exafmm_t::HelmholtzFmm* fmm)
{
  Tree<complex_t>* tree = new Tree<complex_t>(build_tree<complex_t>((*sources), (*targets), (*fmm)));
  exafmm_t::init_rel_coord();
  build_list<complex_t>((*tree), (*fmm));
  fmm->M2L_setup(tree->nonleafs);
  fmm->precompute();
  HelmholtzStruct* helmholtzfmm = new HelmholtzStruct();
  helmholtzfmm->fmm = fmm;
  helmholtzfmm->tree = tree;
  return helmholtzfmm;
}

void* setup_modifiedhelmholtz(Bodies<real_t>* sources, Bodies<real_t>* targets, exafmm_t::ModifiedHelmholtzFmm* fmm)
{
  Tree<real_t>* tree = new Tree<real_t>(build_tree<real_t>((*sources), (*targets), (*fmm)));
  exafmm_t::init_rel_coord();
  build_list<real_t>((*tree), (*fmm));
  fmm->M2L_setup(tree->nonleafs);
  fmm->precompute();
  ModifiedHelmholtzStruct* modifiedhelmholtzfmm = new ModifiedHelmholtzStruct();
  modifiedhelmholtzfmm->fmm = fmm;
  modifiedhelmholtzfmm->tree = tree;
  return modifiedhelmholtzfmm;
}

real_t* evaluate_laplace(void* laplace) {

  LaplaceStruct* laplacefmm = reinterpret_cast<LaplaceStruct*> (laplace);
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

  HelmholtzStruct* helmholtzfmm = reinterpret_cast<HelmholtzStruct*> (helmholtz);
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

real_t* evaluate_modifiedhelmholtz(void* helmholtz) {

  ModifiedHelmholtzStruct* modifiedhelmholtzfmm = reinterpret_cast<ModifiedHelmholtzStruct*> (helmholtz);
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
  LaplaceStruct* laplacefmm = reinterpret_cast<LaplaceStruct*> (fmm);
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
  HelmholtzStruct* helmholtzfmm = reinterpret_cast<HelmholtzStruct*> (fmm);
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
  LaplaceStruct* laplacefmm = reinterpret_cast<LaplaceStruct*> (fmm);
#pragma omp parallel for
  for (size_t i=0; i<laplacefmm->tree->nodes.size(); ++i) {
    auto& node = laplacefmm->tree->nodes[i];
    std::fill(node.up_equiv.begin(), node.up_equiv.end(), 0.);
    std::fill(node.dn_equiv.begin(), node.dn_equiv.end(), 0.);
    if (node.is_leaf)
      std::fill(node.trg_value.begin(), node.trg_value.end(), 0.);
  }
}

real_t* verify_laplace(void* fmm) {
  LaplaceStruct* laplacefmm = reinterpret_cast<LaplaceStruct*> (fmm);
  real_t* perr = new real_t[2]; 
  auto err = laplacefmm->fmm->verify(laplacefmm->tree->leafs);
  perr[0] = err[0];
  perr[1] = err[1];

  return perr;
}

real_t* verify_helmholtz(void* fmm) {
  HelmholtzStruct* helmholtzfmm = reinterpret_cast<HelmholtzStruct*> (fmm);
  real_t* perr = new real_t[2]; 
  auto err = helmholtzfmm->fmm->verify(helmholtzfmm->tree->leafs);
  perr[0] = err[0];
  perr[1] = err[1];

  return perr;
}

real_t* verify_modifiedhelmholtz(void* fmm) {
  ModifiedHelmholtzStruct* mhelmholtzfmm = reinterpret_cast<ModifiedHelmholtzStruct*> (fmm);
  real_t* perr = new real_t[2]; 
  auto err = mhelmholtzfmm->fmm->verify(mhelmholtzfmm->tree->leafs);
  perr[0] = err[0];
  perr[1] = err[1];

  return perr;
}

void freestorage_real (exafmm_t::FmmBase<real_t>* fmm, void* fmmstruct, Bodies<real_t>* src, Bodies<real_t>* trg) {
  LaplaceStruct* fmms = reinterpret_cast<LaplaceStruct*> (fmmstruct);
  free(fmms->fmm);
  free(fmms->tree);
  free(src);
  free(trg);
}

void freestorage_cplx(exafmm_t::FmmBase<complex_t>* fmm, void* fmmstruct, Bodies<complex_t>* src, Bodies<complex_t>* trg) {
  LaplaceStruct* fmms = reinterpret_cast<LaplaceStruct*> (fmmstruct);
  free(fmms->fmm);
  free(fmms->tree);
  free(src);
  free(trg);
}