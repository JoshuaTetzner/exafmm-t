#ifndef _PVFMM_FMM_TREE_HPP_
#define _PVFMM_FMM_TREE_HPP_
#include "intrinsics.h"
#include "pvfmm.h"
#include <queue>
#include "geometry.h"
#include "precomp_mat.hpp"
#include "build_tree.h"

namespace pvfmm {
  // Construct list of leafs, nonleafs and initialize members
  void CollectNodeData() {
    leafs.clear();
    nonleafs.clear();
    allnodes.clear();
    std::queue<FMM_Node*> nodesQueue;
    nodesQueue.push(root_node);
    while (!nodesQueue.empty()) {
      FMM_Node* curr = nodesQueue.front();
      nodesQueue.pop();
      if (curr != root_node)  allnodes.push_back(curr);
      if (!curr->IsLeaf()) nonleafs.push_back(curr);
      else leafs.push_back(curr);
      for (int i=0; i<8; i++) {
        FMM_Node* child = curr->Child(i);
        if (child!=NULL) nodesQueue.push(child);
      }
    }
    allnodes.push_back(root_node);   // level 0 root is the last one

    for (long i=0; i<leafs.size(); i++) {
      FMM_Node* leaf = leafs[i];
      leaf->pt_trg.resize(leaf->numBodies * TRG_DIM);
    }

    for(long i=0; i<allnodes.size(); i++) {
      FMM_Node* node = allnodes[i];
      node->idx = i;
    }
    LEVEL = leafs.back()->depth;
  }


  void ClearFMMData() {
    for(size_t i=0; i<allnodes.size(); i++) {
      std::vector<real_t>& upward_equiv = allnodes[i]->upward_equiv;
      std::vector<real_t>& dnward_equiv = allnodes[i]->dnward_equiv;
      std::vector<real_t>& pt_trg = allnodes[i]->pt_trg;
      std::fill(upward_equiv.begin(), upward_equiv.end(), 0);
      std::fill(dnward_equiv.begin(), dnward_equiv.end(), 0);
      std::fill(pt_trg.begin(), pt_trg.end(), 0);
    }
  }

  void M2LSetup(M2LData& M2Ldata) {
    size_t mat_cnt = rel_coord[M2L_Type].size();
    // construct nodes_out & nodes_in
    std::vector<FMM_Node*>& nodes_out = nonleafs;
    std::set<FMM_Node*> nodes_in_;
    for(size_t i=0; i<nodes_out.size(); i++) {
      std::vector<FMM_Node*>& M2Llist = nodes_out[i]->interac_list[M2L_Type];
      for(size_t k=0; k<mat_cnt; k++) {
        if(M2Llist[k]!=NULL)
          nodes_in_.insert(M2Llist[k]);
      }
    }
    std::vector<FMM_Node*> nodes_in;
    for(std::set<FMM_Node*>::iterator node=nodes_in_.begin(); node!=nodes_in_.end(); node++) {
      nodes_in.push_back(*node);
    }
    // prepare fft displ & fft scal
    std::vector<size_t> fft_vec(nodes_in.size());
    std::vector<size_t> ifft_vec(nodes_out.size());
    std::vector<real_t> fft_scl(nodes_in.size());
    std::vector<real_t> ifft_scl(nodes_out.size());
    for(size_t i=0; i<nodes_in.size(); i++) {
      fft_vec[i] = nodes_in[i]->child[0]->idx * NSURF;
      fft_scl[i] = 1;
    }
    for(size_t i=0; i<nodes_out.size(); i++) {
      int depth = nodes_out[i]->depth+1;
      ifft_vec[i] = nodes_out[i]->child[0]->idx * NSURF;
      ifft_scl[i] = powf(2.0, depth);
    }
    // calculate interac_vec & interac_dsp
    std::vector<size_t> interac_vec;
    std::vector<size_t> interac_dsp;
    for(size_t i=0; i<nodes_in.size(); i++) {
     nodes_in[i]->node_id=i;
    }
    size_t n_blk1 = nodes_out.size() * sizeof(real_t) / CACHE_SIZE;
    if(n_blk1==0) n_blk1 = 1;
    size_t interac_dsp_ = 0;
    for(size_t blk1=0; blk1<n_blk1; blk1++) {
      size_t blk1_start=(nodes_out.size()* blk1   )/n_blk1;
      size_t blk1_end  =(nodes_out.size()*(blk1+1))/n_blk1;
      for(size_t k=0; k<mat_cnt; k++) {
        for(size_t i=blk1_start; i<blk1_end; i++) {
          std::vector<FMM_Node*>& M2Llist = nodes_out[i]->interac_list[M2L_Type];
          if(M2Llist[k]!=NULL) {
            interac_vec.push_back(M2Llist[k]->node_id * FFTSIZE);   // node_in dspl
            interac_vec.push_back(        i           * FFTSIZE);   // node_out dspl
            interac_dsp_++;
          }
        }
        interac_dsp.push_back(interac_dsp_);
      }
    }
    M2Ldata.fft_vec     = fft_vec;
    M2Ldata.ifft_vec    = ifft_vec;
    M2Ldata.fft_scl     = fft_scl;
    M2Ldata.ifft_scl    = ifft_scl;
    M2Ldata.interac_vec = interac_vec;
    M2Ldata.interac_dsp = interac_dsp;
  }

  void SetupFMM(FMM_Nodes& cells) {
    Profile::Tic("SetupFMM", true);
    SetColleagues(cells);
    Profile::Tic("BuildLists", false, 3);
    BuildInteracLists(cells);
    Profile::Toc();
    Profile::Tic("CollectNodeData", false, 3);
    CollectNodeData();
    Profile::Toc();
    Profile::Tic("M2LListSetup", false, 3);
    M2LSetup(M2Ldata);
    Profile::Toc();
    ClearFMMData();
    Profile::Toc();
  }

  void UpwardPass() {
    Profile::Tic("P2M", false, 5);
    P2M();
    Profile::Toc();
    Profile::Tic("M2M", false, 5);
    #pragma omp parallel
    #pragma omp single nowait
    M2M(root_node);
    Profile::Toc();
  }

  void DownwardPass() {
    Profile::Tic("P2L", false, 5);
    P2L();
    Profile::Toc();
    Profile::Tic("M2P", false, 5);
    M2P();
    Profile::Toc();
    Profile::Tic("P2P", false, 5);
    P2P();
    Profile::Toc();
    Profile::Tic("M2L", false, 5);
    M2L(M2Ldata);
    Profile::Toc();
    Profile::Tic("L2L", false, 5);
    #pragma omp parallel
    #pragma omp single nowait
    L2L(root_node);
    Profile::Toc();
    Profile::Tic("L2P", false, 5);
    L2P();
    Profile::Toc();
  }

  void RunFMM() {
    Profile::Tic("RunFMM", true);
    Profile::Tic("UpwardPass", false, 2);
    UpwardPass();
    Profile::Toc();
    Profile::Tic("DownwardPass", true, 2);
    DownwardPass();
    Profile::Toc();
    Profile::Toc();
  }

  void CheckFMMOutput(std::vector<FMM_Node*>& leafs) {
    int numTargets = 10;
    int stride = leafs.size() / numTargets;
    FMM_Nodes targets;
    for(size_t i=0; i<numTargets; i++) {
      targets.push_back(*(leafs[i*stride]));
    }
    FMM_Nodes targets2 = targets;    // used for direct summation
#pragma omp parallel for
    for(size_t i=0; i<targets2.size(); i++) {
      FMM_Node *target = &targets2[i];
      std::fill(target->pt_trg.begin(), target->pt_trg.end(), 0.);
      for(size_t j=0; j<leafs.size(); j++) {
        gradientP2P(leafs[j]->pt_coord, leafs[j]->pt_src, target->pt_coord, target->pt_trg);
      }
    }

    real_t p_diff = 0, p_norm = 0, g_diff = 0, g_norm = 0;
    for(size_t i=0; i<targets.size(); i++) {
      p_norm += targets2[i].pt_trg[0] * targets2[i].pt_trg[0];
      p_diff += (targets2[i].pt_trg[0] - targets[i].pt_trg[0]) * (targets2[i].pt_trg[0] - targets[i].pt_trg[0]);
      for(int d=1; d<4; d++) {
        g_diff += (targets2[i].pt_trg[d] - targets[i].pt_trg[d]) * (targets2[i].pt_trg[d] - targets[i].pt_trg[d]);
        g_norm += targets2[i].pt_trg[d] * targets2[i].pt_trg[d];
      }
    }
    std::cout << std::setw(20) << std::left << "Potn Error" << " : " << std::scientific << sqrt(p_diff/p_norm) << std::endl;
    std::cout << std::setw(20) << std::left << "Grad Error" << " : " << std::scientific << sqrt(g_diff/g_norm) << std::endl;
  }
}//end namespace
#endif //_PVFMM_FMM_TREE_HPP_
