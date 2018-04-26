#ifndef _PVFMM_FMM_NODE_HPP_
#define _PVFMM_FMM_NODE_HPP_
#include "pvfmm.h"
namespace pvfmm {

class FMM_Node {
 public:
  size_t idx;     // index of node in level-order, to help remove data_buff
  int depth;
  int max_depth;
  int octant;
  FMM_Node* parent;
  FMM_Node** child;
  size_t max_pts;
  size_t node_id;
  real_t coord[3];
  FMM_Node * colleague[27];
  std::vector<real_t> pt_coord;
  std::vector<real_t> pt_value;
  std::vector<real_t> trg_value;
  std::vector<real_t> upward_equiv; // M
  std::vector<real_t> dnward_equiv; // L
  size_t pt_cnt[2];
  std::vector<FMM_Node*> interac_list[Type_Count];

  FMM_Node() : depth(0), max_depth(MAX_DEPTH), parent(NULL), child(NULL) {
  }

  ~FMM_Node() {
    if(!child) return;
    int n=(1UL<<3);
    for(int i=0; i<n; i++) {
      if(child[i]!=NULL)
        delete child[i];
    }
    delete[] child;
    child=NULL;
  }

  void Initialize(FMM_Node* parent_, int octant_, InitData* data_) {
    parent=parent_;
    depth=(parent==NULL?0:parent->depth+1);
    if(data_!=NULL) {
      max_depth=data_->max_depth;
      if(max_depth>MAX_DEPTH) max_depth=MAX_DEPTH;
    } else if(parent!=NULL)
      max_depth=parent->max_depth;
    assert(octant_>=0 && octant_<8);
    octant=octant_;
    real_t coord_offset=((real_t)1.0)/((real_t)(((uint64_t)1)<<depth));
    if(!parent_) {
      for(int j=0; j<3; j++) coord[j]=0;
    } else if(parent_) {
      int flag=1;
      for(int j=0; j<3; j++) {
        coord[j]=parent_->coord[j]+
                 ((octant & flag)?coord_offset:0.0f);
        flag=flag<<1;
      }
    }
    for(int i=0; i<27; i++) colleague[i]=NULL;
    InitData* data=data_;
    if(data_) {
      max_pts=data->max_pts;
      pt_coord=data->coord;
      pt_value=data->value;
    } else if(parent)
      max_pts =parent->max_pts;
  }

  void NodeDataVec(std::vector<std::vector<real_t>*>& coord,
                   std::vector<std::vector<real_t>*>& value) {
    coord  .push_back(&pt_coord  );
    value  .push_back(&pt_value  );
  }

  void Truncate() {
    if(!child) return;
    int n=8;
    for(int i=0; i<n; i++) {
      if(child[i]!=NULL)
        delete child[i];
    }
    delete[] child;
    child=NULL;
  }

  FMM_Node* NewNode() {
    FMM_Node* n=new FMM_Node();
    n->max_depth=max_depth;
    n->max_pts=max_pts;
    return n;
  }

  void Subdivide() {
    if(!IsLeaf()) return;
    if(child) return;
    int n = 8;
    child=new FMM_Node* [n];
    for(int i=0; i<n; i++) {
      child[i]=NewNode();
      child[i]->parent=this;
      child[i]->Initialize(this, i, NULL);
    }
    int nchld = 8;
    std::vector<std::vector<real_t>*> pt_c;
    std::vector<std::vector<real_t>*> pt_v;
    NodeDataVec(pt_c, pt_v);
    std::vector<std::vector<std::vector<real_t>*> > chld_pt_c(nchld);
    std::vector<std::vector<std::vector<real_t>*> > chld_pt_v(nchld);
    for(size_t i=0; i<nchld; i++)
      Child(i)->NodeDataVec(chld_pt_c[i], chld_pt_v[i]);
    real_t* c=Coord();
    real_t s=powf(0.5, depth+1);
    for(size_t j=0; j<pt_c.size(); j++) {
      if(!pt_c[j] || !pt_c[j]->size()) continue;
      std::vector<real_t>& coord=*pt_c[j];
      size_t npts=coord.size()/3;
      std::vector<size_t> cdata(nchld+1);
      for(size_t i=0; i<nchld+1; i++) {
        long long pt1=-1, pt2=npts;
        while(pt2-pt1>1) {
          long long pt3=(pt1+pt2)/2;
          assert(pt3<npts);
          if(pt3<0) pt3=0;
          int ch_id=(coord[pt3*3+0]>=c[0]+s)*1+
                    (coord[pt3*3+1]>=c[1]+s)*2+
                    (coord[pt3*3+2]>=c[2]+s)*4;
          if(ch_id< i) pt1=pt3;
          if(ch_id>=i) pt2=pt3;
        }
        cdata[i]=pt2;
      }
      if(pt_c[j]) {
        std::vector<real_t>& vec=*pt_c[j];
        size_t dof=vec.size()/npts;
        assert(dof>0);
        for(size_t i=0; i<nchld; i++) {
          std::vector<real_t>& chld_vec=*chld_pt_c[i][j];
          chld_vec.resize((cdata[i+1]-cdata[i])*dof);
          for (int k=cdata[i]*dof; k<cdata[i+1]*dof; k++)
            chld_vec[k-cdata[i]*dof] = vec[k];
        }
        vec.resize(0);
      }
      if(pt_v[j]) {
        std::vector<real_t>& vec=*pt_v[j];
        size_t dof=vec.size()/npts;
        for(size_t i=0; i<nchld; i++) {
          std::vector<real_t>& chld_vec=*chld_pt_v[i][j];
          chld_vec.resize((cdata[i+1]-cdata[i])*dof);
          for (int k=cdata[i]*dof; k<cdata[i+1]*dof; k++)
            chld_vec[k-cdata[i]*dof] = vec[k];
        }
        vec.resize(0);
      }
    }
  }

  bool IsLeaf() {
    return child == NULL;
  }

  FMM_Node* Child(int id) {
    assert(id<8);
    if(child==NULL) return NULL;
    return child[id];
  }

  FMM_Node* Parent() {
    return parent;
  }

  inline MortonId GetMortonId() {
    assert(coord);
    real_t s=0.25/(1UL<<MAX_DEPTH);
    return MortonId(coord[0]+s, coord[1]+s, coord[2]+s, depth);
  }

  inline void SetCoord(MortonId& mid) {
    assert(coord);
    mid.GetCoord(coord);
    depth=mid.GetDepth();
  }

  FMM_Node * Colleague(int index) {
    return colleague[index];
  }

  void SetColleague(FMM_Node * node_, int index) {
    colleague[index]=node_;
  }

  real_t* Coord() {
    assert(coord!=NULL);
    return coord;
  }

};

}//end namespace

#endif //_PVFMM_FMM_NODE_HPP_
