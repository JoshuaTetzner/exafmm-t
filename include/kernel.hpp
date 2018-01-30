#ifndef _PVFMM_FMM_KERNEL_HPP_
#define _PVFMM_FMM_KERNEL_HPP_
#include "intrinsics.h"

namespace pvfmm {
struct Kernel{
  public:
  typedef void (*Ker_t)(real_t* r_src, int src_cnt, real_t* v_src,
                        real_t* r_trg, int trg_cnt, real_t* k_out);

  int ker_dim[2];
  std::string ker_name;
  Ker_t ker_poten;

  mutable bool init;
  mutable bool scale_invar;
  mutable std::vector<real_t> src_scal;
  mutable std::vector<real_t> trg_scal;
  mutable std::vector<Permutation<real_t> > perm_vec;

  mutable const Kernel* k_s2m;
  mutable const Kernel* k_s2l;
  mutable const Kernel* k_s2t;
  mutable const Kernel* k_m2m;
  mutable const Kernel* k_m2l;
  mutable const Kernel* k_m2t;
  mutable const Kernel* k_l2l;
  mutable const Kernel* k_l2t;

  Kernel(Ker_t poten, const char* name, std::pair<int,int> k_dim) {
    ker_dim[0]=k_dim.first;
    ker_dim[1]=k_dim.second;
    ker_name=std::string(name);
    ker_poten=poten;
 
    init=false;
    scale_invar=true;
    src_scal.resize(ker_dim[0], 0.); 
    trg_scal.resize(ker_dim[1], 0.); 
    perm_vec.resize(Perm_Count);
    std::fill(perm_vec.begin(), perm_vec.begin()+C_Perm, Permutation<real_t>(ker_dim[0]));
    std::fill(perm_vec.begin()+C_Perm, perm_vec.end(), Permutation<real_t>(ker_dim[1]));

    k_s2m=NULL;
    k_s2l=NULL;
    k_s2t=NULL;
    k_m2m=NULL;
    k_m2l=NULL;
    k_m2t=NULL;
    k_l2l=NULL;
    k_l2t=NULL;
  }

  void Initialize(bool verbose=false) const{
    if(init) return;
    init = true;
    real_t eps = 1.0;
    while(eps+(real_t)1.0>1.0) eps *= 0.5;
    real_t scal = 1.0;

    Matrix<real_t> M_scal(ker_dim[0],ker_dim[1]);
    size_t N = 1024;
    real_t eps_ = N * eps;
    real_t src_coord[3]={0,0,0};
    std::vector<real_t> trg_coord1(N*3);
    Matrix<real_t> M1(N,ker_dim[0]*ker_dim[1]);
    // generate trg_coord1 randomly r>0.25, a unit charge source at origin
    // save potential/field in M1, for potential kernel: M1[i][0] = 1/(4*pi*|r|)
    //                             for gradient          M1[i] = -r/(4*pi*|r|^3)
    while(true){
      real_t abs_sum=0;
      for(size_t i=0;i<N/2;i++){
        real_t x,y,z,r;
        do {
          x = (drand48()-0.5);
          y = (drand48()-0.5);
          z = (drand48()-0.5);
          r = sqrtf(x*x+y*y+z*z);
        } while (r<0.25);
        trg_coord1[i*3+0]=x*scal;
        trg_coord1[i*3+1]=y*scal;
        trg_coord1[i*3+2]=z*scal;
      }
      for(size_t i=N/2;i<N;i++){
        real_t x,y,z,r;
        do {
          x = (drand48()-0.5);
          y = (drand48()-0.5);
          z = (drand48()-0.5);
          r = sqrtf(x*x+y*y+z*z);
        } while(r<0.25);
        trg_coord1[i*3+0]=x*1.0/scal;
        trg_coord1[i*3+1]=y*1.0/scal;
        trg_coord1[i*3+2]=z*1.0/scal;
      }
      for(size_t i=0;i<N;i++){
        BuildMatrix(&src_coord [          0], 1,
                    &trg_coord1[i*3], 1, &(M1[i][0]));
        for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
          abs_sum+=fabs(M1[i][j]);
        }
      }
      // always break at the first iteration, scal = 1
      if(abs_sum>sqrtf(eps) || scal<eps) { 
#if DEBUG
        std::cout << ker_name << " scal loop break at scal = " << scal << std::endl; 
#endif
        break;
      }
      scal=scal*0.5;
    }

    // half the trg_coord1 to get trg_coord2
    // interact with unit charge source at origin to generate M2
    std::vector<real_t> trg_coord2(N*3);
    Matrix<real_t> M2(N,ker_dim[0]*ker_dim[1]);
    for(size_t i=0;i<N*3;i++){
      trg_coord2[i]=trg_coord1[i]*0.5;
    }
    for(size_t i=0;i<N;i++){
      BuildMatrix(&src_coord [          0], 1,
                  &trg_coord2[i*3], 1, &(M2[i][0]));
    }

    // Calculate M_scal (scal of current kernel) 
    for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
      real_t dot11=0, dot12=0, dot22=0;
      for(size_t j=0;j<N;j++){
        dot11+=M1[j][i]*M1[j][i];  // sum of squared potential/force
        dot12+=M1[j][i]*M2[j][i];
        dot22+=M2[j][i]*M2[j][i];
      }
      real_t max_val=std::max<real_t>(dot11,dot22);
      if(dot11>max_val*eps &&      // always goes to this branch since eps is tiny
         dot22>max_val*eps ){
        real_t s=dot12/dot11;      // due to the form of Laplace kernel: s=2 for potential s=4 for force
        M_scal[0][i]=log(s)/log(2.0);     // scal=1 for potential and 2 for force
        real_t err=sqrtf(0.5*(dot22/dot11)/(s*s)-0.5); // always 0 for Laplace
        if(err>eps_){
          scale_invar=false;
          M_scal[0][i]=0.0;
        }
      }else if(dot11>max_val*eps ||
               dot22>max_val*eps ){
        scale_invar=false;
        M_scal[0][i]=0.0;
      }else{
        M_scal[0][i]=-1;
      }
    }
#if DEBUG
    for(int a=0; a<ker_dim[1]; a++) {std::cout << ker_name << " " << a << " M_scal: " << M_scal[0][a] << std::endl;}
    std::cout << ker_name << " "  << " scale_invar: " << scale_invar << std::endl;
#endif
    // Calculate src_scal & trg_scal: Mx=b solve for x
    if(scale_invar){    // always true for Laplace
      Matrix<real_t> b(ker_dim[0]*ker_dim[1]+1,1); b.SetZero();
      memcpy(&b[0][0],&M_scal[0][0],ker_dim[0]*ker_dim[1]*sizeof(real_t)); // the last elem in b is still zero
      Matrix<real_t> M(ker_dim[0]*ker_dim[1]+1,ker_dim[0]+ker_dim[1]); M.SetZero();
      M[ker_dim[0]*ker_dim[1]][0]=1;
      for(size_t i0=0;i0<ker_dim[0];i0++)
        for(size_t i1=0;i1<ker_dim[1];i1++){
          size_t j=i0*ker_dim[1]+i1;
          if(fabs(b[j][0])>=0){
            M[j][ 0+        i0]=1;
            M[j][i1+ker_dim[0]]=1;
          }
        }

      Matrix<real_t> x=M.pinv()*b;   // x = [0,1] for potential and [0,2,2,2] for gradient
      for(size_t i=0;i<ker_dim[0];i++){
        src_scal[i]=x[i][0];
      }
      for(size_t i=0;i<ker_dim[1];i++){
        trg_scal[i]=x[ker_dim[0]+i][0];
      }
      for(size_t i0=0;i0<ker_dim[0];i0++)
        for(size_t i1=0;i1<ker_dim[1];i1++){
          if(M_scal[i0][i1]>=0){
            if(fabs(src_scal[i0]+trg_scal[i1]-M_scal[i0][i1])>eps_){
              scale_invar=false;
            }
          }
        }
    }
    if(!scale_invar){
      std::fill(src_scal.begin(), src_scal.end(), 0.);
      std::fill(trg_scal.begin(), trg_scal.end(), 0.);
    }


      for(size_t i=0;i<2*C_Perm;i++){
	if(perm_vec[i].Dim()==0){
	  perm_vec.resize(0);
	  std::cout<<"no-symmetry for: "<<ker_name<<'\n';
	  break;
	}
      }

    if(!k_s2m) k_s2m=this;
    if(!k_s2l) k_s2l=this;
    if(!k_s2t) k_s2t=this;
    if(!k_m2m) k_m2m=this;
    if(!k_m2l) k_m2l=this;
    if(!k_m2t) k_m2t=this;
    if(!k_l2l) k_l2l=this;
    if(!k_l2t) k_l2t=this;
    assert(k_s2t->ker_dim[0]==ker_dim[0]);
    assert(k_s2m->ker_dim[0]==k_s2l->ker_dim[0]);
    assert(k_s2m->ker_dim[0]==k_s2t->ker_dim[0]);
    assert(k_m2m->ker_dim[0]==k_m2l->ker_dim[0]);
    assert(k_m2m->ker_dim[0]==k_m2t->ker_dim[0]);
    assert(k_l2l->ker_dim[0]==k_l2t->ker_dim[0]);
    assert(k_s2t->ker_dim[1]==ker_dim[1]);
    assert(k_s2m->ker_dim[1]==k_m2m->ker_dim[1]);
    assert(k_s2l->ker_dim[1]==k_l2l->ker_dim[1]);
    assert(k_m2l->ker_dim[1]==k_l2l->ker_dim[1]);
    assert(k_s2t->ker_dim[1]==k_m2t->ker_dim[1]);
    assert(k_s2t->ker_dim[1]==k_l2t->ker_dim[1]);
    k_s2m->Initialize(verbose);
    k_s2l->Initialize(verbose);
    k_s2t->Initialize(verbose);
    k_m2m->Initialize(verbose);
    k_m2l->Initialize(verbose);
    k_m2t->Initialize(verbose);
    k_l2l->Initialize(verbose);
    k_l2t->Initialize(verbose);
  }

  //! Laplace P2P save pairwise contributions to k_out (not aggregate over each target)
  // For Laplace: ker_dim[0] = 1, j = 0; Force a unit charge (q=1)
  // r_src layout: [x1, y1, z1, x2, y2, z2, ...] 
  // k_out layout (potential): [p11, p12, p13, ..., p21, p22, ...]  (1st digit: src_idx; 2nd: trg_idx)
  // k_out layout (gradient) : [Fx11, Fy11, Fz11, Fx12, Fy12, Fz13, ... Fx1n, Fy1n, Fz1n, ...
  //                            Fx21, Fy21, Fz21, Fx22, Fy22, Fz22, ... Fx2n, Fy2n, Fz2n, ...
  //                            ...]
  void BuildMatrix(real_t* r_src, int src_cnt, real_t* r_trg, int trg_cnt, real_t* k_out) const{
    memset(k_out, 0, src_cnt*ker_dim[0]*trg_cnt*ker_dim[1]*sizeof(real_t));
    for(int i=0;i<src_cnt;i++)
      for(int j=0;j<ker_dim[0];j++){
	std::vector<real_t> v_src(ker_dim[0],0);
	v_src[j]=1.0;
        // do P2P: i-th source
	ker_poten(&r_src[i*3], 1, &v_src[0], r_trg, trg_cnt,
		  &k_out[(i*ker_dim[0]+j)*trg_cnt*ker_dim[1]]);
      }
  }
};

template<void (*A)(real_t*, int, real_t*, real_t*, int, real_t*)>
Kernel BuildKernel(const char* name, std::pair<int,int> k_dim, 
                   const Kernel* k_s2m=NULL, const Kernel* k_s2l=NULL, const Kernel* k_s2t=NULL, const Kernel* k_m2m=NULL, 
                   const Kernel* k_m2l=NULL, const Kernel* k_m2t=NULL, const Kernel* k_l2l=NULL, const Kernel* k_l2t=NULL) {
  Kernel K(A, name, k_dim);
  K.k_s2m=k_s2m;
  K.k_s2l=k_s2l;
  K.k_s2t=k_s2t;
  K.k_m2m=k_m2m;
  K.k_m2l=k_m2l;
  K.k_m2t=k_m2t;
  K.k_l2l=k_l2l;
  K.k_l2t=k_l2t;
  return K;
}

//! Laplace potential P2P 1/(4*pi*|r|) with matrix interface, potentials saved in trg_value matrix
// source & target coord matrix size: 3 by N
void potentialP2P(Matrix<real_t>& src_coord, Matrix<real_t>& src_value, Matrix<real_t>& trg_coord, Matrix<real_t>& trg_value){
#define SRC_BLK 1000
  size_t VecLen=sizeof(vec_t)/sizeof(real_t);
  real_t nwtn_scal=1;
  for(int i=0;i<2;i++){
    nwtn_scal = 2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const real_t zero = 0;
  const real_t OOFP = 1.0/(4*nwtn_scal*M_PI);
  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      vec_t tx=load_intrin(&trg_coord[0][t]);
      vec_t ty=load_intrin(&trg_coord[1][t]);
      vec_t tz=load_intrin(&trg_coord[2][t]);
      vec_t tv=zero_intrin(zero);
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        vec_t dx=sub_intrin(tx,set_intrin(src_coord[0][s]));
        vec_t dy=sub_intrin(ty,set_intrin(src_coord[1][s]));
        vec_t dz=sub_intrin(tz,set_intrin(src_coord[2][s]));
        vec_t sv=              set_intrin(src_value[0][s]) ;
        vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));
        vec_t rinv=rsqrt_intrin2(r2);
        tv=add_intrin(tv,mul_intrin(rinv,sv));
      }
      vec_t oofp=set_intrin(OOFP);
      tv=add_intrin(mul_intrin(tv,oofp),load_intrin(&trg_value[0][t]));
      store_intrin(&trg_value[0][t],tv);
    }
  }
  {
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*20);
  }
#undef SRC_BLK
}

//! Laplace gradient P2P -r/(4*pi*|r|^3) with matrix interface, gradients saved in trg_value matrix
// source & target coord matrix size: 3 by N
void gradientP2P(Matrix<real_t>& src_coord, Matrix<real_t>& src_value, Matrix<real_t>& trg_coord, Matrix<real_t>& trg_value){
#define SRC_BLK 500
  size_t VecLen=sizeof(vec_t)/sizeof(real_t);
  real_t nwtn_scal=1;
  for(int i=0;i<2;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const real_t zero = 0;
  const real_t OOFP = -1.0/(4*nwtn_scal*nwtn_scal*nwtn_scal*M_PI);
  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      vec_t tx=load_intrin(&trg_coord[0][t]);
      vec_t ty=load_intrin(&trg_coord[1][t]);
      vec_t tz=load_intrin(&trg_coord[2][t]);
      vec_t tv0=zero_intrin(zero);
      vec_t tv1=zero_intrin(zero);
      vec_t tv2=zero_intrin(zero);
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        vec_t dx=sub_intrin(tx,set_intrin(src_coord[0][s]));
        vec_t dy=sub_intrin(ty,set_intrin(src_coord[1][s]));
        vec_t dz=sub_intrin(tz,set_intrin(src_coord[2][s]));
        vec_t sv=              set_intrin(src_value[0][s]) ;
        vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));
        vec_t rinv=rsqrt_intrin2(r2);
        vec_t r3inv=mul_intrin(mul_intrin(rinv,rinv),rinv);
        sv=mul_intrin(sv,r3inv);
        tv0=add_intrin(tv0,mul_intrin(sv,dx));
        tv1=add_intrin(tv1,mul_intrin(sv,dy));
        tv2=add_intrin(tv2,mul_intrin(sv,dz));
      }
      vec_t oofp=set_intrin(OOFP);
      tv0=add_intrin(mul_intrin(tv0,oofp),load_intrin(&trg_value[0][t]));
      tv1=add_intrin(mul_intrin(tv1,oofp),load_intrin(&trg_value[1][t]));
      tv2=add_intrin(mul_intrin(tv2,oofp),load_intrin(&trg_value[2][t]));
      store_intrin(&trg_value[0][t],tv0);
      store_intrin(&trg_value[1][t],tv1);
      store_intrin(&trg_value[2][t],tv2);
    }
  }
  {
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*27);
  }
#undef SRC_BLK
}

//! Wrap around the above P2P functions with matrix interface to provide array interface
//! Evaluate potential / gradient based on the argument grad
// r_src & r_trg coordinate array: [x1, y1, z1, x2, y2, z2, ...]
void laplaceP2P(real_t* r_src, int src_cnt, real_t* v_src, real_t* r_trg, int trg_cnt, real_t* v_trg, bool grad=false){
int SRC_DIM = 1;
int TRG_DIM = (grad) ? 3 : 1;
#if FLOAT
  int VecLen=8;
#else
  int VecLen=4;
#endif
#define STACK_BUFF_SIZE 4096
  real_t stack_buff[STACK_BUFF_SIZE+MEM_ALIGN];
  real_t* buff=NULL;
  Matrix<real_t> src_coord;
  Matrix<real_t> src_value;
  Matrix<real_t> trg_coord;
  Matrix<real_t> trg_value;
  {
    size_t src_cnt_, trg_cnt_;
    src_cnt_=((src_cnt+VecLen-1)/VecLen)*VecLen;
    trg_cnt_=((trg_cnt+VecLen-1)/VecLen)*VecLen;
    size_t buff_size=src_cnt_*(3+SRC_DIM)+
                     trg_cnt_*(3+TRG_DIM);
    if(buff_size>STACK_BUFF_SIZE){
      int err = posix_memalign((void**)&buff, MEM_ALIGN, buff_size*sizeof(real_t));
    }
    real_t* buff_ptr=buff;
    if(!buff_ptr){
      uintptr_t ptr=(uintptr_t)stack_buff;
      static uintptr_t     ALIGN_MINUS_ONE=MEM_ALIGN-1;
      static uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
      ptr=((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
      buff_ptr=(real_t*)ptr;
    }
    src_coord.ReInit(3, src_cnt_,buff_ptr,false);  buff_ptr+=3*src_cnt_;
    src_value.ReInit(  SRC_DIM, src_cnt_,buff_ptr,false);  buff_ptr+=  SRC_DIM*src_cnt_;
    trg_coord.ReInit(3, trg_cnt_,buff_ptr,false);  buff_ptr+=3*trg_cnt_;
    trg_value.ReInit(  TRG_DIM, trg_cnt_,buff_ptr,false);
    {
      size_t i=0;
      for(   ;i<src_cnt ;i++){
        for(size_t j=0;j<3;j++){
          src_coord[j][i]=r_src[i*3+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(size_t j=0;j<3;j++){
          src_coord[j][i]=0;
        }
      }
    }
    {
      size_t i=0;
      for(   ;i<src_cnt ;i++){
        for(size_t j=0;j<SRC_DIM;j++){
          src_value[j][i]=v_src[i*SRC_DIM+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(size_t j=0;j<SRC_DIM;j++){
          src_value[j][i]=0;
        }
      }
    }
    {
      size_t i=0;
      for(   ;i<trg_cnt ;i++){
        for(size_t j=0;j<3;j++){
          trg_coord[j][i]=r_trg[i*3+j];
        }
      }
      for(   ;i<trg_cnt_;i++){
        for(size_t j=0;j<3;j++){
          trg_coord[j][i]=0;
        }
      }
    }
    {
      size_t i=0;
      for(   ;i<trg_cnt_;i++){
        for(size_t j=0;j<TRG_DIM;j++){
          trg_value[j][i]=0;
        }
      }
    }
  }
  if (grad) gradientP2P(src_coord,src_value,trg_coord,trg_value);
  else potentialP2P(src_coord,src_value,trg_coord,trg_value);
  {
    for(size_t i=0;i<trg_cnt ;i++){
      for(size_t j=0;j<TRG_DIM;j++){
        v_trg[i*TRG_DIM+j]+=trg_value[j][i];
      }
    }
  }
  if(buff){
    free(buff);
  }
}

//! Laplace potential P2P with array interface
void potentialP2P(real_t* r_src, int src_cnt, real_t* v_src, real_t* r_trg, int trg_cnt, real_t* v_trg){
  laplaceP2P(r_src, src_cnt, v_src,  r_trg, trg_cnt, v_trg, false);
}
//! Laplace gradient P2P with array interface
void gradientP2P(real_t* r_src, int src_cnt, real_t* v_src,  real_t* r_trg, int trg_cnt, real_t* v_trg){
  laplaceP2P(r_src, src_cnt, v_src, r_trg, trg_cnt, v_trg, true);
}

}//end namespace

#endif //_PVFMM_FMM_KERNEL_HPP_
