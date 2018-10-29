#ifndef precompute_helmholtz_h
#define precompute_helmholtz_h
#include "exafmm_t.h"
#include "geometry.h"
#include "helmholtz.h"

namespace exafmm_t {
  std::vector<ComplexVec> M2M_U, M2M_V;
  std::vector<ComplexVec> L2L_U, L2L_V;
  std::vector<std::vector<ComplexVec>> mat_M2M, mat_L2L;
  std::vector<RealVec> mat_M2L_Helper;
  std::vector<RealVec> mat_M2L;

  void PrecompCheck2Equiv() {
    real_t c[3] = {0, 0, 0};
    M2M_V.resize(MAXLEVEL+1);
    M2M_U.resize(MAXLEVEL+1);
    L2L_V.resize(MAXLEVEL+1);
    L2L_U.resize(MAXLEVEL+1);
    for(int level = 0; level <= MAXLEVEL; level++) {
      // caculate M2M_U and M2M_V
      RealVec uc_coord = surface(MULTIPOLE_ORDER,c,2.95,level);
      RealVec ue_coord = surface(MULTIPOLE_ORDER,c,1.05,level);
      ComplexVec M_e2c(NSURF*NSURF);
      kernelMatrix(&ue_coord[0], NSURF, &uc_coord[0], NSURF, &M_e2c[0]);
      RealVec S(NSURF*NSURF);
      ComplexVec U(NSURF*NSURF), VT(NSURF*NSURF);
      svd(NSURF, NSURF, &M_e2c[0], &S[0], &U[0], &VT[0]);
      // inverse S
      real_t max_S = 0;
      for(size_t i=0; i<NSURF; i++) {
        max_S = fabs(S[i*NSURF+i])>max_S ? fabs(S[i*NSURF+i]) : max_S;
      }
      for(size_t i=0; i<NSURF; i++) {
        S[i*NSURF+i] = S[i*NSURF+i]>EPS*max_S*4 ? 1.0/S[i*NSURF+i] : 0.0;
      }
      // save matrix
      ComplexVec V = transpose(VT, NSURF, NSURF);
      M2M_V[level].resize(NSURF*NSURF);
      M2M_U[level] = transpose(U, NSURF, NSURF);
      gemm(NSURF, NSURF, NSURF, &V[0], &S[0], &(M2M_V[level][0]));

      L2L_V[level].resize(NSURF*NSURF);
      L2L_U[level] = VT;
      gemm(NSURF, NSURF, NSURF, &U[0], &S[0], &(L2L_V[level][0]));
    }
  }

  void PrecompM2M() {
    real_t parent_coord[3] = {0, 0, 0};
    mat_M2M.resize(MAXLEVEL+1);
    mat_L2L.resize(MAXLEVEL+1);
    for(int level = 0; level <= MAXLEVEL; level++) {
      RealVec p_check_surf = surface(MULTIPOLE_ORDER,parent_coord,2.95,level);
      real_t s = R0 * powf(0.5, level+1);

      int numRelCoord = rel_coord[M2M_Type].size();
      mat_M2M[level].resize(numRelCoord);
      mat_L2L[level].resize(numRelCoord);
#pragma omp parallel for
      for(int i=0; i<numRelCoord; i++) {
        ivec3& coord = rel_coord[M2M_Type][i];
        real_t child_coord[3] = {(coord[0]+1)*s, (coord[1]+1)*s, (coord[2]+1)*s};
        RealVec c_equiv_surf = surface(MULTIPOLE_ORDER,child_coord,1.05,level+1);
        ComplexVec M_e2c(NSURF*NSURF);
        kernelMatrix(&c_equiv_surf[0], NSURF, &p_check_surf[0], NSURF, &M_e2c[0]);
        // M2M: child's upward_equiv to parent's check
        ComplexVec buffer(NSURF*NSURF);
        mat_M2M[level][i].resize(NSURF*NSURF);
        gemm(NSURF, NSURF, NSURF, &M_e2c[0], &(M2M_V[level][0]), &buffer[0]);
        gemm(NSURF, NSURF, NSURF, &buffer[0], &(M2M_U[level][0]), &(mat_M2M[level][i][0]));
        // L2L: parent's dnward_equiv to child's check, reuse surface coordinates
        M_e2c = transpose(M_e2c, NSURF, NSURF);
        mat_L2L[level][i].resize(NSURF*NSURF);
        gemm(NSURF, NSURF, NSURF, &(L2L_U[level][0]), &M_e2c[0], &buffer[0]);
        gemm(NSURF, NSURF, NSURF, &(L2L_V[level][0]), &buffer[0], &(mat_L2L[level][i][0]));
      }
    }
  }

  void PrecompM2LHelper() {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    // create fftw plan
    ComplexVec fftw_in(n3);
    RealVec fftw_out(2*n3);
    int dim[3] = {2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER, 2*MULTIPOLE_ORDER};
    fft_plan plan = fft_plan_dft(3, dim, reinterpret_cast<fft_complex*>(&fftw_in[0]), 
                                (fft_complex*)(&fftw_out[0]), FFTW_FORWARD, FFTW_ESTIMATE);
    // evaluate DFTs of potentials at convolution grids
    int numRelCoord = rel_coord[M2L_Helper_Type].size();
    mat_M2L_Helper.resize(numRelCoord);
    #pragma omp parallel for
    for(int i=0; i<numRelCoord; i++) {
      real_t coord[3];
      for(int d=0; d<3; d++) {
        coord[d] = rel_coord[M2L_Helper_Type][i][d] * R0 / 0.5;
      }
      RealVec conv_coord = conv_grid(coord, 0);
      RealVec r_trg(3, 0.0);
      ComplexVec conv_poten(n3);
      kernelMatrix(&conv_coord[0], n3, &r_trg[0], 1, &conv_poten[0]);
      mat_M2L_Helper[i].resize(2*n3);
      fft_execute_dft(plan, reinterpret_cast<fft_complex*>(&conv_poten[0]), (fft_complex*)(&mat_M2L_Helper[i][0]));
    }
    fft_destroy_plan(plan);
  }

  void PrecompM2L() {
    int n1 = MULTIPOLE_ORDER * 2;
    int n3 = n1 * n1 * n1;
    int numParentRelCoord = rel_coord[M2L_Type].size();
    int numChildRelCoord = rel_coord[M2L_Helper_Type].size();
    mat_M2L.resize(numParentRelCoord);
    RealVec zero_vec(n3*2, 0);
    #pragma omp parallel for schedule(dynamic)
    for(int i=0; i<numParentRelCoord; i++) {
      ivec3& parentRelCoord = rel_coord[M2L_Type][i];
      std::vector<real_t*> M_ptr(NCHILD*NCHILD, &zero_vec[0]);
      for(int j1=0; j1<NCHILD; j1++) {
        for(int j2=0; j2<NCHILD; j2++) {
          int childRelCoord[3]= { parentRelCoord[0]*2 - (j1/1)%2 + (j2/1)%2,
                                  parentRelCoord[1]*2 - (j1/2)%2 + (j2/2)%2,
                                  parentRelCoord[2]*2 - (j1/4)%2 + (j2/4)%2 };
          for(int k=0; k<numChildRelCoord; k++) {
            ivec3& childRefCoord = rel_coord[M2L_Helper_Type][k];
            if (childRelCoord[0] == childRefCoord[0] &&
                childRelCoord[1] == childRefCoord[1] &&
                childRelCoord[2] == childRefCoord[2]) {
              M_ptr[j2*NCHILD+j1] = &mat_M2L_Helper[k][0];
              break;
            }
          }
        }
      }
      mat_M2L[i].resize(n3*2*NCHILD*NCHILD);  // N3 by (2*NCHILD*NCHILD) matrix
      for(int k=0; k<n3; k++) {                      // loop over frequencies
        for(size_t j=0; j<NCHILD*NCHILD; j++) {       // loop over child's relative positions
          int index = k*(2*NCHILD*NCHILD) + 2*j;
          mat_M2L[i][index+0] = M_ptr[j][k*2+0]/n3;   // real
          mat_M2L[i][index+1] = M_ptr[j][k*2+1]/n3;   // imag
        }
      }
    }
  }

  void Precompute() {
    PrecompCheck2Equiv();
    PrecompM2M();
    PrecompM2LHelper();
    PrecompM2L();
  }
}//end namespace
#endif
