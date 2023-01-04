#include <iostream>
#include <omp.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <cstring>      /* strcasecmp */
#include <cstdint>
#include <assert.h>
#include <vector>       // std::vector
#include <algorithm>    // std::random_shuffle
#include <random>
#include <stdexcept>
#include <iomanip>
#include <list>

using namespace std;

using idx_t = std::uint32_t;
using val_t = float;
using ptr_t = std::uintptr_t;

/**
 * CSR structure to store search results
 */
typedef struct csr_t {
  idx_t nrows; // number of rows
  idx_t ncols; // number of rows
  idx_t * ind; // column ids
  val_t * val; // values
  ptr_t * ptr; // pointers (start of row in ind/val)

  csr_t()
  {
    nrows = ncols = 0;
    ind = nullptr;
    val = nullptr;
    ptr = nullptr;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param nnz   Number of non-zeros
   */
  void reserve(const idx_t nrows, const ptr_t nnz)
  {
    if(nrows > this->nrows){
      if(ptr){
        ptr = (ptr_t*) realloc(ptr, sizeof(ptr_t) * (nrows+1));
      } else {
        ptr = (ptr_t*) malloc(sizeof(ptr_t) * (nrows+1));
        ptr[0] = 0;
      }
      if(!ptr){
        throw std::runtime_error("Could not allocate ptr array.");
      }
    }
    if(nnz > ptr[this->nrows]){
      if(ind){
        ind = (idx_t*) realloc(ind, sizeof(idx_t) * nnz);
      } else {
        ind = (idx_t*) malloc(sizeof(idx_t) * nnz);
      }
      if(!ind){
        throw std::runtime_error("Could not allocate ind array.");
      }
      if(val){
        val = (val_t*) realloc(val, sizeof(val_t) * nnz);
      } else {
        val = (val_t*) malloc(sizeof(val_t) * nnz);
      }
      if(!val){
        throw std::runtime_error("Could not allocate val array.");
      }
    }
    this->nrows = nrows;
  }

  /**
   * Reserve space for more rows or non-zeros. Structure may only grow, not shrink.
   * @param nrows Number of rows
   * @param ncols Number of columns
   * @param factor   Sparsity factor
   */
  static csr_t * random(const idx_t nrows, const idx_t ncols, const double factor)
  {
    ptr_t nnz = (ptr_t) (factor * nrows * ncols);
    if(nnz >= nrows * ncols / 2.0){
      throw std::runtime_error("Asking for too many non-zeros. Matrix is not sparse.");
    }
    auto mat = new csr_t();
    mat->reserve(nrows, nnz);
    mat->ncols = ncols;

    /* fill in ptr array; generate random row sizes */
    unsigned int seed = (unsigned long) mat;
    long double sum = 0;
    for(idx_t i=1; i <= mat->nrows; ++i){
      mat->ptr[i] = rand_r(&seed) % ncols;
      sum += mat->ptr[i];
    }
    for(idx_t i=0; i < mat->nrows; ++i){
      double percent = mat->ptr[i+1] / sum;
      mat->ptr[i+1] = mat->ptr[i] + (ptr_t)(percent * nnz);
      if(mat->ptr[i+1] > nnz){
        mat->ptr[i+1] = nnz;
      }
    }
    if(nnz - mat->ptr[mat->nrows-1] <= ncols){
      mat->ptr[mat->nrows] = nnz;
    }

    /* fill in indices and values with random numbers */
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      unsigned int seed = (unsigned long) mat * (1+tid);
      std::vector<int> perm;
      for(idx_t i=0; i < ncols; ++i){
        perm.push_back(i);
      }
      std::random_device seeder;
      std::mt19937 engine(seeder());

      #pragma omp for
      for(idx_t i=0; i < nrows; ++i){
        std::shuffle(perm.begin(), perm.end(), engine);
        for(ptr_t j=mat->ptr[i]; j < mat->ptr[i+1]; ++j){
          mat->ind[j] = perm[j - mat->ptr[i]];
          mat->val[j] = ((double) rand_r(&seed)/rand_r(&seed));
        }
      }
    }

    return mat;
  }

  string info(const string name="") const
  {
    return (name.empty() ? "CSR" : name) + "<" + to_string(nrows) + ", " + to_string(ncols) + ", " +
      (ptr ? to_string(ptr[nrows]) : "0") + ">";
  }

  ~csr_t()
  {
    if(ind){
      free(ind);
    }
    if(val){
      free(val);
    }
    if(ptr){
      free(ptr);
    }
  }
} csr_t;



csr_t * transpose(csr_t * mat) {
  csr_t * result = new csr_t();
  result->ncols = mat->nrows;
  result->reserve(mat->ncols,mat->ptr[mat->nrows]);
  #pragma omp parallel for
  for(int i=0; i<result->nrows; i++) {
    result->ptr[i] = 0;
  }
  #pragma omp parallel for
  for(int i=0; i<mat->nrows; i++) {
    for(int j=mat->ptr[i]; j<mat->ptr[i+1]; j++){
      result->ptr[mat->ind[j]+1]++;
    }
  }
  for(int i=1; i<result->nrows; i++) {
    result->ptr[i] += result->ptr[i-1];
  }
  for(int i=0; i<mat->nrows; i++) {
    for(int j=mat->ptr[i]; j<mat->ptr[i+1]; j++){
      int idx = result->ptr[mat->ind[j]]++;
      result->val[idx] = mat->val[j];
      result->ind[idx] = i;
    }
  }
  for(int i=result->nrows; i>0; i--) {
    result->ptr[i] = result->ptr[i - 1];
  }
  result->ptr[0] = 0;
  return result;
}

/**
 * Ensure the matrix is valid
 * @param mat Matrix to test
 */
void test_matrix(csr_t * mat){
  auto nrows = mat->nrows;
  auto ncols = mat->ncols;
  assert(mat->ptr);
  auto nnz = mat->ptr[nrows];
  for(idx_t i=0; i < nrows; ++i){
    assert(mat->ptr[i] <= nnz);
  }
  for(ptr_t j=0; j < nnz; ++j){
    assert(mat->ind[j] < ncols);
  }
}

/**
 * Multiply A and B (transposed given) and write output in C.
 * Note that C has no data allocations (i.e., ptr, ind, and val pointers are null).
 * Use `csr_t::reserve` to increase C's allocations as necessary.
 * @param A  Matrix A.
 * @param B The transpose of matrix B.
 * @param C  Output matrix
 */
void sparsematmult(csr_t * A, csr_t * B, csr_t *C)
{
    auto Bt = transpose(B);
    test_matrix(Bt);
    C->ncols = B->ncols;
    int totalSize = 0;

    C->reserve(A->nrows,1);
    C->ptr[A->nrows] = 1;

    csr_t ** rows = new csr_t*[A->nrows];
    #pragma omp parallel for
    for(int i=0; i<A->nrows; i++) {
      rows[i] = new csr_t();
      val_t *accumulator = new val_t[Bt->ncols];
      for(int j=0; j<Bt->ncols; j++) {
        accumulator[j] = 0.0f;
      }
      vector<int> counter;

      for(int j=A->ptr[i]; j<A->ptr[i+1]; j++) {
        int aCol = A->ind[j];
        val_t val = A->val[j];
        int startK = Bt->ptr[aCol];
        int endK = Bt->ptr[aCol+1];
        for(int k = startK; k<endK; k++) {
          if(accumulator[Bt->ind[k]] == 0.0) {
            if(i == 100) {
            }
            counter.push_back(Bt->ind[k]);
          }
          accumulator[Bt->ind[k]] += val * Bt->val[k];
        }
      }
      rows[i]->ncols = C->ncols;
      int size = counter.size();
      if(i == 100) {
      }
      rows[i]->reserve(1,size);
      C->ptr[i+1] = size;
      #pragma omp critical
      totalSize += size;

      int idx=0;
      for(int j=0; j<size; j++) {
        int accIdx = counter[j];
        rows[i]->val[idx] = accumulator[accIdx];
        rows[i]->ind[idx] = accIdx;
        idx++;
      }
      free(accumulator);
    }
    C->reserve(C->nrows,totalSize);
    for(int i=0; i<A->nrows; i++) {
      memcpy(&C->val[C->ptr[i]], rows[i]->val, C->ptr[i+1] * sizeof( val_t ) );
      memcpy(&C->ind[C->ptr[i]], rows[i]->ind, C->ptr[i+1] * sizeof( idx_t ) );
      C->ptr[i+1] += C->ptr[i];
    }
}

int main(int argc, char *argv[])
{
  if(argc < 4){
    cerr << "Invalid options." << endl << "./sparsematmult <A_nrows> <A_ncols> <B_ncols> <fill_factor> [<num_threads>]" << endl;
    exit(1);
  }
  int nrows = atoi(argv[1]);
  int ncols = atoi(argv[2]);
  int ncols2 = atoi(argv[3]);
  double factor = atof(argv[4]);
  int nthreads = 1;
  if(argc == 7 && strcasecmp(argv[5], "-t") == 0){
    nthreads = atoi(argv[6]);
    omp_set_num_threads(nthreads);
  }

  cout << "A_nrows: " << nrows << endl;
  cout << "A_ncols: " << ncols << endl;
  cout << "B_ncols: " << ncols2 << endl;
  cout << "factor: " << factor << endl;
  cout << "nthreads: " << nthreads << endl;

  /* initialize random seed: */
  srand (time(NULL));

  auto A = csr_t::random(nrows, ncols, factor);
  auto B = csr_t::random(ncols2, ncols, factor); // Note B is already transposed.
  test_matrix(A);
  test_matrix(B);
  auto C = new csr_t(); // Note that C has no data allocations so far.

  cout << A->info("A") << endl;
  cout << B->info("B") << endl;

  auto t1 = omp_get_wtime();
  sparsematmult(A, B, C);
  auto t2 = omp_get_wtime();
  cout << C->info("C") << endl;

  double time = (t2-t1);
  cout << setprecision(4) << fixed;
  cout << "Execution time (seconds):" << time << endl;

  delete A;
  delete B;
  delete C;

  return 0;
}
