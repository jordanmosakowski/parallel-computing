/* assert */
#include <assert.h>

/* errno */
#include <errno.h>

/* fopen, fscanf, fprintf, fclose */
#include <stdio.h>

/* EXIT_SUCCESS, EXIT_FAILURE, malloc, free */
#include <stdlib.h>

#include <omp.h>

static int create_mat(size_t const nrows, size_t const ncols, double ** const matp)
{
    double * mat=NULL;
    if (!(mat = (double*) malloc(nrows*ncols*sizeof(*mat)))) {
        goto cleanup;
    }

    /** Initialize matrix with random values **/
    for(size_t i = 0; i < nrows; i++){
        for (size_t j = 0; j < ncols; j++){
            mat[(i * ncols) + j] = (double)(rand() % 1000) / 353.0;
        }
    }
    /** End random initialization **/

    *matp = mat;

    return 0;

    cleanup:
    free(mat);
    return -1;
}

static int mult_mat_omp_block(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{
  size_t i, j, k;
  double sum;
  double * C = NULL;

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }
  
  // Parallelize each column in A and row in B (1-d tiling)
  #pragma omp parallel for collapse(2)
  for (i=0; i<n; ++i) {
    for (j=0; j<p; ++j) {
      for (k=0, sum=0.0; k<m; ++k) {
        sum += A[i*m+k] * B[k*p+j];
      }
      C[i*p+j] = sum;
    }
  }

  *Cp = C;

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}

static void multiply_tile(size_t const m, size_t const p,
                    size_t const nStart, size_t const mStart, size_t const pStart,
                    size_t const nEnd, size_t const mEnd, size_t const pEnd,
                    double const * const A, double const * const B,
                    double* const C, omp_lock_t * locks) {
  size_t i, j, k;
  double sum;
  // Iterate across each element in the tile and update the output matrix C            
  for (i=nStart; i<nEnd; ++i) {
    for (j=pStart; j<pEnd; ++j) {
      for (k=mStart, sum=0.0; k<mEnd; ++k) {
        sum += A[i*m+k] * B[k*p+j];
      }
      // Use locks to ensure multiple threads don't try to write to the same piece of memory at the same time.
      omp_set_lock(&(locks[i]));
      C[i*p+j] += sum;
      omp_unset_lock(&(locks[i]));
    }
  }
}


static int mult_mat_omp_tile(size_t const n, size_t const m, size_t const p,
                    double const * const A, double const * const B,
                    double ** const Cp)
{
  size_t nSize, mSize, pSize;
  size_t numthreads = omp_get_max_threads();

  // Calculate tile sizes
  nSize = n * numthreads / 8;
  mSize = m * numthreads / 8;
  pSize = p * numthreads / 8;
  
  // Apply upper and lower limits for tile dimensions
  if(nSize > 1600) {
    nSize = 1600;
  }
  if(nSize > n/2) {
    nSize = n/2;
  }
  else if(nSize < 128) {
    nSize = 128;
  }

  if(mSize > 1600) {
    mSize = 1600;
  }
  if(mSize > m/2) {
    mSize = m/2;
  }
  else if(mSize < 128) {
    mSize = 128;
  }

  if(pSize > 1600) {
    pSize = 1600;
  }
  if(pSize > p/2) {
    pSize = p/2;
  }
  else if(pSize < 128) {
    pSize = 128;
  }


  size_t i, j, k;
  size_t nEnd, mEnd, pEnd;
  double * C = NULL;
  omp_lock_t * locks = NULL;

  if (!(C = (double*) malloc(n*p*sizeof(*C)))) {
    goto cleanup;
  }
  if (!(locks = (omp_lock_t*) malloc(n*p*sizeof(omp_lock_t)))) {
    goto cleanup;
  }

  for(i=0; i<n*p; i++) {
    omp_init_lock(&(locks[i]));
  }

  // Call the multiply_tile function for each tile in the matrix.  
  #pragma omp parallel for collapse(3)
  for (i=0; i<n; i+=nSize) {
    for (j=0; j<p; j+=pSize) {
      for (k=0; k<m; k+=mSize) {
        // Ensure tiles don't go out of bounds
        nEnd = i + nSize;
        if(nEnd > n) {
          nEnd = n;
        }
        pEnd = j + pSize;
        if(pEnd > p) {
          pEnd = p;
        }
        mEnd = k + mSize;
        if(mEnd > m) {
          mEnd = m;
        }

        multiply_tile(m,p,i,k,j,nEnd,mEnd,pEnd,A,B,C,locks);
      }
    }
  }

  *Cp = C;

  return 0;

  cleanup:
  free(C);

  /*failure:*/
  return -1;
}


int main(int argc, char * argv[])
{
  // size_t stored an unsigned integer
  size_t nrows, ncols, ncols2, nthreads;
  double * A=NULL, * B=NULL, * C=NULL, * C2=NULL;

  double timer;

  if (argc != 4 && argc != 5) {
    fprintf(stderr, "usage: matmult nrows ncols ncols2 [number of threads]\n");
    goto failure;
  }

  nrows = atoi(argv[1]);
  ncols = atoi(argv[2]);
  ncols2 = atoi(argv[3]);

  // Fetch the optional parameter for the number of threads. 
  // If not provided, use 2 by deafult
  if(argc == 5) {
    nthreads = atoi(argv[4]);
  }
  else {
    nthreads = 2;
  }
  omp_set_dynamic(0);
  omp_set_num_threads(nthreads);

  if (create_mat(nrows, ncols, &A)) {
    perror("error");
    goto failure;
  }

  if (create_mat(ncols, ncols2, &B)) {
    perror("error");
    goto failure;
  }

  // Run with 1-D tiling (blocks)
  printf("Running with blocks (%d threads)\n", (int)nthreads);
  timer = omp_get_wtime();
  if (mult_mat_omp_block(nrows, ncols, ncols2, A, B, &C)) {
    perror("error");
    goto failure;
  }
  printf("Time: %f sec\n", omp_get_wtime() - timer); 

  // Run with 2-D tiling (tiling)
  printf("Running with tiles (%d threads)\n", (int)nthreads);
  timer = omp_get_wtime();
  if (mult_mat_omp_tile(nrows, ncols, ncols2, A, B, &C2)) {
    perror("error");
    goto failure;
  }
  printf("Time: %f sec\n", omp_get_wtime()-timer); 

  // Checks if the two matrices have the same output to verify that the multiplication was done correctly.
  double EPSILON = 0.00001;
  for(size_t i=0; i<nrows * ncols2; i++) {
    double diff = C[i] - C2[i];
    if((diff > EPSILON) || (-diff > EPSILON)) {
      fprintf(stderr, "Error: Matrices do not match at index %d: %f %f\n", (int) i, C[i], C2[i]);
      goto failure;
    }
  }
  
  // Cleanup
  free(A);
  free(B);
  free(C);
  free(C2);

  return EXIT_SUCCESS;

  failure:
  if(A){
    free(A);
  }
  if(B){
    free(B);
  }
  if(C){
    free(C);
  }
  return EXIT_FAILURE;
}
