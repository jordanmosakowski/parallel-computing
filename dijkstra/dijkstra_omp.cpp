/* assert */
#include <assert.h>
/* INFINITY */
#include <math.h>
/* FILE, fopen, fclose, fscanf, rewind */
#include <stdio.h>
/* EXIT_SUCCESS, malloc, calloc, free */
#include <stdlib.h>
/* time, CLOCKS_PER_SEC */
#include <time.h>
#include <omp.h>
#include <string.h>

#define ROWMJR(R,C,NR,NC) (R*NC+C)
#define COLMJR(R,C,NR,NC) (C*NR+R)
/* define access directions for matrices */
#define a(R,C) a[ROWMJR(R,C,ln,n)]
#define b(R,C) b[ROWMJR(R,C,nn,n)]

static void
load(
  const char * const filename,
  int * const np,
  float ** const ap
)
{
  int i, j, n, ret;
  FILE * fp=NULL;
  float * a;

  /* open the file */
  fp = fopen(filename, "r");
  assert(fp);

  /* get the number of nodes in the graph */
  ret = fscanf(fp, "%d", &n);
  assert(1 == ret);

  /* allocate memory for local values */
  a = (float*) malloc(n*n*sizeof(*a));
  assert(a);

  /* read in roots local values */
  for (i=0; i<n; ++i) {
    for (j=0; j<n; ++j) {
      ret = fscanf(fp, "%f", &a(i,j));
      assert(1 == ret);
    }
  }

  /* close file */
  ret = fclose(fp);
  assert(!ret);

  /* record output values */
  *np = n;
  *ap = a;
}

static void
dijkstra(
  const int s,
  const int n,
  const float * const a,
  float ** const lp
)
{
  int i, j;
  struct float_int {
    float l;
    int u;
  } min;
  char * m;
  float * l;

  m = (char*) calloc(n, sizeof(*m));
  assert(m);

  l = (float*) malloc(n*sizeof(*l));
  assert(l);
  #pragma omp parallel for
  for (i=0; i<n; ++i) {
    l[i] = a(i,s);
  }

  m[s] = 1;
  min.u = -1; /* avoid compiler warning */

  for (i=1; i<n; ++i) {
    min.l = INFINITY;

    #pragma omp parallel
    {

      /* find local minimum */
      #pragma omp for
      for (j=0; j<n; ++j) {
        if (!m[j] && l[j] < min.l) {
          min.l = l[j];
          min.u = j;
        }
      }

      m[min.u] = 1;
      
      #pragma omp for
      for (j=0; j<n; ++j) {
        if (!m[j] && min.l+a(j,min.u) < l[j])
          l[j] = min.l+a(j,min.u);
      }

    }
  }

  free(m);

  *lp = l;
}

static void
print_time(const double seconds)
{
  printf("%0.06fs\n", seconds);
}

static void
print_numbers(
  const char * const filename,
  const int n,
  const float * const numbers)
{
  int i;
  FILE * fout;

  /* open file */
  if(NULL == (fout = fopen(filename, "w"))) {
    fprintf(stderr, "error opening '%s'\n", filename);
    abort();
  }

  /* write numbers to fout */
  for(i=0; i<n; ++i) {
    fprintf(fout, "%10.4f\n", numbers[i]);
  }

  fclose(fout);
}

int
main(int argc, char ** argv)
{
  int n, nthreads;
  double ts, te;
  float * a, * l;

  if(argc < 3 || argc > 6){
     printf("Invalid number of arguments.\nUsage: dijkstra <graph> <num_sources> [<output_file>] [-t num_threads].\n");
     return EXIT_FAILURE;
  }

  if(argc >=4 && strcmp(argv[3],"-t") == 0) {
    nthreads = atoi(argv[4]);
  }
  else if(argc >=5 && strcmp(argv[4],"-t") == 0) {
    nthreads = atoi(argv[5]);
  }
  else {
    nthreads = 1;
  }
  printf("Running with %d threads\n", nthreads);

  omp_set_dynamic(0);
  omp_set_num_threads(nthreads);
  /* initialize random seed: */
  srand (time(NULL));
  unsigned int seed = time(NULL);

  /* figure out number of random sources to search from */
  int nsources = atoi(argv[2]);
  assert(nsources > 0);

  /* load data */
  printf("Loading graph from %s.\n", argv[1]);
  load(argv[1], &n, &a);

  printf("Performing %d searches from random sources.\n", nsources);
  ts = omp_get_wtime();
  #pragma omp parallel for
  for(int i=0; i < nsources; ++i){
    dijkstra(rand_r(&seed) % n, n, a, &l);
  }
  te = omp_get_wtime();

  print_time(te-ts);
  if(argc >= 4 && strcmp(argv[3],"-t") != 0){
    printf("Computing result for source 0.\n");
    dijkstra(0, n, a, &l);
    printf("Writing result to %s.\n", argv[3]);
    print_numbers(argv[3], n, l);
  }

  free(a);
  free(l);

  return EXIT_SUCCESS;
}
