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
#include <mpi.h>
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
  int * const nrowsp,
  float ** const ap, int ** offsetsp,
  int rank, int numProcesses
)
{

    MPI_Status status;
  float * a;
  int * offsets;
  offsets = (int*) malloc(numProcesses*sizeof(*offsets));
  int i, j, n, ret, numRows;
  if(rank == 0) {
    FILE * fp=NULL;

    /* open the file */
    fp = fopen(filename, "r");
    assert(fp);

    /* get the number of nodes in the graph */
    ret = fscanf(fp, "%d", &n);
    assert(1 == ret);

    // int offset = 0;

    int size = n / numProcesses;
    int remainder = n % numProcesses;

    int off = 0;
    for (int i = 0; i < numProcesses; i++) {
      offsets[i] = off;
      off += size;
      if(i < remainder) {
        off++;
      }
    }

    for(i=0; i<numProcesses; i++) {
      numRows = size;
      if(i < remainder) {
        numRows++;
      }

      if(i != 0){
        MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(&numRows, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        MPI_Send(offsets, numProcesses, MPI_INT, i, 2, MPI_COMM_WORLD);
      }

      /* allocate memory for local values */
      a = (float*) malloc(n*numRows*sizeof(*a));
      assert(a);
      /* read in roots local values */
      for (j=0; j<n*numRows; ++j) {
        ret = fscanf(fp, "%f", &a[j]);
        assert(1 == ret);
      }

      if(i == 0) {
        /* record output values */
        *ap = a;
        *nrowsp = numRows;
        *offsetsp = offsets;
      }
      else{
        MPI_Send(a, n*numRows, MPI_INT, i, 3, MPI_COMM_WORLD);
      }
    }

    /* close file */
    ret = fclose(fp);
    assert(!ret);
  }
  else {
    MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    MPI_Recv(&numRows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    *nrowsp = numRows;
    MPI_Recv(offsets, numProcesses, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
    *offsetsp = offsets;
    a = (float*) malloc(n*numRows*sizeof(*a));
    MPI_Recv(a, n*numRows, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
    *ap = a;
  }

  *np = n;
}

static void
dijkstra(
  int s,
  const int n,
  const float * const a,
  const int * const offsets,
  const int numRows,
  float ** const lp,
  int rank, int numProcesses
)
{
  int i, j;
  struct float_int {
    float l;
    int u;
  } min;
  char * m;
  float * l, * l2;

  m = (char*) calloc(n, sizeof(*m));
  assert(m);

  l = (float*) malloc(n*sizeof(*l));
  assert(l);

  l2 = (float*) malloc(n*sizeof(*l2));
  assert(l2);

  MPI_Bcast(&s,1, MPI_INT, 0, MPI_COMM_WORLD);
  int sProcess = 0;
  for(i=numProcesses-1; i>=0; --i) {
    if(s>=offsets[i]) {
      sProcess = i;
      break;
    }
  }

  // load initial values from the process that contains the source node
  if(rank == sProcess) {
    for (i=0; i<n; ++i) {
      l[i] = a[(s-offsets[rank])*n + i];
    }
  }
  //one to all
  MPI_Bcast(l,n, MPI_FLOAT, sProcess, MPI_COMM_WORLD);

  m[s] = 1;
  min.u = -1; /* avoid compiler warning */

  for (i=1; i<n; ++i) {
    min.l = INFINITY;

    /* find local minimum */
    for (j=0; j<n; ++j) {
      if (!m[j] && l[j] < min.l) {
        min.l = l[j];
        min.u = j;
      }
    }

    m[min.u] = 1;

    for (j=0; j<numRows; j++) {
      if (!m[offsets[rank] + j] && min.l+a(j,min.u) < l[offsets[rank] + j]){
        l[offsets[rank] + j] = min.l+a(j,min.u);
      }
    }

    if(i%(n/5) == 0) {
      MPI_Allreduce(l, l2, n, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
      memcpy(l, l2, n * sizeof( float ));
    }
  }

  MPI_Allreduce(l, l2, n, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  memcpy(l, l2, n * sizeof( float ));

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

  int numProcesses, rank;

  MPI_Init(&argc, &argv);

  MPI_Comm_size (MPI_COMM_WORLD, &numProcesses);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

  int n, nrows;
  double ts, te;
  float * a, * l;
  int *offsets;

  if(argc < 3){
    if(rank == 0) {
      printf("Invalid number of arguments.\nUsage: dijkstra <graph> <num_sources> [<output_file>].\n");
    }
    return EXIT_FAILURE;
  }
  /* initialize random seed: */
  srand (time(NULL));
  unsigned int seed = time(NULL);

  /* figure out number of random sources to search from */
  int nsources = atoi(argv[2]);
  assert(nsources > 0);

  /* load data */
  if(rank == 0) {
    printf("Loading graph from %s.\n", argv[1]);
  }
  load(argv[1], &n, &nrows, &a, &offsets, rank, numProcesses);

  if(rank == 0) {
    printf("Performing %d searches from random sources.\n", nsources);
  }
  ts = MPI_Wtime();
  for(int i=0; i < nsources; ++i){
    dijkstra(rand_r(&seed) % n, n, a, offsets, nrows, &l, rank, numProcesses);
  }
  te = MPI_Wtime();
  if(rank == 0) {
    print_time((te-ts));
  }
  if(argc >= 4){
    if(rank == 0) {
      printf("Computing result for source 0.\n");
    }
    dijkstra(0, n, a, offsets, nrows, &l, rank, numProcesses);
    if(rank == 0){
      printf("Writing result to %s.\n", argv[3]);
      print_numbers(argv[3], n, l);
    }
  }

  MPI_Finalize();

  free(a);
  free(l);

  return EXIT_SUCCESS;
}
