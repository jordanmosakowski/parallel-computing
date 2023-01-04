CC = gcc

default: matmult matmult_omp

matmult: matmult.c
	${CC} -O3 -Wall -Wextra -o $@ matmult.c -std=c99

matmult_omp: $(SRC)
	${CC} -O3 -Wall -Wextra -fopenmp -o $@ matmult_omp.c -std=c99

clean:
	-rm -f matmult
	-rm -f matmult_omp