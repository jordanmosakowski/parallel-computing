SRC := dijkstra.cpp
MPISRC := dijkstra_mpi.cpp
OMPSRC := dijkstra_omp.cpp

default: dijkstra dijkstra_omp dijkstra_mpi

dijkstra: $(SRC)
	g++ -O3 -Wall -Wextra -o $@ $<

dijkstra_mpi: $(MPISRC)
	mpic++ -O3 -Wall -Wextra -Wno-cast-function-type -o $@ $<

dijkstra_omp: $(OMPSRC)
	g++ -fopenmp -O3 -Wall -Wextra -o $@ $<

clean: 
	rm -f dijkstra dijkstra_mpi dijkstra_omp
