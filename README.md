# Parallel Computing
This repository contains code from 3 projects completed for the Parallel Computing course offered at Santa Clara University, completed in Fall 2022.

## Project 1: Dense Matrix Multiplication (OpenMP)

This project used block and tile methods discussed in class to create a parallel implementation of a dense matrix multiplication algorithm.

Below are the results of the block and tile methods for various matrix sizes, with analysis run on 1, 2, 4, 8, 12, 14, 16, 20, 24, and 28 threads:
![Project 1 results](/dense-matmult/results.svg)

## Project 2: Sparse Matrix Multiplication (OpenMP)

An implementation of the accumulator sparse matrix multiplication algorithm in OpenMP.

Below are the results of the algorithm using a 10% fill factor, with analysis run on 1, 2, 4, 8, 12, 14, 16, 20, 24, and 28 threads:

![Project 2 results](/sparse-matmult/results.svg)

## Project 3: Dijkstra's Algorithm (OpenMP and MPI)

Dijkstra’s algorithm is a way to find the shortest path between two nodes in a graph. To find such a path, it runs n-1 iterations, where n is the number of nodes in the graph. For each iteration, it finds the node that is closest to the source node that hasn’t already been updated. Then, it uses that node’s list of distances to update the source’s list of distances.

Below are the results of the OpenMP and MPI implementations of Dijkstra's Algorithm, with analysis run on 1, 2, 4, 8, 16, and 28 threads/processes:

![Project 3 results](/dijkstra/results.svg)