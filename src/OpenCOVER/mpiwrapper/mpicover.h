#ifndef COVER_MPI_MAIN_H
#define COVER_MPI_MAIN_H

#include <mpi.h>
#include <pthread.h>

typedef int mpi_main_t(MPI_Comm comm, int shmGroupRoot, pthread_barrier_t *shmBarrier, int argc, char *argv[]);

#endif
