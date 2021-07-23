#ifndef COVER_MPI_MAIN_H
#define COVER_MPI_MAIN_H

#include <mpi.h>
#ifdef __APPLE__
typedef void pthread_barrier_t;
#else
#include <pthread.h>
#endif

typedef int mpi_main_t(MPI_Comm comm, int shmGroupRoot, pthread_barrier_t *shmBarrier, int argc, char *argv[]);

#endif
