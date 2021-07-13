#ifndef COVER_MPI_MAIN_H
#define COVER_MPI_MAIN_H

#include <mpi.h>

typedef int mpi_main_t(MPI_Comm comm, int shmGroupRoot, int argc, char *argv[]);

#endif
