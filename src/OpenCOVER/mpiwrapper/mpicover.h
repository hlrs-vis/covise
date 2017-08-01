#ifndef COVER_MPI_MAIN_H
#define COVER_MPI_MAIN_H

#include <mpi.h>
#include "export.h"

extern "C" int MPICOVEREXPORT mpi_main(MPI_Comm comm, int argc, char *argv[]);

#endif
