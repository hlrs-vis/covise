#ifndef MPI_COVER_EXPORT_H
#define MPI_COVER_EXPORT_H

#include <util/coExport.h>

#if defined(mpicover_EXPORTS)
#define MPICOVEREXPORT COEXPORT
#else
#define MPICOVEREXPORT COIMPORT
#endif

#endif
