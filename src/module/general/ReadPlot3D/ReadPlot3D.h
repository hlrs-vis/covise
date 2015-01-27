/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Read module for Plot3D data                               **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Lars Frenzel                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:                                                                  **
\**************************************************************************/

#ifndef _READ_PLOT3D_H
#define _READ_PLOT3D_H

// includes
#include "appl/ApplInterface.h"
#include "util/coviseCompat.h"

// defines
#define _FILE_TYPE_BINARY 1
#define _FILE_TYPE_FORTRAN 2
#define _FILE_TYPE_FORTRAN64 3
#define _FILE_TYPE_ASCII 4

#define _FILE_STRUCTURED_GRID 4
#define _FILE_UNSTRUCTURED_GRID 5
#define _FILE_IBLANKED 6

#define _FILE_SOLUTION 7
#define _FILE_DATA 8
#define _FILE_FUNCTION 9

#define _FILE_SINGLE_ZONE 10
#define _FILE_MULTI_ZONE 11

#define _READ_GRID 12
#define _READ_DATA 13

using namespace covise;

class Application
{

private:
    // callback-stuff
    static void computeCallback(void *userData, void *callbackData);

    // main
    void compute(const char *port);

    // parameters
    char *grid_path;
    char *data_path;
    int filetype, gridtype, datatype, subtype;
    char buffer[1024];
    int new_buffer;
    long int remove_timesteps;
    long int max_timesteps;
    // will we have to convert between LE<->BE
    int byteswap_flag;
    void byteswap(void *p, int n);

    // functions
    coDistributedObject **ReadPlot3D(FILE *fFile, int read_flag, const char *name1, const char *name2, const char *name3);

    // sub-functions
    void file_read(FILE *f, double *p, int n);
    void file_read(FILE *f, float *p, int n);
    void file_read(FILE *f, int *p, int n);
    void file_read(FILE *f, coInt64 *p, int n);
    void file_read_ascii(FILE *f, char *p);

    void file_beginBlock(FILE *f);
    int file_endBlock(FILE *f);

    void read_size_header(FILE *f, int *x, int *y, int *z, int *n = NULL);
    void read_multi_header(FILE *f, int **x, int **y, int **z, int nblocks);
    void read_structured_grid_record(FILE *f,
                                     float *x, int x_dim,
                                     float *y, int y_dim,
                                     float *z, int z_dim,
                                     int n, int *iblank,
                                     int gridtype);
    void read_solution_conditions(FILE *f, float *mach = NULL, float *alpha = NULL,
                                  float *re = NULL, float *time = NULL);
    void set_solution_attributes(coDistributedObject *, float, float, float, float);
    void read_solution_record(FILE *f, float *val, int x_dim, int y_dim,
                              int z_dim, int n);
    void read_data_header(FILE *f, int *x, int *y, int *z, int *c, int *n = NULL);
    void read_multi_data_header(FILE *f, int **x, int **y, int **z, int **c, int nblocks);
    void read_data_record(FILE *f, float *val, int x_dim, int y_dim, int z_dim, int n);
    void read_iblanked(FILE *f, int n, int *p = NULL);
    void read_nzones(FILE *f, int *n);

    void read_unstructured_header(FILE *f, int *points, int *triang, int *tetra);
    void read_unstructured_coord(FILE *f, float *x, float *y, float *z, int n);
    void read_single_triangle(FILE *f, int *v1, int *v2, int *v3);
    void read_single_tetrahedra(FILE *f, int *v1, int *v2, int *v3, int *v4);
    void read_unstructured_triangle_flags(FILE *f, int n, int *flags = NULL);

public:
    Application(int argc, char *argv[]);
    void run()
    {
        Covise::main_loop();
    }

private:
    int blockSize;
    coInt64 blockSize64;
};
#endif // _READ_PLOT3D_H
