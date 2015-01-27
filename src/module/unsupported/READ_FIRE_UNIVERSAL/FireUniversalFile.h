/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FIREUNIVERSALFILE_H
#define FIREUNIVERSALFILE_H

#include <unistd.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#ifdef __hpux
#include <string.h>
#include <strings.h>
#endif

const int SCALAR_DATA = 1;
const int VECTOR_DATA = 3;

class FireFile
{
    char *name;
    FILE *hdl;
    char line[255];
    long start_gridpoints;
    long start_elements;
    long start_data[20];
    int type_data[20];

public:
    FireFile(char *n);
    char *read_line()
    {
        return fgets(line, 255, hdl);
    };
    char *get_filename()
    {
        return name;
    };
    int skip_block();
    int is_minus_one();
    int read_nodes(int &len);
    int read_nodes(int len, float *x, float *y, float *z);
    int read_elements(int &len);
    int read_elements(int len, int *no_elem, int *no_vert);
    int determine_data(int no_of_grid_points, long &data_start,
                       char **data_name, int &data_type);
    int read_data(int no_of_grid_points, long data_start,
                  float *s);
    int read_data(int no_of_grid_points, long data_start,
                  float *u, float *v, float *w);
    int read_vectordata(int &len, float **x, float **y, float **z){};
    int read_scalardata(int &len, float **d){};
    //#ifdef __hpux
    long set_fseek()
    {
        return ftell(hdl);
    };
    //#else
    //	long set_fseek() { return fseek(hdl, 0, SEEK_CUR); };
    //#endif
    long goto_fseek(long ls)
    {
        return fseek(hdl, ls, SEEK_SET);
    };
};
#endif
