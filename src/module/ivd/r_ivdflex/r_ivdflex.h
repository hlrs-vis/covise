/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _R_IVDFLEX_H
#define _R_IVDFLEX_H
/**************************************************************************\
**                                                   	      (C)1999 RUS **
**                                                                        **
** Description: Simple Reader for Wavefront OBJ Format	                  **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: D. Rainer                                                      **
**                                                                        **
** History:                                                               **
** April 99         v1                                                    ** 
** September 99     new covise api                                        **                               **
**                                                                        **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>

class ReadIVDdata : public coModule
{

private:
#define MAXscalar 24
#define MAXvector 1
#define VECTOR 1
#define SCALAR 2
#define EMPTY 3
#define LINE_size 500
#define STR_MAX 150
#define MAXgrids 250
    //  member functions
    virtual int compute(const char *);
    virtual void quit();

    int read_general(FILE *fp); //function to read general-file
    //    int read_gridconnectivity(FILE *filepointer,coDoUnstructuredGrid* pGrid_data);
    bool read_gridconnectivity2(coDoUnstructuredGrid *pGrid_data);
    /* reads connectivity from general file */
    bool open_data(FILE **dfp,
                   const char *generalfilename, const char *datafilename,
                   int stepnr);
    bool read_grid(coDoUnstructuredGrid *pGrid_data);
    bool skip_block(FILE *dfp); // skip one block, e.g. to skip grid information.
    bool read_scalar(FILE *dfp, int index, int local_coord, int offset); // read one scalar field
    bool read_vector(FILE *dfp, int index, int local_coord, int offset); // read one vector field

    // Parameter names
    const char *GridObjectName; // output object name assigned by the controller

    const char *general_filename; // file name of general File

    char *connection_filename; // file name of connectivity file
    FILE *fp;
    FILE *dfp;
    // Ports
    coOutputPort *p_Grid;
    coOutputPort *p_Vector[MAXvector];
    coOutputPort *p_Scalar[MAXscalar]; // Array of OutPorts that later can be
    // selected
    //  member data

    /*  General information read from the general file */
    int total_coord; //number of coordinates in total
    int total_elem; //number of elements in total
    int total_conn; //number of connections in total

    int local_coord[MAXgrids]; //number of coordinates in one field
    int local_elem[MAXgrids]; //number of elements in one field
    int local_conn[MAXgrids]; //number of connections in one field
    char *data_filename[MAXgrids]; //name of data file of one field

    int num_scalar; //number of scalar variables in data file
    int num_vector; //number of vector variables in data file
    int num_grids; //number of local grids to be read in general file
    char *field[MAXscalar + MAXvector]; //Desription of Data as read from field=
    int structure[MAXscalar + MAXvector]; //Is is vector or scalar
    int timsteps;
    int numberTimesteps; //this information is not read from general
    //file. It's chosen in Mapeditor
    int startTimesteps; //start with this timestepnr
    /*    int read_var[MAXscalar+MAXvector];    //should this data be read?
                                          //yes if selected in choice bottom
*/
    //  Pointer to File Browser that gets the general file name
    coFileBrowserParam *p_FileBrowser;

    //define output ports choices
    //    coChoiceParam *pSelectPort;
    //define number of gridfields;
    coIntScalarParam *pnumberTimestop;
    //define swapbyte variable
    coBooleanParam *pswapdata;
    int swapdata;
    //define starting number. Which is the first timestep that should be read
    coIntScalarParam *pnumberTimestart;
    //local data
    float *p_scalarcoord; //pointer to scalar coord field
    float *p_xcoord; //pointer to x_coords field
    float *p_ycoord;
    float *p_zcoord;
    int *p_el; //pointer to element list of unstructured grid
    int *p_vl; //pointer to vertex list of unstructured grid
    int *p_tl; //pointer to type list of unstructured grid

    float *p_scalar[MAXscalar];
    float *p_vector[MAXvector];

    //shared memory data
    //changed that to Unstructured Grid;
    coDoUnstructuredGrid *d_Grid;
    coDoVec3 *d_Vector[MAXvector];
    coDoFloat *d_Scalar[MAXscalar];

public:
    ReadIVDdata(int argc, char *argv[]);
    virtual ~ReadIVDdata();
};

#endif
