/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SPLIT_USG_H
#define _SPLIT_USG_H

/**************************************************************************\ 
 **                                                     (C)2001 Vircinity  **
 **                                                                        **
 ** Description: Splits grid into subgrids with elements of the same       **
 **              dimensionality                                            **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Sergio Leseduarte                           **
 **                            Vircinity GmbH                              **
 **                            Nobelstr. 15                                **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  29.1.2001  (coding begins)                                      **
 ** Sergio Leseduarte:    6.2.2001                                         **
 **                   The original version was modified in order to change **
 **                   the types of some output objects; what originally    **
 **                   was a 2D coDoUnstructuredGrid is now coDoPolygons,     **
 **                   1D coDoUnstructuredGrid has been substituted for      **
 **                   coDoLines and 0D coDoUnstructuredGrid for coDoPoints.   **
\**************************************************************************/

#include <api/coSimpleModule.h>
using namespace covise;
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoData.h>
#include <util/coviseCompat.h>

// #define _DEBUG_SPLIT_

class SplitUSG : public coSimpleModule
{
private:
    virtual int compute(const char *port);

    // auxiliary functions (create and fill lists of output grids and data)
    void build_out_3D_grid();
    void build_out_2D_grid();
    void build_out_1D_grid();
    void build_out_0D_grid();

    // put_marks_in_map fills the map coord_map with marks
    // a map is an array of as many ints as number of points
    // in the original grid (see below, coord_map), whose
    // elements are set to 0 on entering the build_out_?D_grid
    // functions. Then the grid elements are sequentially investigated,
    // and for any one with the required dimensionality, its
    // points are assigned a positive number (mark) in coord_map.
    // Positive numbers are assigned starting from 1 and
    // without skipping any number. If in this process
    // a point is found to be already marked with a positive number,
    // then this number is not modified (it has already been
    // previously registered).
    void put_marks_in_map(int, int);

    // write_conn_list fills the connectivity list of an
    // output grid. In this process coord_map is used.
    // An element of coord_map is accessed using the
    // connectivity list of the input grid and we decrement
    // its value by 1 in order to obtain the value for
    // the element of the connectivity list being filled
    void write_conn_list(int *, unsigned int, int, unsigned int);

    // fills lists for output data
    void fill_data_objects(unsigned int);

    // input and output ports
    coInputPort *p_grid, *p_data_scal, *p_data_vect;
    coOutputPort *p_grid_3D, *p_data_3D_scal, *p_data_3D_vect;
    coOutputPort *p_grid_2D, *p_data_2D_scal, *p_data_2D_vect;
    coOutputPort *p_grid_1D, *p_data_1D_scal, *p_data_1D_vect;
    coOutputPort *p_grid_0D, *p_data_0D_scal, *p_data_0D_vect;

    // Flags to distinguish cell-centred from coordinate-based data
    enum
    {
        S_PER_CELL,
        S_PER_VERTEX,
        S_NULL
    } scal_cell_or_vertex;
    enum
    {
        V_PER_CELL,
        V_PER_VERTEX,
        V_NULL
    } vect_cell_or_vertex;

    // input objects
    coDoUnstructuredGrid *in_grid;
    coDoFloat *in_data_scal;
    coDoVec3 *in_data_vect;

    // Auxiliary data for input objects
    int *elem_list, *conn_list; // addresses of in_grid
    float *x_in, *y_in, *z_in; // addresses of in_grid
    int *type_list; // list with element types
    float *scal_data; // list of scalar data
    //lists of vector data
    float *vect_data_x, *vect_data_y, *vect_data_z;

    // Size of input objects
    int numEl, numConn, numCoord; // grid size
    int scal_num_of_points; // size of scalar data
    int vect_num_of_points; // size of vector data

    // Coordinate map
    int *coord_map;
    int mark;

    // Output objects
    coDoUnstructuredGrid *out_grid_3D;
    coDoFloat *out_data_3D_scal;
    coDoVec3 *out_data_3D_vect;
    coDoPolygons *out_grid_2D;
    coDoFloat *out_data_2D_scal;
    coDoVec3 *out_data_2D_vect;
    coDoLines *out_grid_1D;
    coDoFloat *out_data_1D_scal;
    coDoVec3 *out_data_1D_vect;
    coDoPoints *out_grid_0D;
    coDoFloat *out_data_0D_scal;
    coDoVec3 *out_data_0D_vect;

#ifdef _DEBUG_SPLIT_
    int nou_grid_3D;
    int nou_data_3D_scal;
    int nou_data_3D_vect;
    int nou_grid_2D;
    int nou_data_2D_scal;
    int nou_data_2D_vect;
    int nou_grid_1D;
    int nou_data_1D_scal;
    int nou_data_1D_vect;
    int nou_grid_0D;
    int nou_data_0D_scal;
    int nou_data_0D_vect;
#endif

    virtual void copyAttributesToOutObj(coInputPort **input_ports,
                                        coOutputPort **output_ports, int i);

public:
    SplitUSG(int argc, char *argv[]);
    virtual ~SplitUSG(void)
    {
    }
    void postInst()
    {
        setCopyNonSetAttributes(0);
    }
};
#endif
