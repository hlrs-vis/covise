/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SHOW_GRID_H_
#define _SHOW_GRID_H_

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE CuttingPlane application module                   **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C) 1994                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, D.Rantzau                                             **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include <api/coSimpleModule.h>
#include <util/coviseCompat.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoUnstructuredGrid.h>

using namespace covise;

class ShowGrid : public coSimpleModule
{
private:
    enum
    {
        ALL_LINES = 0,
        HULL = 1,
        THREE_SIDES_PPP = 2,
        THREE_SIDES_PPN = 3,
        THREE_SIDES_PNP = 4,
        THREE_SIDES_PNN = 5,
        THREE_SIDES_NPP = 6,
        THREE_SIDES_NPN = 7,
        THREE_SIDES_NNP = 8,
        THREE_SIDES_NNN = 9,
        BOUND_BOX = 10,
        EDGES = 11,
        CELL = 12
    };

    coDoLines *Lines;
    coDoPoints *Points;
    const char *LinesN;
    const char *PointsN;

    const char *colorn;
    const char *gtype;

    float *x_in, *y_in, *z_in, *x_ina, *y_ina, *z_ina;
    int i_dim, j_dim, k_dim, option;
    int *cl, *el, *tl, *vl, *pl, numelem, numconn, numcoord;
    long pos;

    void genLinesAndPoints(void);
    // genLinesAndPoints calls the following functions:
    void polygons(void);
    void rct_all_lines(void);
    void rct_hull(void);
    void rct_three_sides(void);
    // inline auxiliary functions for rct_three_sides
    //                            and str_three_sides
    int ref_x(int index_x)
    {
        return ((option - THREE_SIDES_PPP) & 4) ? i_dim - 1 - index_x : index_x;
    }
    int ref_y(int index_y)
    {
        return ((option - THREE_SIDES_PPP) & 2) ? j_dim - 1 - index_y : index_y;
    }
    int ref_z(int index_z)
    {
        return ((option - THREE_SIDES_PPP) & 1) ? k_dim - 1 - index_z : index_z;
    }
    void rct_box(void);
    void str_all_lines(void);
    void str_hull(void);
    void str_three_sides(void);
    void str_box(void);
    void str_curv_box(void);
    int index(int i, int j, int k)
    {
        return i * j_dim * k_dim + j * k_dim + k;
    }
    void unsgrd_elem(void);
    void str_cell(void);

    /*
            void create_strgrid_plane();
            void create_rectgrid_plane();
            void create_unigrid_plane();
            void create_scalar_plane();
            void create_vector_plane();
      */

    /// Ports and params;
    coInputPort *p_meshIn;
    coOutputPort *p_lines, *p_points;
    coChoiceParam *p_options;
    coIntSliderParam *p_pos;

    virtual void preHandleObjects(coInputPort **);

    const coDistributedObject *data_obj;
    const coDoPolygons *poly_in;
    const coDoUniformGrid *u_grid_in;
    const coDoRectilinearGrid *r_grid_in;
    const coDoStructuredGrid *s_grid_in;
    const coDoUnstructuredGrid *uns_grid_in;

public:
    ShowGrid(int argc, char *argv[]);
    int compute(const char *port);

    /*
          void genLinesAndPoints(coDoLines **Lines, coDoPoints **Points,
                                 const char *LinesN, const char *PointsN);
      */
};
#endif // _APPLICATION_H
