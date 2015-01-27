/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_CELLTOVERT_H
#define CO_CELLTOVERT_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                     (C)2005 Visenso ++
// ++ Description: Interpolation from Cell Data to Vertex Data            ++
// ++                 ( CellToVert module functionality  )                ++
// ++                                                                     ++
// ++ Author: Sven Kufer( sk@visenso.de)                                  ++
// ++                                                                     ++
// ++**********************************************************************/

#include <covise/covise.h>

namespace covise
{

class coDistributedObject;

class ALGEXPORT coCellToVert
{
private:
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    // Original algorithm by Andreas Werner: + Assume data value related to center of elements.
    //                                       + Calculate the average values weighted by the distance to the
    //                                           surrounding center of elements.
    //
    //        - only implemented for unstructured grids
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    bool weightedAlgo(int num_elem, int num_conn, int num_point,
                      const int *elem_list, const int *conn_list, const int *type_list,
                      const int *neighbour_cells, const int *neighbour_idx,
                      const float *xcoord, const float *ycoord, const float *zcoord,
                      int numComp, const float *in_data_0, const float *in_data_1, const float *in_data_2,
                      float *out_data_0, float *out_data_1, float *out_data_2);

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //   Take the average value of all elements which the point includes
    //
    //   implemtented for POLYGN, LINES, UNSGRD
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    bool simpleAlgo(int num_elem, int num_conn, int num_point,
                    const int *elem_list, const int *conn_list,
                    int numComp, const float *in_data_0, const float *in_data_1, const float *in_data_2,
                    float *out_data_0, float *out_data_1, float *out_data_2);

public:
    typedef enum
    {
        SQR_WEIGHT = 1,
        SIMPLE = 2
    } Algorithm;

    //
    //  geoType/dataType: type string, .e.g. "UNSGRD"
    //  objName: name of the coDistributedObject to create
    //
    //  returns NULL in case of an error
    //
    coDistributedObject *interpolate(bool unstructured, int num_elem, int num_conn, int num_point,
                                     const int *elem_list, const int *conn_list, const int *type_list, const int *neighbour_cells, const int *neighbour_idx,
                                     const float *xcoord, const float *ycoord, const float *zcoord,
                                     int numComp, int &dataSize, const float *in_data_0, const float *in_data_1, const float *in_data_2,
                                     const char *objName, Algorithm algo_option = SIMPLE);

    coDistributedObject *interpolate(const coDistributedObject *geo_in,
                                     int numComp, int &dataSize, const float *in_data_0, const float *in_data_1, const float *in_data_2,
                                     const char *objName, Algorithm algo_option = SIMPLE);

    // note: no attributes copied
    coDistributedObject *interpolate(const coDistributedObject *geo_in, const coDistributedObject *data_in, const char *objName,
                                     Algorithm algo_option = SIMPLE);

    //
    //  returns false in case of an error
    //
    bool interpolate(bool unstructured, int num_elem, int num_conn, int num_point,
                     const int *elem_list, const int *conn_list, const int *type_list, const int *neighbour_cells, const int *neighbour_idx,
                     const float *xcoord, const float *ycoord, const float *zcoord,
                     int numComp, int &dataSize, const float *in_data_0, const float *in_data_1, const float *in_data_2,
                     float *out_data_0, float *out_data_1, float *out_data_2, Algorithm algo_option = SIMPLE);
};
}

#endif
