/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_CELLTOVERT_H
#define CO_CELLTOVERT_H

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                     (C)2005 Visenso ++
// ++ Description: Remove unused points                                   ++
// ++                 ( FixUsg module functionality  )                    ++
// ++                                                                     ++
// ++ Author: Sven Kufer( sk@visenso.de)                                  ++
// ++                                                                     ++
// ++**********************************************************************/

#include <covise/covise.h>

namespace covise
{
class coDistributedObject;

class ALGEXPORT coFixUsg
{
private:
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  old module parameters: max_vertices_ in boundingBox, delta_ sphere to merge points, opt_mem_ to
    //                          optimize memory usage
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////

    int max_vertices_;
    float delta_;
    bool opt_mem_;

    bool isEqual(float x1, float y1, float z1,
                 float x2, float y2, float z2, float dist);

    void computeCell(int *replBy, const float *xcoord, const float *ycoord, const float *zcoord,
                     const int *coordInBox, int numCoordInBox,
                     float bbx1, float bby1, float bbz1,
                     float bbx2, float bby2, float bbz2,
                     bool optimize, float maxDistanceSqr, int maxCoord,
                     int recurseLevel = 0);

    void boundingBox(const float *const *x, const float *const *y, const float *const *z, const int *c, int n,
                     float *bbx1, float *bby1, float *bbz1,
                     float *bbx2, float *bby2, float *bbz2);

    int getOctant(float x, float y, float z, float ox, float oy, float oz);
    void getOctantBounds(int o, float ox, float oy, float oz,
                         float bbx1, float bby1, float bbz1,
                         float bbx2, float bby2, float bbz2,
                         float *bx1, float *by1, float *bz1,
                         float *bx2, float *by2, float *bz2);

    // for replace list
    enum
    {
        UNTOUCHED = -2,
        REMOVE = -1
    };
    enum
    {
        UNCHANGED = -1
    };

    void computeWorkingLists(int num_coord, int *replyBy, int **src2fil, int **fil2src, int &num_target);

public:
    coFixUsg();
    coFixUsg(int max_vertices, float delta, bool opt_mem = false);

    enum
    {
        FIX_ERROR = -1
    };

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Remove unused points
    //        @param  geo_in: UNSGRD, LINES, POLYGN
    //	  @param  geoObjName: Object name for outgoing geometry object
    //        @param  numVal: number of data values
    //        @param  data_in: list of USTSDT or USTVDT objects
    //        @param  objName: list of object names for outgoing data objects
    //
    //        @param  geo_out  : simplified geo_in
    //        @param  data_out : simplified list data_in
    //        @return : number of reduced points, coFixUsg::FIX_ERROR in case of an error
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    int fixUsg(const coDistributedObject *geo_in, coDistributedObject **geo_out, const char *geoObjName, int num_val,
               const coDistributedObject *const *data_in, coDistributedObject **data_out, const char **objName);

    ////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  As above but with arrays instead of DO's
    //
    ////////////////////////////////////////////////////////////////////////////////////////////////////
    int fixUsg(int num_elem, int num_conn, int num_point,
               const int *elem_list, const int *conn_list, const int *type_list,
               const float *xcoord, const float *ycoord, const float *zcoord,
               int &new_num_elem, int &new_num_conn, int &new_num_coord,
               int **new_elem_list, int **new_conn_list, int **new_type_list,
               float **new_xcoord, float **new_ycoord, float **new_zcoord,
               int num_val, int *numComp, int *dataSize,
               float **in_x, float **in_y, float **in_z,
               float ***out_x, float ***out_y, float ***out_z);

    //
    // fill target array by: target[i] = src[ filtered2source[i] ]
    //
    template <class T>
    void mapArray(const T *src, T *target, const int *filtered2source, int num_target);

    //
    // update target array by: target[i] = source2filtered[ src[i] ] ]
    //
    template <class T>
    void updateArray(const T *src, T *target, const int *source2filtered, int num_target);
};
}

#endif
