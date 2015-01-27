/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS BBoxes
//
//  This class keeps geometry and displacement info used when tiling.
//  In a BBox we keep minimum and maximum values for 2 of the
//  3 coordinates in the system. These 2 coordinates are dictated
//  by the tiling plane
//
//  Initial version: 2002-05-?? Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _TRANSFORM_BBOXES_H_
#define _TRANSFORM_BBOXES_H_

#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
#include <util/coviseCompat.h>
using namespace covise;

class BBoxes
{
public:
    typedef float BBox[12];
    // UXA, UYA, UZA are used to refer to the displacements in the lower corner
    // UXB, UYB, UZB are used to refer to the displacements in the upper corner
    enum
    {
        MIN_X = 0,
        MAX_X = 1,
        MIN_Y = 2,
        MAX_Y = 3,
        MIN_Z = 4,
        MAX_Z = 5,
        UXA = 6,
        UYA = 7,
        UZA = 8,
        UXB = 9,
        UYB = 10,
        UZB = 11
    };
    /// delete and keep proper values for tiling plane
    void clean(int plane);
    /// this function is no longer maintained
    friend ostream &operator<<(ostream &outfile, const BBoxes &tree);
    /// make no_times BBoxes
    void prepare(int no_times);
    /// get the BBox for the time at issue
    const float *getBBox(int time) const;
    /// fill the BBox for the time at issue
    int FillBBox(const coDistributedObject *inObj, const coDistributedObject *inShift, int time);
    /// get number of steps
    int getNoTimes() const
    {
        return no_times_;
    }
    /// constructor
    BBoxes();
    /// destructor
    ~BBoxes();

private:
    /// fill the BBox for a given time using geometry and displacements
    void FillUBox(const float *x_c, const float *y_c, const float *z_c,
                  const float *u_x, const float *u_y, const float *u_z,
                  int no_points, int time);
    /// fill the BBox for a given time using only the geometry
    void FillMBox(const float *x_c, const float *y_c, const float *z_c,
                  int no_points, int time);
    /// fill the BBox for non-unstructured grids
    int FillStrBBox(const coDistributedObject *inObj,
                    const coDoVec3 *shiftObj,
                    int time);
    /// fill the BBox for unstructured grids
    int FillUnstrBBox(const coDistributedObject *inObj,
                      const coDoVec3 *shiftObj,
                      int time);
    int no_times_; // number of times
    float orientation_[3]; // complementary tiling plane orientation
    BBox *bboxes_; // array of BBoxes
};
#endif
