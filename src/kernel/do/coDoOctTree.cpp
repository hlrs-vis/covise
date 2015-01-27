/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoOctTree.h"

using namespace covise;

coDoOctTree::coDoOctTree(const coObjInfo &info)
    : coDoBasisTree(info)
{
    setType("OCTREE", "OCT-TREE");
    if (info.getName())
    {
        if (getShmArray() != 0)
        {
            if (rebuildFromShm() == 0)
            {
                print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
            }
        }
        else
        {
            print_comment(__LINE__, __FILE__, "object %s doesn't exist", info.getName());
            new_ok = 0;
        }
    }
}

int
coDoOctTree::IsInBBox(int cell,
                      int no_e,
                      const float *point) const
{
    if (cell >= 0 && cell < no_e)
    {
        float *cb = (float *)cellBBoxes.getDataPtr();
        //      float *cb = cellBBoxes_.getArray();
        cb += 6 * cell;
        if (point[0] < cb[0] || point[0] > cb[3])
            return 0;
        if (point[1] < cb[1] || point[1] > cb[4])
            return 0;
        if (point[2] < cb[2] || point[2] > cb[5])
            return 0;
        return 1;
    }
    return 0;
}

coDoOctTree::~coDoOctTree()
{
    // no treacherous arrays to delete
}

// const float coDoOctTree::CELL_FACTOR=5.0;

coDistributedObject *
coDoOctTree::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;
    ret = new coDoOctTree(coObjInfo(), arr);
    return ret;
}

coDoOctTree::coDoOctTree(const coObjInfo &info, coShmArray *arr)
    : coDoBasisTree(info, "OCTREE")
{
    // setType("OCTREE","OCT-TREE");
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
    memcpy(grid_bbox_, (float *)gridBBox.getDataPtr(), 6 * sizeof(float));
    fX_ = fXShm;
    fY_ = fYShm;
    fZ_ = fZShm;
}

coDoOctTree::coDoOctTree(const coObjInfo &info,
                         int nelem_l,
                         int nconn_l,
                         int ncoord_l,
                         int *el,
                         int *conn,
                         float *x_c,
                         float *y_c,
                         float *z_c,
                         int normal_size,
                         int max_no_levels,
                         int min_small_enough,
                         int crit_level,
                         int limit_fX,
                         int limit_fY,
                         int limit_fZ)
    : coDoBasisTree(info, "OCTREE", "OCT-TREE", nelem_l, nconn_l, ncoord_l,
                    el, conn, x_c, y_c, z_c, normal_size, max_no_levels, min_small_enough,
                    crit_level, limit_fX, limit_fY, limit_fZ)
{
}

coDoOctTree *coDoOctTree::cloneObject(const coObjInfo &newinfo) const
{
    return new coDoOctTree(newinfo, nelem, nconn, ncoord,
                           el_, conn_, x_c_, y_c_, z_c_,
                           normal_size_,
                           max_no_levels_,
                           min_small_enough_,
                           crit_level_,
                           limit_fX_,
                           limit_fY_,
                           limit_fZ_);
}
