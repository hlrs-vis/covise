/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoOctTreeP.h"

using namespace covise;

coDoOctTreeP::coDoOctTreeP(const coObjInfo &info)
    : coDoBasisTree(info)
{
    setType("OCTREP", "POL-TREE");
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

coDistributedObject *
coDoOctTreeP::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;
    ret = new coDoOctTreeP(coObjInfo(), arr);
    return ret;
}

coDoOctTreeP::coDoOctTreeP(const coObjInfo &info, coShmArray *arr)
    : coDoBasisTree(info, "OCTREP")
{
    // setType("OCTREE","OCT-TREE");
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
    memcpy(grid_bbox_, gridBBox.getDataPtr(), 6 * sizeof(float));
    fX_ = fXShm;
    fY_ = fYShm;
    fZ_ = fZShm;
}

coDoOctTreeP::coDoOctTreeP(const coObjInfo &info,
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
    : coDoBasisTree(info, "OCTREP", "POL-TREE", nelem_l, nconn_l, ncoord_l,
                    el, conn, x_c, y_c, z_c, normal_size, max_no_levels, min_small_enough,
                    crit_level, limit_fX, limit_fY, limit_fZ)
{
}

coDoOctTreeP::~coDoOctTreeP()
{
    delete[] populations_;
}

coDoOctTreeP *coDoOctTreeP::cloneObject(const coObjInfo &newinfo) const
{
    return new coDoOctTreeP(newinfo, nelem, nconn, ncoord,
                            el_, conn_, x_c_, y_c_, z_c_,
                            normal_size_,
                            max_no_levels_,
                            min_small_enough_,
                            crit_level_,
                            limit_fX_,
                            limit_fY_,
                            limit_fZ_);
}

void
coDoOctTreeP::LoadCellPopulations(std::vector<int> **oct_trees_cells, const float *point,
                                  int search_level)
{
    int no_octs = (2 * search_level + 1) * (2 * search_level + 1) * (2 * search_level + 1);
    int count_oct;
    for (count_oct = 0; count_oct < no_octs; ++count_oct)
    {
        oct_trees_cells[count_oct] = NULL;
    }
    memcpy(grid_bbox_, (float *)gridBBox.getDataPtr(), 6 * sizeof(float));
    fX_ = fXShm;
    fY_ = fYShm;
    fZ_ = fZShm;
    // then find the initial oct-tree
    float i_x_grid_l = 1.0f / (grid_bbox_[3] - grid_bbox_[0]);
    float i_y_grid_l = 1.0f / (grid_bbox_[4] - grid_bbox_[1]);
    float i_z_grid_l = 1.0f / (grid_bbox_[5] - grid_bbox_[2]);
    int key[3];

    // suppress floors
    key[0] = (int)((point[0] - grid_bbox_[0]) * i_x_grid_l * fX_);
    if (key[0] >= fX_)
        key[0] = fX_ - 1;
    if (key[0] < 0)
        key[0] = 0;

    key[1] = (int)((point[1] - grid_bbox_[1]) * i_y_grid_l * fY_);
    if (key[1] >= fY_)
        key[1] = fY_ - 1;
    if (key[1] < 0)
        key[1] = 0;

    key[2] = (int)((point[2] - grid_bbox_[2]) * i_z_grid_l * fZ_);
    if (key[2] >= fZ_)
        key[2] = fZ_ - 1;
    if (key[2] < 0)
        key[2] = 0;

    int allKeys[3];
    int i, j, k;
    int order = 0;
    for (i = -search_level; i <= search_level; ++i)
    {
        allKeys[0] = key[0] + i;
        if (allKeys[0] < 0 || allKeys[0] >= fX_)
        {
            continue;
        }
        for (j = -search_level; j <= search_level; ++j)
        {
            allKeys[1] = key[1] + j;
            if (allKeys[1] < 0 || allKeys[1] >= fY_)
            {
                continue;
            }
            for (k = -search_level; k <= search_level; ++k)
            {
                allKeys[2] = key[2] + k;
                if (allKeys[2] < 0 || allKeys[2] >= fZ_)
                {
                    continue;
                }
                oct_trees_cells[order] = &(populations_[Position(allKeys)]);
                ++order;
            }
        }
    }
}

// call this before interpolateField
// ... for instance in coDoPolygons::GetOctTree
void
coDoOctTreeP::setInfo(int nelem_l,
                      int nconn_l,
                      int ncoord_l,
                      int *el,
                      int *conn,
                      float *x_c,
                      float *y_c,
                      float *z_c)
{
    nelem = nelem_l;
    nconn = nconn_l;
    ncoord = ncoord_l;
    el_ = el;
    conn_ = conn;
    x_c_ = x_c;
    y_c_ = y_c;
    z_c_ = z_c;

    if (populations_)
    {
        return;
    }

    int position;
    int no_p_leaves = fX_ * fY_ * fZ_;
    populations_ = new std::vector<int>[no_p_leaves];
    // we want to reserve space for these infinite arrays...
    int ini_room_per_tree = nelem / (no_p_leaves * 4) + 1;
    int octtreeInd;
    for (octtreeInd = 0; octtreeInd < no_p_leaves; ++octtreeInd)
    {
        populations_[octtreeInd].reserve(ini_room_per_tree);
    }
    int cell;
    float i_x_grid_l = 1.0f / (grid_bbox_[3] - grid_bbox_[0]);
    float i_y_grid_l = 1.0f / (grid_bbox_[4] - grid_bbox_[1]);
    float i_z_grid_l = 1.0f / (grid_bbox_[5] - grid_bbox_[2]);
    for (cell = 0; cell < nelem; ++cell)
    {
        int key[6];
        int base = 6 * cell;

        // code with factor_? bits per coordinate
        // suppress floor
        key[0] = (int)((cellBBoxes[base + 0] - grid_bbox_[0]) * i_x_grid_l * fX_);
        if (key[0] >= fX_)
            key[0] = fX_ - 1;
        if (key[0] < 0)
            key[0] = 0;

        key[1] = (int)((cellBBoxes[base + 1] - grid_bbox_[1]) * i_y_grid_l * fY_);
        if (key[1] >= fY_)
            key[1] = fY_ - 1;
        if (key[1] < 0)
            key[1] = 0;

        key[2] = (int)((cellBBoxes[base + 2] - grid_bbox_[2]) * i_z_grid_l * fZ_);
        if (key[2] >= fZ_)
            key[2] = fZ_ - 1;
        if (key[2] < 0)
            key[2] = 0;

        key[3] = (int)((cellBBoxes[base + 3] - grid_bbox_[0]) * i_x_grid_l * fX_);
        if (key[3] >= fX_)
            key[3] = fX_ - 1;
        if (key[3] < 0)
            key[3] = 0;

        key[4] = (int)((cellBBoxes[base + 4] - grid_bbox_[1]) * i_y_grid_l * fY_);
        if (key[4] >= fY_)
            key[4] = fY_ - 1;
        if (key[4] < 0)
            key[4] = 0;

        key[5] = (int)((cellBBoxes[base + 5] - grid_bbox_[2]) * i_z_grid_l * fZ_);
        if (key[5] >= fZ_)
            key[5] = fZ_ - 1;
        if (key[5] < 0)
            key[5] = 0;

        if (key[0] == key[3] && key[1] == key[4] && key[2] == key[5])
        {
            position = Position(key);
            populations_[position].push_back(cell);
        }
        else
        {
            int sweep_key[3];
            sweep_key[0] = key[0];
            sweep_key[1] = key[1];
            sweep_key[2] = key[2];
            for (sweep_key[0] = key[0]; sweep_key[0] <= key[3]; ++sweep_key[0])
            {
                for (sweep_key[1] = key[1]; sweep_key[1] <= key[4]; ++sweep_key[1])
                {
                    for (sweep_key[2] = key[2]; sweep_key[2] <= key[5]; ++sweep_key[2])
                    {
                        position = Position(sweep_key);
                        populations_[position].push_back(cell);
                    }
                }
            }
        }
    }
}
