/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoBasisTree.h"
#include <util/coVector.h>
#include <vector>
#include <cmath>
using namespace covise;

coDoBasisTree::coDoBasisTree(const coObjInfo &info, const char *label1, const char *label2,
                             int nelem_l, int nconn_l, int ncoord_l,
                             int *el, int *conn, float *x_c, float *y_c, float *z_c,
                             int normal_size,
                             int max_no_levels,
                             int min_small_enough,
                             int crit_level,
                             int limit_fX,
                             int limit_fY,
                             int limit_fZ)
    : coDistributedObject(info)
    , populations_(NULL)
    , normal_size_(normal_size)
    , max_no_levels_(max_no_levels)
    , min_small_enough_(min_small_enough)
    , crit_level_(crit_level)
    , limit_fX_(limit_fX)
    , limit_fY_(limit_fY)
    , limit_fZ_(limit_fZ)
{
    nelem = nelem_l;
    nconn = nconn_l;
    ncoord = ncoord_l;

    cellBBoxes_.resize(6 * nelem_l);
    MakeOctTree(el, conn, x_c, y_c, z_c);
    RecreateShm(label1, label2);
}

coDoBasisTree::coDoBasisTree(const coObjInfo &info)
    : coDistributedObject(info)
    , populations_(NULL)
    , normal_size_(0)
    , max_no_levels_(0)
    , min_small_enough_(0)
    , crit_level_(0)
    , limit_fX_(0)
    , limit_fY_(0)
    , limit_fZ_(0)
{
}

coDoBasisTree::coDoBasisTree(const coObjInfo &info,
                             const char *label1, const char *label2,
                             int cellListSize,
                             int macroCellListSize,
                             int cellBBoxesSize,
                             int gridBBoxSize)
    : coDistributedObject(info)
    , populations_(NULL)
    , normal_size_(0)
    , max_no_levels_(0)
    , min_small_enough_(0)
    , crit_level_(0)
    , limit_fX_(0)
    , limit_fY_(0)
    , limit_fZ_(0)
{
    cellList.set_length(cellListSize);
    macroCellList.set_length(macroCellListSize);
    cellBBoxes.set_length(cellBBoxesSize);
    gridBBox.set_length(gridBBoxSize);

    RecreateShm(label1, label2);
}

void
coDoBasisTree::RecreateShm(const char *label1, const char *label2)
{
    covise_data_list dl[SHM_OBJ];
    setType(label1, label2);

    RecreateShmDL(dl);
}

coDoBasisTree::coDoBasisTree(const coObjInfo &info, const char *label)
    : coDistributedObject(info, label)
    , populations_(NULL)
    , normal_size_(0)
    , max_no_levels_(0)
    , min_small_enough_(0)
    , crit_level_(0)
    , limit_fX_(0)
    , limit_fY_(0)
    , limit_fZ_(0)
{
}

coDoBasisTree::~coDoBasisTree()
{
}

int
coDoBasisTree::IsInMacroCell(const int *okey,
                             const int *macro_keys)
{
    if (okey[3] < macro_keys[0] || okey[0] > macro_keys[0] || okey[4] < macro_keys[1] || okey[1] > macro_keys[1] || okey[5] < macro_keys[2] || okey[2] > macro_keys[2])
    {
        return 0;
    }
    return 1;
}

// OctTree creation prior to shared memory access
void
coDoBasisTree::MakeOctTree(int *el,
                           int *conn,
                           float *x_c,
                           float *y_c,
                           float *z_c)
{
    // keep private pointers
    el_ = el;
    conn_ = conn;
    x_c_ = x_c;
    y_c_ = y_c;
    z_c_ = z_c;
    // calculate cell bboxes and grid bbox
    float *cell_bboxes = &cellBBoxes_[0];
    int i;
    // initialise grid box
    grid_bbox_[0] = FLT_MAX;
    grid_bbox_[1] = FLT_MAX;
    grid_bbox_[2] = FLT_MAX;
    grid_bbox_[3] = -FLT_MAX;
    grid_bbox_[4] = -FLT_MAX;
    grid_bbox_[5] = -FLT_MAX;
    for (i = 0, cell_bbox_ = cell_bboxes; i < nelem; ++i, cell_bbox_ += 6)
    {
        // for each element calculate its BBox...
        // ...and modify the grid BBox if necessary
        BBoxForElement(i);
    }
    // check grid_bbox_ to prevent division by 0
    float dimX = grid_bbox_[3] - grid_bbox_[0];
    float dimY = grid_bbox_[4] - grid_bbox_[1];
    float dimZ = grid_bbox_[5] - grid_bbox_[2];
    if (dimX == 0.0 || dimY == 0.0 || dimZ == 0.0)
    {
        float dimM = dimX;
        if (dimM < dimY)
        {
            dimM = dimY;
        }
        if (dimM < dimZ)
        {
            dimM = dimZ;
        }
        if (dimM == 0.0)
        {
            grid_bbox_[0] = -1.0e-3f;
            grid_bbox_[1] = -1.0e-3f;
            grid_bbox_[2] = -1.0e-3f;
            grid_bbox_[3] = 1.0e-3f;
            grid_bbox_[4] = 1.0e-3f;
            grid_bbox_[5] = 1.0e-3f;
        }
        else
        {
            grid_bbox_[0] -= dimM / 100.0f;
            grid_bbox_[1] -= dimM / 100.0f;
            grid_bbox_[2] -= dimM / 100.0f;
            grid_bbox_[3] += dimM / 100.0f;
            grid_bbox_[4] += dimM / 100.0f;
            grid_bbox_[5] += dimM / 100.0f;
        }
    }
    // zauberkraft
    cellFactor_ = 0.0;
    small_enough_ = min_small_enough_; // 32 @@@ 12.6

    if (nelem > 1)
    {
        float logmeas = log10((float)nelem);
        if (1 || nelem < 100000)
        {
            cellFactor_ = logmeas * 0.20f;
        }
        else
        {
            cellFactor_ = logmeas * 0.8f + (logmeas - 5.0f) * 10.0f;
        }
        if (logmeas >= 4.0)
        {
            //@@@ 150
            small_enough_ += (int)((logmeas - 4.0f) * 50.0f);
        }
    }

    cell_bbox_ = cell_bboxes;
    // rather than handling cell populations from level 0, we
    // produce a plantation of trees
    DivideUpToLevel();
    // share cell population between leaves and continue division
    ShareCellsBetweenLeaves();
}

// assume cell_bbox_ points to the correct place for the i-th element
void
coDoBasisTree::BBoxForElement(int i)
{
    // load cell bbox with first vertex coordinates
    int first_vertex = el_[i]; //cell list
    int point = conn_[first_vertex]; //vertices array
    int numvert;
    cell_bbox_[0] = x_c_[point];
    cell_bbox_[1] = y_c_[point];
    cell_bbox_[2] = z_c_[point];
    cell_bbox_[3] = x_c_[point];
    cell_bbox_[4] = y_c_[point];
    cell_bbox_[5] = z_c_[point];
    // find out number of vertices for this element
    if (i < nelem - 1)
    {
        numvert = el_[i + 1] - el_[i];
    }
    else
    {
        numvert = nconn - el_[i];
    }
    // check other points > 0
    for (i = 1; i < numvert; ++i)
    {
        point = conn_[first_vertex + i];
        if (cell_bbox_[0] > x_c_[point])
            cell_bbox_[0] = x_c_[point];
        if (cell_bbox_[1] > y_c_[point])
            cell_bbox_[1] = y_c_[point];
        if (cell_bbox_[2] > z_c_[point])
            cell_bbox_[2] = z_c_[point];
        if (cell_bbox_[3] < x_c_[point])
            cell_bbox_[3] = x_c_[point];
        if (cell_bbox_[4] < y_c_[point])
            cell_bbox_[4] = y_c_[point];
        if (cell_bbox_[5] < z_c_[point])
            cell_bbox_[5] = z_c_[point];
    }
    // do not let the bounding box be too thin!!!
    // correct grid bbox
    if (grid_bbox_[0] > cell_bbox_[0])
        grid_bbox_[0] = cell_bbox_[0];
    if (grid_bbox_[1] > cell_bbox_[1])
        grid_bbox_[1] = cell_bbox_[1];
    if (grid_bbox_[2] > cell_bbox_[2])
        grid_bbox_[2] = cell_bbox_[2];
    if (grid_bbox_[3] < cell_bbox_[3])
        grid_bbox_[3] = cell_bbox_[3];
    if (grid_bbox_[4] < cell_bbox_[4])
        grid_bbox_[4] = cell_bbox_[4];
    if (grid_bbox_[5] < cell_bbox_[5])
        grid_bbox_[5] = cell_bbox_[5];
}

// share cell population between leaves and continue division
void
coDoBasisTree::ShareCellsBetweenLeaves()
{
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
        key[0] = (int)((cell_bbox_[base + 0] - grid_bbox_[0]) * i_x_grid_l * fX_);
        if (key[0] >= fX_)
            key[0] = fX_ - 1;
        if (key[0] < 0)
            key[0] = 0;

        key[1] = (int)((cell_bbox_[base + 1] - grid_bbox_[1]) * i_y_grid_l * fY_);
        if (key[1] >= fY_)
            key[1] = fY_ - 1;
        if (key[1] < 0)
            key[1] = 0;

        key[2] = (int)((cell_bbox_[base + 2] - grid_bbox_[2]) * i_z_grid_l * fZ_);
        if (key[2] >= fZ_)
            key[2] = fZ_ - 1;
        if (key[2] < 0)
            key[2] = 0;

        key[3] = (int)((cell_bbox_[base + 3] - grid_bbox_[0]) * i_x_grid_l * fX_);
        if (key[3] >= fX_)
            key[3] = fX_ - 1;
        if (key[3] < 0)
            key[3] = 0;

        key[4] = (int)((cell_bbox_[base + 4] - grid_bbox_[1]) * i_y_grid_l * fY_);
        if (key[4] >= fY_)
            key[4] = fY_ - 1;
        if (key[4] < 0)
            key[4] = 0;

        key[5] = (int)((cell_bbox_[base + 5] - grid_bbox_[2]) * i_z_grid_l * fZ_);
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
    // OK, now create the octtrees; let the trees grow
    int key[3];
    int &i = key[0];
    int &j = key[1];
    int &k = key[2];

    int macro_leaf;
    float bbox[6];
    cellList_.reserve(nelem);
    cellList_.push_back(0); // one dummy element for cellList_
    // make room for the fZ_*fY_*fX_ oct-tree entry points
    macCellList_.reserve(no_p_leaves);
    for (k = 0; k < fZ_; ++k)
    {
        for (j = 0; j < fY_; ++j)
        {
            for (i = 0; i < fX_; ++i)
            {
                macCellList_.push_back(0);
            }
        }
    }
    for (k = 0; k < fZ_; ++k)
    {
        for (j = 0; j < fY_; ++j)
        {
            for (i = 0; i < fX_; ++i)
            {
                macro_leaf = Position(key);
                IniBBox(bbox, key);
                // macCellList_.push_back(0);
                SplitOctTree(bbox, populations_[macro_leaf], 0, macro_leaf);
            }
        }
    }
    delete[] populations_;
    populations_ = NULL;
}

// creates bbox for the root of an oct-tree given its key
void
coDoBasisTree::IniBBox(float *bbox,
                       const int *key) const
{
    bbox[0] = grid_bbox_[0] + (grid_bbox_[3] - grid_bbox_[0]) * key[0] / fX_;
    bbox[1] = grid_bbox_[1] + (grid_bbox_[4] - grid_bbox_[1]) * key[1] / fY_;
    bbox[2] = grid_bbox_[2] + (grid_bbox_[5] - grid_bbox_[2]) * key[2] / fZ_;
    bbox[3] = grid_bbox_[0] + (grid_bbox_[3] - grid_bbox_[0]) * (key[0] + 1) / fX_;
    bbox[4] = grid_bbox_[1] + (grid_bbox_[4] - grid_bbox_[1]) * (key[1] + 1) / fY_;
    bbox[5] = grid_bbox_[2] + (grid_bbox_[5] - grid_bbox_[2]) * (key[2] + 1) / fZ_;
}

// divide oct-tree
void
coDoBasisTree::SplitOctTree(const float *bbox,
                            std::vector<int> &population,
                            int level,
                            int offset)
{
    // no more divisions if the population is small enough or if
    // the maximum supported level has been achieved or if all cells are too big
    int cell;
    if (population.size() <= small_enough_ * pow(2.0, 0.0 * 0.125 * level)
        || level == max_no_levels_
        || CellsAreTooBig(bbox, population))
    {
        // negative of the absolute position in cellList_
        if (population.size() > 0)
        {
            macCellList_[offset] = -((int)cellList_.size());
            // dump population
            cellList_.push_back((int)population.size());
            for (cell = 0; cell < population.size(); ++cell)
            {
                cellList_.push_back(population[cell]);
            }
        }
        else
        {
            macCellList_[offset] = 0;
        }
        population.clear();
        return;
    }
    // determine how the cell population of this macrocell
    // would be shared by its 8 sons
    std::vector<int> popu_sons[8];
    int macro_cell;
    for (macro_cell = 0; macro_cell < 8; ++macro_cell)
    {
        popu_sons[macro_cell].reserve(population.size() / 16);
    }
    // new method
    const float &mx_min = bbox[0];
    const float &mx_max = bbox[3];
    const float &my_min = bbox[1];
    const float &my_max = bbox[4];
    const float &mz_min = bbox[2];
    const float &mz_max = bbox[5];
    float mx_med = 0.5f * (mx_min + mx_max);
    float my_med = 0.5f * (my_min + my_max);
    float mz_med = 0.5f * (mz_min + mz_max);
    for (cell = 0; cell < population.size(); ++cell)
    {
        float *this_c_bbox = &cell_bbox_[6 * population[cell]];
        bool X_m_d = !(*(this_c_bbox) > mx_med);
        bool X_m_M = !(*(this_c_bbox) > mx_max);
        bool Y_m_d = !(*(this_c_bbox + 1) > my_med);
        bool Y_m_M = !(*(this_c_bbox + 1) > my_max);
        bool Z_m_d = !(*(this_c_bbox + 2) > mz_med);
        bool Z_m_M = !(*(this_c_bbox + 2) > mz_max);
        bool X_M_d = !(*(this_c_bbox + 3) < mx_med);
        bool X_M_m = !(*(this_c_bbox + 3) < mx_min);
        bool Y_M_d = !(*(this_c_bbox + 4) < my_med);
        bool Y_M_m = !(*(this_c_bbox + 4) < my_min);
        bool Z_M_d = !(*(this_c_bbox + 5) < mz_med);
        bool Z_M_m = !(*(this_c_bbox + 5) < mz_min);

        // tests son 0
        if (X_M_m && X_m_d && Y_M_m && Y_m_d && Z_M_m && Z_m_d)
        {
            popu_sons[0].push_back(population[cell]);
        }
        // tests son 1
        if (X_M_d && X_m_M && Y_M_m && Y_m_d && Z_M_m && Z_m_d)
        {
            popu_sons[1].push_back(population[cell]);
        }
        // tests son 2
        if (X_M_m && X_m_d && Y_M_d && Y_m_M && Z_M_m && Z_m_d)
        {
            popu_sons[2].push_back(population[cell]);
        }
        // tests son 3
        if (X_M_d && X_m_M && Y_M_d && Y_m_M && Z_M_m && Z_m_d)
        {
            popu_sons[3].push_back(population[cell]);
        }
        // tests son 4
        if (X_M_m && X_m_d && Y_M_m && Y_m_d && Z_M_d && Z_m_M)
        {
            popu_sons[4].push_back(population[cell]);
        }
        // tests son 5
        if (X_M_d && X_m_M && Y_M_m && Y_m_d && Z_M_d && Z_m_M)
        {
            popu_sons[5].push_back(population[cell]);
        }
        // tests son 6
        if (X_M_m && X_m_d && Y_M_d && Y_m_M && Z_M_d && Z_m_M)
        {
            popu_sons[6].push_back(population[cell]);
        }
        // tests son 7
        if (X_M_d && X_m_M && Y_M_d && Y_m_M && Z_M_d && Z_m_M)
        {
            popu_sons[7].push_back(population[cell]);
        }
    }
    // which son has the greatest population?
    int son;
    int max_popu = (int)popu_sons[0].size();
    for (son = 1; son < 8; ++son)
    {
        if (max_popu < (int)popu_sons[son].size())
        {
            max_popu = (int)popu_sons[son].size();
        }
    }
    // if we are beyond some critical level and one son inherits the entire
    // cell population, divide no further
    // &&
    if (level >= crit_level_ && max_popu >= population.size())
    {
        // population.size()<NORMAL_SIZE/10){
        // negative of the absolute position in cellList_
        macCellList_[offset] = -((int)cellList_.size());
        // dump population
        cellList_.push_back((int)population.size());
        for (cell = 0; cell < population.size(); ++cell)
        {
            cellList_.push_back(population[cell]);
        }
        population.clear();
        return;
    }

    // else divide the population...
    // we may then release the memory of population.
    population.clear();

    // write in macCellList_ the new offset.
    macCellList_[offset] = (int)macCellList_.size();
    // make room for the 8 sons
    for (son = 0; son < 8; ++son)
    {
        macCellList_.push_back(0);
    }
    // and divide
    for (son = 0; son < 8; ++son)
    {
        float bbox_son[6];
        fillBBoxSon(bbox_son, bbox, son);
        SplitOctTree(bbox_son, popu_sons[son], level + 1, macCellList_[offset] + son);
    }
}

// create bbox for the son of a macrocell
void
coDoBasisTree::fillBBoxSon(float *bbox_son,
                           const float *bbox,
                           int son) const
{
    memcpy(bbox_son, bbox, 6 * sizeof(float));
    switch (son)
    {
    case 0:
        bbox_son[3] -= 0.5f * (bbox[3] - bbox[0]);
        bbox_son[4] -= 0.5f * (bbox[4] - bbox[1]);
        bbox_son[5] -= 0.5f * (bbox[5] - bbox[2]);
        break;
    case 1:
        bbox_son[0] += 0.5f * (bbox[3] - bbox[0]);
        bbox_son[4] -= 0.5f * (bbox[4] - bbox[1]);
        bbox_son[5] -= 0.5f * (bbox[5] - bbox[2]);
        break;
    case 2:
        bbox_son[1] += 0.5f * (bbox[4] - bbox[1]);
        bbox_son[3] -= 0.5f * (bbox[3] - bbox[0]);
        bbox_son[5] -= 0.5f * (bbox[5] - bbox[2]);
        break;
    case 3:
        bbox_son[0] += 0.5f * (bbox[3] - bbox[0]);
        bbox_son[1] += 0.5f * (bbox[4] - bbox[1]);
        bbox_son[5] -= 0.5f * (bbox[5] - bbox[2]);
        break;
    case 4:
        bbox_son[2] += 0.5f * (bbox[5] - bbox[2]);
        bbox_son[3] -= 0.5f * (bbox[3] - bbox[0]);
        bbox_son[4] -= 0.5f * (bbox[4] - bbox[1]);
        break;
    case 5:
        bbox_son[0] += 0.5f * (bbox[3] - bbox[0]);
        bbox_son[2] += 0.5f * (bbox[5] - bbox[2]);
        bbox_son[4] -= 0.5f * (bbox[4] - bbox[1]);
        break;
    case 6:
        bbox_son[1] += 0.5f * (bbox[4] - bbox[1]);
        bbox_son[2] += 0.5f * (bbox[5] - bbox[2]);
        bbox_son[3] -= 0.5f * (bbox[3] - bbox[0]);
        break;
    case 7:
        bbox_son[0] += 0.5f * (bbox[3] - bbox[0]);
        bbox_son[1] += 0.5f * (bbox[4] - bbox[1]);
        bbox_son[2] += 0.5f * (bbox[5] - bbox[2]);
        break;
    default:
        break;
    }
}

// if cells are "quite" big, then stop oct-tree division
int
coDoBasisTree::CellsAreTooBig(const float *bbox, std::vector<int> &population)
{
    // if(cellFactor_<4) return 0;
    // test for every cell if the cell bounding box (kept in cellBBoxes_)
    // is in at least one of the dimensions greater than that of
    // the macro cell bbox
    // int ret=0; assume no
    int count = 0;
    int cell;
    int cell_label;
    for (cell = 0; cell < population.size(); ++cell)
    {
        int mark = 0;
        cell_label = population[cell];
        float *cell_bbox = &cellBBoxes_[0] + 6 * cell_label;
        if (cellFactor_ * (cell_bbox[3] - cell_bbox[0]) < bbox[3] - bbox[0])
            ++mark;
        if (cellFactor_ * (cell_bbox[4] - cell_bbox[1]) < bbox[4] - bbox[1])
            ++mark;
        if (cellFactor_ * (cell_bbox[5] - cell_bbox[2]) < bbox[5] - bbox[2])
            ++mark;
        if (mark < 3) // one cell is greater in at least one dimension
        {
            ++count;
        }
    }
    if (count * 5 >= population.size())
    {
        return 1;
    }
    return 0;
}

// determine initial oct-tree
int
coDoBasisTree::Position(int *key) const
{
    int pos = (key[0] + (key[1] + key[2] * fY_) * fX_);
    if (pos < 0)
    {
        print_error(__LINE__, __FILE__, "bad position: key=(%d %d %d), f=(%d %d %d)\n",
                    key[0], key[1], key[2],
                    fX_, fY_, fZ_);
    }
    return pos;
}

// get a list of candidate elements where point may lie,
// the length of this list is the first element
const int *
coDoBasisTree::search(const float *point) const
{
    memcpy(grid_bbox_, (float *)gridBBox.getDataPtr(), 6 * sizeof(float));
    fX_ = fXShm;
    fY_ = fYShm;
    fZ_ = fZShm;
    max_no_levels_ = max_no_levels_Shm;

    // first test whether the point lies in the grid bbox
    if (point[0] < grid_bbox_[0] || point[0] > grid_bbox_[3])
        return (&cellList[0]);
    if (point[1] < grid_bbox_[1] || point[1] > grid_bbox_[4])
        return (&cellList[0]);
    if (point[2] < grid_bbox_[2] || point[2] > grid_bbox_[5])
        return (&cellList[0]);
    // then find the initial oct-tree
    float i_x_grid_l = 1.0f / (grid_bbox_[3] - grid_bbox_[0]);
    float i_y_grid_l = 1.0f / (grid_bbox_[4] - grid_bbox_[1]);
    float i_z_grid_l = 1.0f / (grid_bbox_[5] - grid_bbox_[2]);
    int key[3];
    // suppress floors
    key[0] = (int)((point[0] - grid_bbox_[0]) * i_x_grid_l * fX_);
    if (key[0] >= fX_)
        key[0] = fX_ - 1;
    key[1] = (int)((point[1] - grid_bbox_[1]) * i_y_grid_l * fY_);
    if (key[1] >= fY_)
        key[1] = fY_ - 1;
    key[2] = (int)((point[2] - grid_bbox_[2]) * i_z_grid_l * fZ_);
    if (key[2] >= fZ_)
        key[2] = fZ_ - 1;
    // and look up in oct-tree...
    // first find the bounding box of the oct-tree
    float bbox[6];
    bbox[0] = grid_bbox_[0] + (grid_bbox_[3] - grid_bbox_[0]) * key[0] / fX_;
    bbox[1] = grid_bbox_[1] + (grid_bbox_[4] - grid_bbox_[1]) * key[1] / fY_;
    bbox[2] = grid_bbox_[2] + (grid_bbox_[5] - grid_bbox_[2]) * key[2] / fZ_;
    bbox[3] = grid_bbox_[0] + (grid_bbox_[3] - grid_bbox_[0]) * (key[0] + 1) / fX_;
    bbox[4] = grid_bbox_[1] + (grid_bbox_[4] - grid_bbox_[1]) * (key[1] + 1) / fY_;
    bbox[5] = grid_bbox_[2] + (grid_bbox_[5] - grid_bbox_[2]) * (key[2] + 1) / fZ_;
    // then for this bbox, and point extract 3 keys with as many bits
    // as the maximum of accepted levels
    int okey[3];
    int treeF = (1 << max_no_levels_);
    int mask = (1 << (max_no_levels_ - 1));
    okey[0] = (int)((point[0] - bbox[0]) / (bbox[3] - bbox[0]) * (treeF));
    okey[1] = (int)((point[1] - bbox[1]) / (bbox[4] - bbox[1]) * (treeF));
    okey[2] = (int)((point[2] - bbox[2]) / (bbox[5] - bbox[2]) * (treeF));
    if (okey[0] == treeF)
    {
        --okey[0];
    }
    if (okey[1] == treeF)
    {
        --okey[1];
    }
    if (okey[2] == treeF)
    {
        --okey[2];
    }
    int position = Position(key);
    return lookUp(position, okey, mask);
}

//extended search. check which octree cells a line between two points hit.
const int *
coDoBasisTree::extended_search(const coVector &point1, const coVector &point2, std::vector<int> &OctreePolygonList) const
{

    memcpy(grid_bbox_, (float *)gridBBox.getDataPtr(), 6 * sizeof(float));
    fX_ = fXShm;
    fY_ = fYShm;
    fZ_ = fZShm;
    max_no_levels_ = max_no_levels_Shm;
    float i_x_grid_l = 1.0f / (grid_bbox_[3] - grid_bbox_[0]);
    float i_y_grid_l = 1.0f / (grid_bbox_[4] - grid_bbox_[1]);
    float i_z_grid_l = 1.0f / (grid_bbox_[5] - grid_bbox_[2]);
    //float point[3];

    coVector CellVector;
    std::vector<coVector> cutVecs;
    coVector CutVec_interim;
    coVector x_cutVector;
    coVector y_cutVector;
    coVector z_cutVector;
    std::vector<coVector> cutVector_temp;
    std::vector<float> XCutList;
    std::vector<float> YCutList;
    std::vector<float> ZCutList;
    std::vector<int> connectList;
    std::vector<int> levelList;

    max_no_levels_ = max_no_levels_Shm;
    //int Punkte[2];
    bool point1_outOfGrid = false;
    bool point2_outOfGrid = false;

    //Schneidet die Gerade die BBox?
    bool didCut = cutLineCuboid(point1, point2, (const float *)grid_bbox_, cutVecs);
    //fprintf(stderr,"size cutVecs = %d\n",(int) cutVecs.size());
    coVector MacroPoint1;
    coVector MacroPoint2;
    MacroPoint1[0] = 0;
    MacroPoint1[1] = 0;
    MacroPoint1[2] = 0;
    MacroPoint2[0] = 0;
    MacroPoint2[1] = 0;
    MacroPoint2[2] = 0;

    //Teste ob Punkte innerhalb der Octree-Bbox liegen:
    if ((point1[0] < grid_bbox_[0] || point1[0] > grid_bbox_[3]) || (point1[1] < grid_bbox_[1] || point1[1] > grid_bbox_[4]) || (point1[2] < grid_bbox_[2] || point1[2] > grid_bbox_[5]))
    {
        point1_outOfGrid = true;
    }
    if ((point2[0] < grid_bbox_[0] || point2[0] > grid_bbox_[3]) || (point2[1] < grid_bbox_[1] || point2[1] > grid_bbox_[4]) || (point2[2] < grid_bbox_[2] || point2[2] > grid_bbox_[5]))
    {
        point2_outOfGrid = true;
    }

    //Fallunterscheidung
    if ((point1_outOfGrid == true) && (point2_outOfGrid == true) && (didCut == false))
    {
        return (&cellList[0]); //Beide Punkte außerhalb und Gerade schneidet nicht! ACHTUNG!!!!!!!!!!!!!!!!!
    }
    else
    {
        if ((point1_outOfGrid == false) && (point2_outOfGrid == false) && (didCut == false))
        {
            //Gerade schneidet Bbox nicht, Punkte liegen aber innherhalb der Bbox

            MacroPoint1 = point1;
            MacroPoint2 = point2;
        }

        if ((point1_outOfGrid == true) && (point2_outOfGrid == false) && (didCut == true))
        {
            //Gerade schneidet Bbox einmal, da Punkt1 außerhalb, aber Punkt2 innerhalb
            MacroPoint1 = point2;
            MacroPoint2 = cutVecs[0];
        }
        if ((point1_outOfGrid == false) && (point2_outOfGrid == true) && (didCut == true))
        {
            //Gerade schneidet Bbox einmal, da Punkt2 außerhalb, aber Punkt1 innerhalb
            MacroPoint1 = point1;
            MacroPoint2 = cutVecs[0];
        }
        if ((point1_outOfGrid == true) && (point2_outOfGrid == true) && (didCut == true))
        {
            //Kein Punkt liegt innerhalb der Bbox, jedoch schneidet die Gerade die Bbox
            MacroPoint1 = cutVecs[0];
            if ((int)cutVecs.size() > 1)
            {
                MacroPoint2 = cutVecs[1];
            }
            else
            {
                fprintf(stderr, "Kein zweiter Schnittpunkt berechnet!\n");
                MacroPoint2 = point2;
            }
        }

        float cutCuboid[6]; //Schleife über alle Makrozellen um herauszufinden welche von der Gerade geschnitten werden
        for (int x_key = 0; x_key < (fX_); x_key++) //fX_,fY_,fZ_ entsprechen der max Zellenanzahl in die jeweilige Richtung
        {
            for (int y_key = 0; y_key < (fY_); y_key++)
            {
                for (int z_key = 0; z_key < (fZ_); z_key++)
                {
                    didCut = false;
                    int xyz_key[3];
                    xyz_key[0] = x_key;
                    xyz_key[1] = y_key;
                    xyz_key[2] = z_key;
                    IniBBox(cutCuboid, xyz_key);

                    bool inBbox = lineInBbox(MacroPoint1, MacroPoint2, cutCuboid);
                    didCut = cutLineCuboid(MacroPoint1, MacroPoint2, cutCuboid, cutVector_temp);
                    if ((didCut == true) || (inBbox == true))
                    {
                        int position = Position(xyz_key);

                        cutThroughOct(position, cutCuboid, levelList, connectList, XCutList, YCutList, ZCutList, point1, point2, OctreePolygonList);
                    }
                }
            }
        }

        cutVector_temp.clear();
    }
    return (&cellList[0]);
}

//function that checks wether a line lies within a bbox
bool
coDoBasisTree::lineInBbox(const coVector &point1, const coVector &point2, const float bbox[6]) const
{
    bool point1_outOfBbox = false;
    bool point2_outOfBbox = false;
    if ((point1[0] < bbox[0] || point1[0] > bbox[3]) || (point1[1] < bbox[1] || point1[1] > bbox[4]) || (point1[2] < bbox[2] || point1[2] > bbox[5]))
    {
        point1_outOfBbox = true;
    }
    if ((point2[0] < bbox[0] || point2[0] > bbox[3]) || (point2[1] < bbox[1] || point2[1] > bbox[4]) || (point2[2] < bbox[2] || point2[2] > bbox[5]))
    {
        point2_outOfBbox = true;
    }
    if ((point1_outOfBbox == false) && (point2_outOfBbox == false))
    {
        return true;
    }
    return false;
}

//function, that gives back all cells which lay between two points
bool
coDoBasisTree::cutThroughOct(int baum, const float bbox[6], std::vector<int> &ll, std::vector<int> &cl, std::vector<float> &xi, std::vector<float> &yi, std::vector<float> &zi,
                             const coVector &point1, const coVector &point2, std::vector<int> &OctreePolygonList) const
{

    if (macroCellList[baum] == 0)
        return false;
    if (macroCellList[baum] < 0)
    {
        if (cellList[-macroCellList[baum]] == 0)
            return false;
        // a leaf with population

        for (int c = 1; c < (int)cellList[-macroCellList[baum]] + 1; c++)
        {
            OctreePolygonList.push_back(cellList[(-macroCellList[baum]) + c]);
        }
    }
    else
    {
        // >0 -> not a leaf
        int son;
        for (son = 0; son < 8; ++son)
        {
            std::vector<coVector> cutVecs;
            float bbox_son[6];
            fillBBoxSon(bbox_son, bbox, son);
            if (((cutLineCuboid(point1, point2, bbox_son, cutVecs)) == true) || (lineInBbox(point1, point2, bbox_son)))
            {
                cutThroughOct(macroCellList[baum] + son, bbox_son, ll, cl, xi, yi, zi, point1, point2, OctreePolygonList);
            }
        }
    }
    return true;
}

//function for cutting a line segment with a cuboid bounding box
bool
coDoBasisTree::cutLineCuboid(const coVector &point1, const coVector &point2, const float *cuboidBbox, std::vector<coVector> &CutVec) const
{

    coVector GeradenStuetz;
    coVector GeradenRicht;
    ///DEKLARATION GAUSS ELIMINATION
    coVector GridStuetz, GridRicht;
    coVector PolyStuetz, PolyRicht1, PolyRicht2;
    ///----------
    CutVec.clear();

    //GeradenStuetz = point1;
    //GeradenRicht = point2 - point1;
    //coVector EbenenStuetz;

    coVector punktArray[8];
    punktArray[0][0] = cuboidBbox[3]; //xmax	//Punkt1
    punktArray[0][1] = cuboidBbox[1]; //ymin
    punktArray[0][2] = cuboidBbox[5]; //zmax
    punktArray[1][0] = cuboidBbox[0]; //xmin	//Punkt2
    punktArray[1][1] = cuboidBbox[1]; //ymin
    punktArray[1][2] = cuboidBbox[5]; //zmax
    punktArray[2][0] = cuboidBbox[0]; //xmin	//Punkt3
    punktArray[2][1] = cuboidBbox[4]; //ymax
    punktArray[2][2] = cuboidBbox[5]; //zmax
    punktArray[3][0] = cuboidBbox[3]; //xmax	//Punkt4
    punktArray[3][1] = cuboidBbox[4]; //ymax
    punktArray[3][2] = cuboidBbox[5]; //zmax
    punktArray[4][0] = cuboidBbox[3]; //xmax	//Punkt5
    punktArray[4][1] = cuboidBbox[1]; //ymin
    punktArray[4][2] = cuboidBbox[2]; //zmin
    punktArray[5][0] = cuboidBbox[0]; //xmin	//Punkt6
    punktArray[5][1] = cuboidBbox[1]; //ymin
    punktArray[5][2] = cuboidBbox[2]; //zmin
    punktArray[6][0] = cuboidBbox[0]; //xmin	//Punkt7
    punktArray[6][1] = cuboidBbox[4]; //ymax
    punktArray[6][2] = cuboidBbox[2]; //zmin
    punktArray[7][0] = cuboidBbox[3]; //xmax	//Punkt8
    punktArray[7][1] = cuboidBbox[4]; //ymax
    punktArray[7][2] = cuboidBbox[2]; //zmin

    coVector richtVek_array[6];

    richtVek_array[0] = punktArray[1] - punktArray[0];
    richtVek_array[1] = punktArray[3] - punktArray[0];
    richtVek_array[2] = punktArray[4] - punktArray[0];
    richtVek_array[3] = punktArray[7] - punktArray[6];
    richtVek_array[4] = punktArray[5] - punktArray[6];
    richtVek_array[5] = punktArray[2] - punktArray[6];

    //Schneide Gerade mit Bounding-Box
    for (int r = 0; r < 8; r = r + 6)
    {
        //EbenenStuetz= punktArray[r];
        int offset;
        if (r < 5)
        {
            offset = 0;
        }
        if (r > 5)
        {
            offset = 3;
        }

        for (int i = 0; i < 3; i++)
        {
            int a;
            coVector NormVek;
            //float NormVekBetrag;
            //float lambdaZaehler;
            //float lambdaNenner;
            //float lambda;
            double Lambda, Lambda1, Lambda2;
            if (i < 2)
            {
                a = i + 1;
            }
            else
            {
                a = i - 2;
            }

            ///BERECHNUNG DURCH GAUSS ELIMINATIONSVERFAHREN
            GridStuetz = point1;
            GridRicht = point2 - point1;
            PolyStuetz = punktArray[r];
            PolyRicht1 = richtVek_array[offset + i];
            PolyRicht2 = richtVek_array[offset + a];

            const int n = 3;
            double A[n][n]; //Matrix des LGS
            //float L[n][n];	//Untere linke Dreiecksmatrix
            //float R[n][n];	//Obere rechte Dreiecksmatrix
            double y[n]; //Hilfsvektor
            double b[n]; //Vektor der Eingangsgroessen des LGS ... rechte Seite
            //double x, sum;
            bool weg1 = false;
            bool weg2 = false;
            bool weg3 = false;
            bool weg4 = false;
            bool weg5 = false;
            bool weg6 = false;

            bool lambdaausGl1 = true;
            bool lambdaausGl2 = true;
            bool lambdaausGl3 = true;
            if ((GridRicht[0] > -0.000001) && (GridRicht[0] < 0.000001)) //Wenn GridRicht in x-Ri Null, kann lambda nicht aus 1.Gl bestimmt werden!
            {
                lambdaausGl1 = false;
            }
            if ((GridRicht[1] > -0.000001) && (GridRicht[1] < 0.000001)) // Wenn GridRicht iny-Ri Null, kann lambda nicht aus 2.Gl bestimmt werden!
            {
                lambdaausGl2 = false;
            }
            if ((GridRicht[2] > -0.000001) && (GridRicht[2] < 0.000001)) //Wenn GridRicht...
            {
                lambdaausGl3 = false;
            }

            bool lambda1ausGl1 = true;
            bool lambda1ausGl2 = true;
            bool lambda1ausGl3 = true;
            if ((PolyRicht1[0] > -0.000001) && (PolyRicht1[0] < 0.000001)) //Wenn PolyRicht1 in x-Ri Null, kann lambda nicht aus 1.Gl bestimmt werden!
            {
                lambda1ausGl1 = false;
            }
            if ((PolyRicht1[1] > -0.000001) && (PolyRicht1[1] < 0.000001)) // Wenn PolyRicht1 in y-Ri Null, kann lambda nicht aus 2.Gl bestimmt werden!
            {
                lambda1ausGl2 = false;
            }
            if ((PolyRicht1[2] > -0.000001) && (PolyRicht1[2] < 0.000001)) //Wenn PolyRicht1...
            {
                lambda1ausGl3 = false;
            }

            bool lambda2ausGl1 = true;
            bool lambda2ausGl2 = true;
            bool lambda2ausGl3 = true;
            if ((PolyRicht2[0] > -0.000001) && (PolyRicht2[0] < 0.000001)) //Wenn PolyRicht2 in x-Ri Null, kann lambda nicht aus 1.Gl bestimmt werden!
            {
                lambda2ausGl1 = false;
            }
            if ((PolyRicht2[1] > -0.000001) && (PolyRicht2[1] < 0.000001)) // Wenn PolyRicht2 in y-Ri Null, kann lambda nicht aus 2.Gl bestimmt werden!
            {
                lambda2ausGl2 = false;
            }
            if ((PolyRicht2[2] > -0.000001) && (PolyRicht2[2] < 0.000001)) //Wenn PolyRicht2...
            {
                lambda2ausGl3 = false;
            }
            //3 Ausgangsvarianten zur Bestimmung der Lambdas. Aus Gleichung 1, 2 oder 3.
            if ((lambdaausGl1 == true) && (((lambda1ausGl2 == true) && (lambda2ausGl3 == true)) || ((lambda1ausGl3 == true) && (lambda2ausGl2 == true)))) //Variante1
            {
                A[0][0] = GridRicht[0];
                A[0][1] = -PolyRicht1[0];
                A[0][2] = -PolyRicht2[0];
                b[0] = PolyStuetz[0] - GridStuetz[0];

                if ((lambda1ausGl2 == true) && (lambda2ausGl3 == true))
                {
                    A[1][0] = GridRicht[1];
                    A[1][1] = -PolyRicht1[1];
                    A[1][2] = -PolyRicht2[1];
                    b[1] = PolyStuetz[1] - GridStuetz[1];
                    A[2][0] = GridRicht[2];
                    A[2][1] = -PolyRicht1[2];
                    A[2][2] = -PolyRicht2[2];
                    b[2] = PolyStuetz[2] - GridStuetz[2];
                    weg1 = true;
                }
                else
                {
                    if ((lambda1ausGl3 == true) && (lambda2ausGl2 == true))
                    {
                        A[1][0] = GridRicht[2];
                        A[1][1] = -PolyRicht1[2];
                        A[1][2] = -PolyRicht2[2];
                        b[1] = PolyStuetz[2] - GridStuetz[2];
                        A[2][0] = GridRicht[1];
                        A[2][1] = -PolyRicht1[1];
                        A[2][2] = -PolyRicht2[1];
                        b[2] = PolyStuetz[1] - GridStuetz[1];
                        weg2 = true;
                    }
                    else
                    {
                        fprintf(stderr, "This is impossible!");
                    }
                }
            }
            else
            {
                if ((lambdaausGl2 == true) && (((lambda1ausGl1 == true) && (lambda2ausGl3 == true)) || ((lambda1ausGl3 == true) && (lambda2ausGl1 == true)))) //Variante2
                {
                    A[0][0] = GridRicht[1];
                    A[0][1] = -PolyRicht1[1];
                    A[0][2] = -PolyRicht2[1];
                    b[0] = PolyStuetz[1] - GridStuetz[1];

                    if ((lambda1ausGl1 == true) && (lambda2ausGl3 == true))
                    {
                        A[1][0] = GridRicht[0];
                        A[1][1] = -PolyRicht1[0];
                        A[1][2] = -PolyRicht2[0];
                        b[1] = PolyStuetz[0] - GridStuetz[0];
                        A[2][0] = GridRicht[2];
                        A[2][1] = -PolyRicht1[2];
                        A[2][2] = -PolyRicht2[2];
                        b[2] = PolyStuetz[2] - GridStuetz[2];
                        weg3 = true;
                    }
                    else
                    {
                        if ((lambda1ausGl3 == true) && (lambda2ausGl1 == true))
                        {
                            A[1][0] = GridRicht[2];
                            A[1][1] = -PolyRicht1[2];
                            A[1][2] = -PolyRicht2[2];
                            b[1] = PolyStuetz[2] - GridStuetz[2];
                            A[2][0] = GridRicht[0];
                            A[2][1] = -PolyRicht1[0];
                            A[2][2] = -PolyRicht2[0];
                            b[2] = PolyStuetz[0] - GridStuetz[0];
                            weg4 = true;
                        }
                        else
                        {
                            fprintf(stderr, "This is impossible!");
                        }
                    }
                }
                else
                {
                    if ((lambdaausGl3 == true) && (((lambda1ausGl1 == true) && (lambda2ausGl2 == true)) || ((lambda1ausGl2 == true) && (lambda2ausGl1 == true)))) //Variante3
                    {
                        A[0][0] = GridRicht[2];
                        A[0][1] = -PolyRicht1[2];
                        A[0][2] = -PolyRicht2[2];
                        b[0] = PolyStuetz[2] - GridStuetz[2];

                        if ((lambda1ausGl1 == true) && (lambda2ausGl2 == true))
                        {
                            A[1][0] = GridRicht[0];
                            A[1][1] = -PolyRicht1[0];
                            A[1][2] = -PolyRicht2[0];
                            b[1] = PolyStuetz[0] - GridStuetz[0];
                            A[2][0] = GridRicht[1];
                            A[2][1] = -PolyRicht1[1];
                            A[2][2] = -PolyRicht2[1];
                            b[2] = PolyStuetz[1] - GridStuetz[1];
                            weg5 = true;
                        }
                        else
                        {
                            if ((lambda1ausGl2 == true) && (lambda2ausGl1 == true))
                            {
                                A[1][0] = GridRicht[1];
                                A[1][1] = -PolyRicht1[1];
                                A[1][2] = -PolyRicht2[1];
                                b[1] = PolyStuetz[1] - GridStuetz[1];
                                A[2][0] = GridRicht[0];
                                A[2][1] = -PolyRicht1[0];
                                A[2][2] = -PolyRicht2[0];
                                b[2] = PolyStuetz[0] - GridStuetz[0];
                                weg6 = true;
                            }
                            else
                            {
                                fprintf(stderr, "This is impossible!");
                            }
                        }
                    }
                    else
                    {
                        //fprintf(stderr,"Unbekannter LGS-Fall oder LGS ist nicht berechenbar!\n");
                    }
                }
            }
            if ((weg1 == true) || (weg2 == true) || (weg3 == true) || (weg4 == true) || (weg5 == true) || (weg6 == true))
            {
                //Vorwärtselimination! Zerlegung von A in R und L mit lösen des Hilfsvektors y
                for (int u = 0; u < n; u++) // Matrixeintraege uebertragen bzw. Nullen
                {
                    for (int j = u; j < n; j++)
                    { //Bestimmen von R
                        for (int k = 0; k < u; k++)
                        {
                            A[u][j] = A[u][j] - A[u][k] * A[k][j];
                        }
                    }
                    for (int j = u + 1; j < n; j++)
                    { //Bestimmen von L
                        for (int k = 0; k < u; k++)
                        {
                            A[j][u] = A[j][u] - A[j][k] * A[k][u];
                        }
                        A[j][u] = A[j][u] / A[u][u];
                    }
                }
                //Loesen der Vorwaertselimination
                for (int a = 0; a < n; a++)
                {
                    y[a] = b[a];
                    for (int k = 0; k < a; k++)
                    {
                        y[a] = y[a] - A[a][k] * y[k];
                    }
                }
                //Rueckwaertseinsetzen
                for (int a = n - 1; a > -1; a--)
                {
                    b[a] = y[a] / A[a][a];
                    for (int k = a + 1; k < n; k++)
                    {
                        b[a] = b[a] - A[a][k] * b[k] / A[a][a];
                    }
                }
                Lambda = b[0];
                Lambda1 = b[1];
                Lambda2 = b[2];

                ///ENDE GAUSS ELIMINATION
                if ((Lambda1 >= -0.00001) && (Lambda1 <= 1.00001) && (Lambda2 <= 1.00001) && (Lambda2 >= -0.00001) && (Lambda >= -0.00001) && (Lambda <= 1.00001))
                {
                    coVector Cut;
                    int laufindex = 0;
                    coVector tempVec;

                    Cut = GridStuetz + (GridRicht * Lambda);

                    if ((int)CutVec.size() < 1)
                    {
                        CutVec.push_back(Cut);
                    }
                    for (int d = 0; d < (int)CutVec.size(); d++)
                    {
                        tempVec = CutVec[d];
                        if ((Cut[0] != tempVec[0]) || (Cut[1] != tempVec[1]) || (Cut[2] != tempVec[2]))
                        {
                            laufindex++;
                        }
                        if (laufindex == (int)CutVec.size())
                        {
                            CutVec.push_back(Cut);
                            laufindex = 0;
                        }
                    }
                }
                else
                {
                    //fprintf(stderr,"Kein Schittpunkt berechnet!\n");
                }
            }
        }
    }

    return (int)CutVec.size() > 0;
}

const int *
coDoBasisTree::getBbox(std::vector<float> &boundingBox) const
{
    memcpy(grid_bbox_, (float *)gridBBox.getDataPtr(), 6 * sizeof(float));
    for (int r = 0; r < 6; r++)
    {
        boundingBox.push_back(grid_bbox_[r]);
    }
    return (&cellList[0]);
}
///Bereich fuer Gitterelemente Octtree
//finde heraus, welche Zellen des Gridelemente-Octree innerhalb oder auf der Polygonen-Bbox liegen
//Fall 1 - Zellen liegen innerhalb der Polygonen-Bbox
//Fall 2 - Zellen liegen auf der Polygonen-Bbox (teilweise auserhalb)
const int * //sucht alle Zellen die innerhalb eines bestimmten räumlichen Gebiets liegen
    coDoBasisTree::area_search(std::vector<float> &Bbox1, std::vector<int> &GridElemList) const
{
    memcpy(grid_bbox_, (float *)gridBBox.getDataPtr(), 6 * sizeof(float));
    fX_ = fXShm;
    fY_ = fYShm;
    fZ_ = fZShm;
    //fprintf(stderr,"fX_ = %d, fY_ = %d, fZ_ = %d\n",fX_,fY_,fZ_);
    float Bbox[6];
    for (int n = 0; n < 6; n++)
    {
        Bbox[n] = Bbox1[n];
    }
    coVector point1;
    coVector point2;
    std::vector<coVector> CutVec;
    int hex_start[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
    int hex_end[12] = { 1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7 };

    float MacroCellBbox[6];
    for (int x_key = 0; x_key < (fX_); x_key++) //fX_,fY_,fZ_ entsprechen der max Zellenanzahl in die jeweilige Richtung
    {
        //fprintf(stderr,"x_key = %d , ",x_key);
        for (int y_key = 0; y_key < (fY_); y_key++)
        {
            //fprintf(stderr,"y_key = %d , ",y_key);

            for (int z_key = 0; z_key < (fZ_); z_key++)
            {
                //fprintf(stderr,"z_key = %d , \n",z_key);

                int xyz_key[3];
                xyz_key[0] = x_key;
                xyz_key[1] = y_key;
                xyz_key[2] = z_key;
                int position = Position(xyz_key);
                IniBBox(MacroCellBbox, xyz_key);

                //if Makrozelle liegt komplett innerhalb der PolygonenBbox -> alle Zellen uebertragen
                if ((((MacroCellBbox[0] >= Bbox[0]) && (MacroCellBbox[0] <= Bbox[3])) && ((MacroCellBbox[3] >= Bbox[0]) && (MacroCellBbox[3] <= Bbox[3]))) && (((MacroCellBbox[1] >= Bbox[1]) && (MacroCellBbox[1] <= Bbox[4])) && ((MacroCellBbox[4] >= Bbox[1]) && (MacroCellBbox[4] <= Bbox[4]))) && (((MacroCellBbox[2] >= Bbox[2]) && (MacroCellBbox[2] <= Bbox[5])) && ((MacroCellBbox[5] >= Bbox[2]) && (MacroCellBbox[5] <= Bbox[5]))))
                {
                    getMacroCellElements(position, MacroCellBbox, GridElemList, Bbox);
                }
                //else Makrozelle liegt gar nicht oder teilweise innerhalb in PolygonenBbox. Oder die Polygonen-Bbox ist zu klein fuer die Makrozelle -> weitere Pruefung durch Schnittueberpruefung
                else
                { //Jede Kante der Makrozelle wird mit den Grenzen der PolygonenBbox geschnitten, tritt ein Schnitt liegt ein Teil der Makrozelle innerhalb der Polygonen-Bbox
                    ///////////////////////////////////
                    coVector punktArray[8];
                    punktArray[0][0] = MacroCellBbox[3]; //xmax	//Punkt1
                    punktArray[0][1] = MacroCellBbox[1]; //ymin
                    punktArray[0][2] = MacroCellBbox[5]; //zmax
                    punktArray[1][0] = MacroCellBbox[0]; //xmin	//Punkt2
                    punktArray[1][1] = MacroCellBbox[1]; //ymin
                    punktArray[1][2] = MacroCellBbox[5]; //zmax
                    punktArray[2][0] = MacroCellBbox[0]; //xmin	//Punkt3
                    punktArray[2][1] = MacroCellBbox[4]; //ymax
                    punktArray[2][2] = MacroCellBbox[5]; //zmax
                    punktArray[3][0] = MacroCellBbox[3]; //xmax	//Punkt4
                    punktArray[3][1] = MacroCellBbox[4]; //ymax
                    punktArray[3][2] = MacroCellBbox[5]; //zmax
                    punktArray[4][0] = MacroCellBbox[3]; //xmax	//Punkt5
                    punktArray[4][1] = MacroCellBbox[1]; //ymin
                    punktArray[4][2] = MacroCellBbox[2]; //zmin
                    punktArray[5][0] = MacroCellBbox[0]; //xmin	//Punkt6
                    punktArray[5][1] = MacroCellBbox[1]; //ymin
                    punktArray[5][2] = MacroCellBbox[2]; //zmin
                    punktArray[6][0] = MacroCellBbox[0]; //xmin	//Punkt7
                    punktArray[6][1] = MacroCellBbox[4]; //ymax
                    punktArray[6][2] = MacroCellBbox[2]; //zmin
                    punktArray[7][0] = MacroCellBbox[3]; //xmax	//Punkt8
                    punktArray[7][1] = MacroCellBbox[4]; //ymax
                    punktArray[7][2] = MacroCellBbox[2]; //zmin

                    //Teile BoundingBox in Geraden auf
                    bool didCut = false;
                    for (int r = 0; r < 12; r++)
                    {
                        if (didCut == false)
                        {
                            point1[0] = punktArray[hex_start[r]][0];
                            point1[1] = punktArray[hex_start[r]][1];
                            point1[2] = punktArray[hex_start[r]][2];

                            point2[0] = punktArray[hex_end[r]][0];
                            point2[1] = punktArray[hex_end[r]][1];
                            point2[2] = punktArray[hex_end[r]][2];

                            if ((cutLineCuboid(point1, point2, Bbox, CutVec)) == true)
                            {
                                //fprintf(stderr,"CutVec.size = %d\n",(int) CutVec.size());
                                getMacroCellElements(position, MacroCellBbox, GridElemList, Bbox);
                                //fprintf(stderr,"cut true1\n");
                                didCut = true;
                            }
                        }
                        if (didCut == true)
                        {
                            r = 12;
                        }
                    }
                    if (didCut == false)
                    {
                        if ((((MacroCellBbox[0] < Bbox[0]) && (MacroCellBbox[0] > Bbox[3])) && ((MacroCellBbox[3] < Bbox[0]) && (MacroCellBbox[3] > Bbox[3]))) && (((MacroCellBbox[1] < Bbox[1]) && (MacroCellBbox[1] > Bbox[4])) && ((MacroCellBbox[4] < Bbox[1]) && (MacroCellBbox[4] > Bbox[4]))) && (((MacroCellBbox[2] < Bbox[2]) && (MacroCellBbox[2] > Bbox[5])) && ((MacroCellBbox[5] < Bbox[2]) && (MacroCellBbox[5] > Bbox[5]))))
                        { //dann liegt die Polygonen-Bbox in einer Makrozelle
                            getMacroCellElements(position, MacroCellBbox, GridElemList, Bbox);
                        }
                    }
                    //////////////////////////////////
                }
            }
        }
    }
    return (&cellList[0]);
}

//void
//coDoBasisTree::sonsInSpecificArea(std::vector<float> area, std::vector<int> &ElementList,...

//get all Elements of one single macro cell (octree)
void
coDoBasisTree::getMacroCellElements(int baum, float bbox[6], std::vector<int> &ElementList, const float *reference_Bbox) const
{
    if (macroCellList[baum] == 0)
        return;
    if (macroCellList[baum] < 0)
    {
        if (cellList[-macroCellList[baum]] == 0)
            return;
        // a leaf with population
        //fprintf(stderr,"Anz Elem in Son%d = %d\n",baum,cellList[(-macroCellList[baum])]);
        int laufindex = cellList[-macroCellList[baum]] + 1;
        int laufindex2 = 0;

        for (int c = 1; c < laufindex; c++)
        {
            ElementList.push_back(cellList[(-macroCellList[baum]) + c]);
        }
    }
    else
    {
        // >0 -> not a leaf
        int son;
        for (son = 0; son < 8; ++son)
        {
            //fprintf(stderr,"son = %d\n", son);
            float bbox_son[6];
            fillBBoxSon(bbox_son, bbox, son);
            //liegt bbox innerhalb der referenz-grenzen?
            if ((((bbox_son[0] > reference_Bbox[0]) && (bbox_son[0] < reference_Bbox[3])) && ((bbox_son[3] > reference_Bbox[0]) && (bbox_son[3] < reference_Bbox[3]))) && (((bbox_son[1] > reference_Bbox[1]) && (bbox_son[1] < reference_Bbox[4])) && ((bbox_son[4] > reference_Bbox[1]) && (bbox_son[4] < reference_Bbox[4]))) && (((bbox_son[2] > reference_Bbox[2]) && (bbox_son[2] < reference_Bbox[5])) && ((bbox_son[5] > reference_Bbox[2]) && (bbox_son[5] < reference_Bbox[5]))))
            {
                getMacroCellElements(macroCellList[baum] + son, bbox_son, ElementList, reference_Bbox);
            }
            else
            {
                //wenn nicht, schneiden sich die Grenzen möglicherweise?
                ///////////////////////////////////
                coVector punktArray[8];
                punktArray[0][0] = bbox_son[3]; //xmax	//Punkt1
                punktArray[0][1] = bbox_son[1]; //ymin
                punktArray[0][2] = bbox_son[5]; //zmax
                punktArray[1][0] = bbox_son[0]; //xmin	//Punkt2
                punktArray[1][1] = bbox_son[1]; //ymin
                punktArray[1][2] = bbox_son[5]; //zmax
                punktArray[2][0] = bbox_son[0]; //xmin	//Punkt3
                punktArray[2][1] = bbox_son[4]; //ymax
                punktArray[2][2] = bbox_son[5]; //zmax
                punktArray[3][0] = bbox_son[3]; //xmax	//Punkt4
                punktArray[3][1] = bbox_son[4]; //ymax
                punktArray[3][2] = bbox_son[5]; //zmax
                punktArray[4][0] = bbox_son[3]; //xmax	//Punkt5
                punktArray[4][1] = bbox_son[1]; //ymin
                punktArray[4][2] = bbox_son[2]; //zmin
                punktArray[5][0] = bbox_son[0]; //xmin	//Punkt6
                punktArray[5][1] = bbox_son[1]; //ymin
                punktArray[5][2] = bbox_son[2]; //zmin
                punktArray[6][0] = bbox_son[0]; //xmin	//Punkt7
                punktArray[6][1] = bbox_son[4]; //ymax
                punktArray[6][2] = bbox_son[2]; //zmin
                punktArray[7][0] = bbox_son[3]; //xmax	//Punkt8
                punktArray[7][1] = bbox_son[4]; //ymax
                punktArray[7][2] = bbox_son[2]; //zmin

                coVector point1;
                coVector point2;
                std::vector<coVector> CutVec;
                int hex_start[12] = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
                int hex_end[12] = { 1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7 };

                //Teile BoundingBox in Geraden auf
                bool didCut = false;
                for (int r = 0; r < 12; r++)
                {
                    if (didCut == false)
                    {
                        point1[0] = punktArray[hex_start[r]][0];
                        point1[1] = punktArray[hex_start[r]][1];
                        point1[2] = punktArray[hex_start[r]][2];
                        point2[0] = punktArray[hex_end[r]][0];
                        point2[1] = punktArray[hex_end[r]][1];
                        point2[2] = punktArray[hex_end[r]][2];
                        if ((cutLineCuboid(point1, point2, reference_Bbox, CutVec)) == true)
                        {
                            //fprintf(stderr,"CutVec.size = %d\n",(int) CutVec.size());
                            getMacroCellElements(macroCellList[baum] + son, bbox_son, ElementList, reference_Bbox);
                            //fprintf(stderr,"cut true\n");
                            didCut = true;
                        }
                    }
                    if (didCut == true)
                    {
                        r = 12;
                    }
                }
                if (didCut == false)
                {
                    if ((((bbox_son[0] < reference_Bbox[0]) && (bbox_son[0] > reference_Bbox[3])) && ((bbox_son[3] < reference_Bbox[0]) && (bbox_son[3] > reference_Bbox[3]))) && (((bbox_son[1] < reference_Bbox[1]) && (bbox_son[1] > reference_Bbox[4])) && ((bbox_son[4] < reference_Bbox[1]) && (bbox_son[4] > reference_Bbox[4]))) && (((bbox_son[2] < reference_Bbox[2]) && (bbox_son[2] > reference_Bbox[5])) && ((bbox_son[5] < reference_Bbox[2]) && (bbox_son[5] > reference_Bbox[5]))))
                    { //dann liegt die Polygonen-Bbox in einer Makrozelle
                        getMacroCellElements(macroCellList[baum] + son, bbox_son, ElementList, reference_Bbox);
                    }
                }
                //////////////////////////////////
            }
        }
    }
}
/// ///////////////////////////////////////////////
////////////////////////////
void
coDoBasisTree::getChunks(vector<const int *> &chunks, const functionObject *test)
{
    chunks.clear();
    chunks.reserve(macroCellList.get_length() / 8);
    if (!test)
        return;
    // loop over trees
    int key[3];
    memcpy(grid_bbox_, (float *)gridBBox.getDataPtr(), 6 * sizeof(float));
    for (key[2] = 0; key[2] < int(fZShm); ++(key[2]))
    {
        for (key[1] = 0; key[1] < int(fYShm); ++(key[1]))
        {
            for (key[0] = 0; key[0] < int(fXShm); ++(key[0]))
            {
                int position = Position(key);
                float bbox[6];
                bbox[0] = gridBBox[0] + (gridBBox[3] - gridBBox[0]) * key[0] / int(fXShm);
                bbox[1] = gridBBox[1] + (gridBBox[4] - gridBBox[1]) * key[1] / int(fYShm);
                bbox[2] = gridBBox[2] + (gridBBox[5] - gridBBox[2]) * key[2] / int(fZShm);
                bbox[3] = gridBBox[0] + (gridBBox[3] - gridBBox[0]) * (key[0] + 1) / int(fXShm);
                bbox[4] = gridBBox[1] + (gridBBox[4] - gridBBox[1]) * (key[1] + 1) / int(fYShm);
                bbox[5] = gridBBox[2] + (gridBBox[5] - gridBBox[2]) * (key[2] + 1) / int(fZShm);
                addChunk(chunks, position, bbox, test);
            }
        }
    }
}

void
coDoBasisTree::addChunk(vector<const int *> &chunks,
                        int position, const float bbox[6], const functionObject *test)
{
    if (!test->operator()(bbox))
        return;
    if (macroCellList[position] < 0)
    {
        chunks.push_back(&(cellList[-macroCellList[position]]));
    }
    else if (macroCellList[position] > 0)
    {
        int son;
        for (son = 0; son < 8; ++son)
        {
            // evaluate bbox for this son
            float sonBox[6];
            fillBBoxSon(sonBox, bbox, son);
            addChunk(chunks, macroCellList[position] + son, sonBox, test);
        }
    }
}

// delve into oct-tree
const int *
coDoBasisTree::lookUp(int position, int *okey, int mask) const
{
    int son = 0;
    if (macroCellList[position] <= 0)
    {
        return &cellList[-macroCellList[position]];
    }
    if (okey[2] & mask)
        son = 1;
    son <<= 1;
    if (okey[1] & mask)
        son += 1;
    son <<= 1;
    if (okey[0] & mask)
        son += 1;

    // we got the son
    mask >>= 1;
    return lookUp(macroCellList[position] + son, okey, mask);
}

// here the grid_bbox_ array has been already initialised
void
coDoBasisTree::DivideUpToLevel()
{
    // Calculate initial fX_,fY_,fZ_
    fX_ = 1;
    fY_ = 1;
    fZ_ = 1;
    float lx = grid_bbox_[3] - grid_bbox_[0];
    float ly = grid_bbox_[4] - grid_bbox_[1];
    float lz = grid_bbox_[5] - grid_bbox_[2];
    while (fX_ * fY_ * fZ_ * normal_size_ < nelem
           && (2 * fX_ <= limit_fX_
               || 2 * fY_ <= limit_fY_
               || 2 * fZ_ <= limit_fZ_))
    {
        int whichMayBeDivided[3] = // all in pple
            {
              1, 1, 1
            };
        if (2 * fX_ > limit_fX_)
        {
            whichMayBeDivided[0] = 0;
        }
        if (2 * fY_ > limit_fY_)
        {
            whichMayBeDivided[1] = 0;
        }
        if (2 * fZ_ > limit_fZ_)
        {
            whichMayBeDivided[2] = 0;
        }

        float max = 0.0;
        int whichToDivide = -1;
        if (whichMayBeDivided[0] && max < lx)
        {
            max = lx;
            whichToDivide = 0;
        }
        if (whichMayBeDivided[1] && max < ly)
        {
            max = ly;
            whichToDivide = 1;
        }
        if (whichMayBeDivided[2] && max < lz)
        {
            max = lz;
            whichToDivide = 2;
        }

        if (whichToDivide == 0)
        {
            fX_ *= 2;
            lx *= 0.5;
        }
        else if (whichToDivide == 1)
        {
            fY_ *= 2;
            ly *= 0.5;
        }
        else if (whichToDivide == 2)
        {
            fZ_ *= 2;
            lz *= 0.5;
        }
        else
        {
            break; // this is impossible!!!
        }
    }
}

// used in operator<<
void
coDoBasisTree::treePrint(ostream &outfile, int level, int *key, int offset)
{
    int entry = macroCellList[offset];
    if (entry <= 0)
    {
        // leaf
        outfile << "Leaf key: " << key[0] << ' ' << key[1] << ' ' << key[2] << " has " << cellList[-entry] << " cells:" << endl;
        int i;
        for (i = 0; i < cellList[-entry]; ++i)
        {
            outfile << ' ' << cellList[-entry + 1 + i];
        }
        outfile << endl;
    }
    else
    {
        outfile << "Macro key: " << key[0] << ' ' << key[1] << ' ' << key[2] << " has 8 sons:" << endl;
        int son;
        int son_key[3];
        for (son = 0; son < 8; ++son)
        {
            son_key[0] = key[0];
            son_key[1] = key[1];
            son_key[2] = key[2];
            son_key[0] <<= 1;
            son_key[1] <<= 1;
            son_key[2] <<= 1;
            if (son & 1)
                son_key[0] += 1;
            if (son & 2)
                son_key[1] += 1;
            if (son & 4)
                son_key[2] += 1;
            outfile << "       " << son_key[0] << ' ' << son_key[1] << ' ' << son_key[2] << endl;
        }
        for (son = 0; son < 8; ++son)
        {
            son_key[0] = key[0];
            son_key[1] = key[1];
            son_key[2] = key[2];
            son_key[0] <<= 1;
            son_key[1] <<= 1;
            son_key[2] <<= 1;
            if (son & 1)
                son_key[0] += 1;
            if (son & 2)
                son_key[1] += 1;
            if (son & 4)
                son_key[2] += 1;
            treePrint(outfile, level + 1, son_key, entry + son);
        }
    }
}

namespace covise
{

ostream &
operator<<(ostream &outfile, coDoBasisTree &tree)
{
    int key[3] = { 0, 0, 0 };
    int i, j, k;
    for (k = 0; k < tree.fZ_; ++k)
    {
        for (j = 0; j < tree.fY_; ++j)
        {
            for (i = 0; i < tree.fX_; ++i)
            {
                outfile << "OctTree loc: " << i << ' ' << j << ' ' << k << endl;
                tree.treePrint(outfile, 0, key, i + j * tree.fX_ + k * tree.fX_ * tree.fY_);
            }
        }
    }
    return outfile;
}
}

// with the output list, the user programmer may build up
// a coDoLines object that represent the OctTree
void
coDoBasisTree::Visualise(std::vector<int> &ll,
                         std::vector<int> &cl,
                         std::vector<float> &xi,
                         std::vector<float> &yi,
                         std::vector<float> &zi)
{
    ll.clear();
    cl.clear();
    xi.clear();
    yi.clear();
    zi.clear();

    fX_ = fXShm;
    fY_ = fYShm;
    fZ_ = fZShm;
    // loop over octtrees
    int baum;
    int key[3];
    int &i = key[0];
    int &j = key[1];
    int &k = key[2];
    float bbox[6];
    for (k = 0; k < fZ_; ++k)
    {
        for (j = 0; j < fY_; ++j)
        {
            for (i = 0; i < fX_; ++i)
            {
                baum = Position(key);
                IniBBox(bbox, key);
                VisualiseOneOctTree(baum, bbox, ll, cl, xi, yi, zi);
            }
        }
    }
}

// used in the previous function. It gets
// recursively the geometric info for the deepest children
// with a non-null population
void
coDoBasisTree::VisualiseOneOctTree(int baum,
                                   const float bbox[6],
                                   std::vector<int> &ll,
                                   std::vector<int> &cl,
                                   std::vector<float> &xi,
                                   std::vector<float> &yi,
                                   std::vector<float> &zi)
{
    if (macroCellList[baum] == 0)
        return;
    if (macroCellList[baum] < 0)
    {
        if (cellList[-macroCellList[baum]] == 0)
            return;
        // a leaf with population
        ll.push_back((int)cl.size());
        int base_conn = (int)xi.size();
        cl.push_back(base_conn + 0);
        cl.push_back(base_conn + 1);
        cl.push_back(base_conn + 2);
        cl.push_back(base_conn + 3);
        cl.push_back(base_conn + 0);
        cl.push_back(base_conn + 4);
        cl.push_back(base_conn + 5);
        cl.push_back(base_conn + 1);
        cl.push_back(base_conn + 2);
        cl.push_back(base_conn + 6);
        cl.push_back(base_conn + 7);
        cl.push_back(base_conn + 4);
        cl.push_back(base_conn + 5);
        cl.push_back(base_conn + 6);
        cl.push_back(base_conn + 7);
        cl.push_back(base_conn + 3);
        xi.push_back(bbox[0]); // point 0
        yi.push_back(bbox[1]);
        zi.push_back(bbox[2]);
        xi.push_back(bbox[3]); // point 1
        yi.push_back(bbox[1]);
        zi.push_back(bbox[2]);
        xi.push_back(bbox[3]); // point 2
        yi.push_back(bbox[4]);
        zi.push_back(bbox[2]);
        xi.push_back(bbox[0]); // point 3
        yi.push_back(bbox[4]);
        zi.push_back(bbox[2]);
        xi.push_back(bbox[0]); // point 4
        yi.push_back(bbox[1]);
        zi.push_back(bbox[5]);
        xi.push_back(bbox[3]); // point 5
        yi.push_back(bbox[1]);
        zi.push_back(bbox[5]);
        xi.push_back(bbox[3]); // point 6
        yi.push_back(bbox[4]);
        zi.push_back(bbox[5]);
        xi.push_back(bbox[0]); // point 7
        yi.push_back(bbox[4]);
        zi.push_back(bbox[5]);
    }
    else
    {
        // >0 -> not a leaf
        int son;
        for (son = 0; son < 8; ++son)
        {
            float bbox_son[6];
            fillBBoxSon(bbox_son, bbox, son);
            VisualiseOneOctTree(macroCellList[baum] + son, bbox_son, ll, cl, xi, yi, zi);
        }
    }
}

int
coDoBasisTree::getObjInfo(int no,
                          coDoInfo **il) const
{
    if (no == SHM_OBJ)
    {
        (*il)[0].description = "Cell list";
        (*il)[1].description = "Macrocell list";
        (*il)[2].description = "Grid bounding box";
        (*il)[3].description = "Cell bounding boxes";
        (*il)[4].description = "Number of X div.";
        (*il)[5].description = "Number of Y div.";
        (*il)[6].description = "Number of Z div.";
        (*il)[7].description = "Max. number of levels.";
        return SHM_OBJ;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

void
coDoBasisTree::RecreateShmDL(covise_data_list *dl)
{
    cellList.set_length((int)cellList_.size());
    macroCellList.set_length((int)macCellList_.size());
    cellBBoxes.set_length(6 * nelem);
    gridBBox.set_length(6);

    dl[0].type = INTSHMARRAY;
    dl[0].ptr = (void *)&cellList;
    dl[1].type = INTSHMARRAY;
    dl[1].ptr = (void *)&macroCellList;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&gridBBox;
    dl[3].type = FLOATSHMARRAY;
    dl[3].ptr = (void *)&cellBBoxes;
    dl[4].type = INTSHM;
    dl[4].ptr = (void *)&fXShm;
    dl[5].type = INTSHM;
    dl[5].ptr = (void *)&fYShm;
    dl[6].type = INTSHM;
    dl[6].ptr = (void *)&fZShm;
    dl[7].type = INTSHM;
    dl[7].ptr = (void *)&max_no_levels_Shm;
    // update_shared_dl(SHM_OBJ, dl);
    new_ok = store_shared_dl(SHM_OBJ, dl) != 0;

    if (!new_ok)
        return;
    cellBBoxes.get_length();

    fXShm = fX_;
    fYShm = fY_;
    fZShm = fZ_;
    max_no_levels_Shm = max_no_levels_;

    // fill lists
    float *cell_list = (float *)cellList.getDataPtr();
    float *mac_cell_list = (float *)macroCellList.getDataPtr();
    if (cellList_.size() > 0)
    {
        memcpy(cell_list, &cellList_[0], cellList_.size() * sizeof(int));
    }
    if (macCellList_.size() > 0)
    {
        memcpy(mac_cell_list, &macCellList_[0], macCellList_.size() * sizeof(int));
    }

    // fill bboxes
    float *gb = (float *)gridBBox.getDataPtr();
    memcpy(gb, &grid_bbox_[0], 6 * sizeof(float));
    float *cb = NULL;
    if (nelem > 0)
    {
        cb = (float *)cellBBoxes.getDataPtr();
        memcpy(cb, &cellBBoxes_[0], cellBBoxes_.size() * sizeof(float));
    }
    cellBBoxes_.clear();
}

int
coDoBasisTree::rebuildFromShm()
{
    covise_data_list dl[SHM_OBJ];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = INTSHMARRAY;
    dl[0].ptr = (void *)&cellList;
    dl[1].type = INTSHMARRAY;
    dl[1].ptr = (void *)&macroCellList;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&gridBBox;
    dl[3].type = FLOATSHMARRAY;
    dl[3].ptr = (void *)&cellBBoxes;
    dl[4].type = INTSHM;
    dl[4].ptr = (void *)&fXShm;
    dl[5].type = INTSHM;
    dl[5].ptr = (void *)&fYShm;
    dl[6].type = INTSHM;
    dl[6].ptr = (void *)&fZShm;
    dl[7].type = INTSHM;
    dl[7].ptr = (void *)&max_no_levels_Shm;
    return restore_shared_dl(SHM_OBJ, dl);
}

int
coDoBasisTree::getNumCellLists()
{
    return cellList.get_length();
}
int
coDoBasisTree::getNumMacroCellLists()
{
    return macroCellList.get_length();
}
int
coDoBasisTree::getNumCellBBoxes()
{
    return cellBBoxes.get_length();
}
int
coDoBasisTree::getNumGridBBoxes()
{
    return gridBBox.get_length();
}

void coDoBasisTree::getAddresses(int **cl, int **mcl, float **cb, float **gb, int **fX, int **fY, int **fZ, int **ml)
{
    *cl = (int *)cellList.getDataPtr();
    *mcl = (int *)macroCellList.getDataPtr();
    *cb = (float *)cellBBoxes.getDataPtr();
    *gb = (float *)gridBBox.getDataPtr();
    *fX = (int *)fXShm.getDataPtr();
    *fY = (int *)fYShm.getDataPtr();
    *fZ = (int *)fZShm.getDataPtr();
    *ml = (int *)max_no_levels_Shm.getDataPtr();
}
