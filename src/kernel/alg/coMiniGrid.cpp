/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coMiniGrid.h"
#if defined(__INTEL_COMPILER)
#include <algorithm>
#endif

using namespace covise;

coTriEdge::coTriEdge()
    : min_(0)
    , max_(0)
{
}

coTriEdge::coTriEdge(int i, int j)
    : min_(i)
    , max_(j)
{
    if (j < i)
    {
        min_ = j;
        max_ = i;
    }
}

bool
    coTriEdge::
    operator==(const coTriEdge &rhs) const
{
    return (min_ == rhs.min_ && max_ == rhs.max_);
}

bool
    coTriEdge::
    operator<(const coTriEdge &rhs) const
{
    return (min_ < rhs.min_ || (min_ == rhs.min_ && max_ < rhs.max_));
}

int
coTriEdge::getMin() const
{
    return min_;
}

int
coTriEdge::getMax() const
{
    return max_;
}

// +++++++++++++++++++++++++++++++++++++++++++++

size_t
    HashCoTriEdge::
    operator()(const coTriEdge &edge) const
{
    return size_t(557 * edge.getMin() + edge.getMax());
}

bool
    HashCoTriEdge::
    operator()(const coTriEdge &edge0, const coTriEdge &edge1) const
{

    // wer macht so einen Scheiss???????return (this->operator()(edge0) < this->operator()(edge0));
    // das ist ==´return(false);
    return (!(edge0.getMin() == edge1.getMin() && edge0.getMax() == edge1.getMax()));
}

// +++++++++++++++++++++++++++++++++++++++++++++

coTestNeighbour::coTestNeighbour(int cell, coTriEdge edge)
    : cell_(cell)
    , edge_(edge)
{
}

int
coTestNeighbour::CellLabel() const
{
    return cell_;
}

coTriEdge
coTestNeighbour::GetEdge() const
{
    return edge_;
}

// ++++++++++++++++++++++++++++++++++++++++++++++++

coFeatureBorder::coFeatureBorder(
    const unordered_set<coTriEdge, HashCoTriEdge> &set_kanten)
    : hash_segments_(set_kanten)
{
}

coFeatureBorder::~coFeatureBorder()
{
}

bool
coFeatureBorder::IsIn(const coTriEdge &edge) const
{
    return (hash_segments_.find(edge) != hash_segments_.end());
}

bool
coFeatureBorder::IsIn(const coTestNeighbour &tn) const
{
    return IsIn(tn.GetEdge());
}

// +++++++++++++++++++++++++++++++++++++++++++++++

coMiniGrid::coMiniGrid(
    const vector<int> &minigrid_cells,
    const vector<int> &elem_start_neigh,
    const vector<int> &elem_number_neigh,
    const vector<int> &elem_neighbours, // neighbour cells of a cell
    const vector<MagmaUtils::Edge> &edge_neighbours)
    : //as long
    // as elem_neighbours
    minigrid_cells_(minigrid_cells)
    , elem_start_neigh_(elem_start_neigh)
    , elem_number_neigh_(elem_number_neigh)
    , elem_neighbours_(elem_neighbours)
    , edge_neighbours_(edge_neighbours)
{
}

coMiniGrid::~coMiniGrid()
{
}

int
coMiniGrid::CellSize() const
{
    return (int)minigrid_cells_.size();
}

// be careful now, cell refers to the internal 'mini'-numbering,
// not the global one.
// In this function a container of TestNeighbours is build up, and
// from those elements, 'CellLabel' has to return a label
// referring to the 'mini'-numbering. Otherwise, the RainAlgorithm
// would go nuts.
void
coMiniGrid::GetEdgeNeighbours(int cell, coTestNeighbourhoodContainer &nc) const
{
    // find global label
    int global_cell = minigrid_cells_[cell];
    int start_cell_neighbours = elem_start_neigh_[global_cell];
    int number_cell_neighbours = elem_number_neigh_[global_cell];
    // now take into account, that some global neighbours
    // are not in minigrid_cells_
    for (int neigh = 0; neigh < number_cell_neighbours; ++neigh)
    {
        // cell_label refers to the global numbering!!!
        int cell_label = elem_neighbours_[start_cell_neighbours + neigh];
        // see comment above
        vector<int>::const_iterator minigrid_cells_const_it = std::find(minigrid_cells_.begin(), minigrid_cells_.end(), cell_label);
        if (minigrid_cells_const_it == minigrid_cells_.end())
        {
            continue; // cell_label is not in this MiniGrid
        }
        int mini_cell_label = (int)(minigrid_cells_const_it - minigrid_cells_.begin());
        MagmaUtils::Edge edge = edge_neighbours_[start_cell_neighbours + neigh];
        coTestNeighbour tn(mini_cell_label, coTriEdge(edge.first, edge.second));
        nc.Add(tn);
    }
}
