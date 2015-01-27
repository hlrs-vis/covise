/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _COVISE_MINI_GRID_H_
#define _COVISE_MINI_GRID_H_

// These are auxiliary classes involved in the RainAlgorithm
// as used in class coFeatureLines

#include "MagmaUtils.h"
#include "unordered_set.h"

namespace covise
{

// coTriEdge stands for an ordered pair of nodes defining an
// edge in a grid
class coTriEdge
{
public:
    coTriEdge();
    coTriEdge(int i, int j);
    bool operator==(const coTriEdge &rhs) const;
    bool operator<(const coTriEdge &rhs) const;
    int getMin() const;
    int getMax() const;

private:
    int min_;
    int max_;
};

// HashCoTriEdge is defined in order to improve
// the performance of the algorithm, as it makes
// it possible to use hashed containers
struct HashCoTriEdge
{
    size_t operator()(const coTriEdge &edge) const;
    bool operator()(const coTriEdge &edge0, const coTriEdge &edge1) const;
#if defined(__INTEL_COMPILER) || defined(WIN32)
    enum
    {
        bucket_size = 4,
        min_buckets = 8
    };
#endif
};

// coTestNeighbour describes a neighbourhood
// relationship between a given cell (not the one
// indicated in the contructor) and another one (the one
// referrred to by parameter cell in constructor or variable cell_).
class coTestNeighbour
{
public:
    // cell has to refer to the mini-numbering (see below)
    coTestNeighbour(int cell, coTriEdge edge);
    int CellLabel() const;
    coTriEdge GetEdge() const;

private:
    int cell_;
    coTriEdge edge_;
};

// coFeatureBorder describes a set of edges
class coFeatureBorder
{
public:
    coFeatureBorder(const unordered_set<coTriEdge, HashCoTriEdge> &set_kanten);
    virtual ~coFeatureBorder();
    // tests whether the edge associated with neighbourhood tn
    // belongs to this set of edges
    bool IsIn(const coTestNeighbour &tn) const;

private:
    bool IsIn(const coTriEdge &edge) const;
    unordered_set<coTriEdge, HashCoTriEdge> hash_segments_;
};

// coTestNeighbourhoodContainer describes a set of
// neighbour relationships between a given cell and
// its neighbours
class coTestNeighbourhoodContainer
{
public:
    void Add(const coTestNeighbour &tn)
    {
        container_.push_back(tn);
    }
    typedef vector<coTestNeighbour>::iterator iterator;
    iterator begin()
    {
        return container_.begin();
    }
    iterator end()
    {
        return container_.end();
    }

private:
    vector<coTestNeighbour> container_;
};

// coMiniGrid describes a subset of a grid
// In our application, it describes the subgrid around
// a given node
class coMiniGrid
{
public:
    typedef coFeatureBorder Border;
    typedef coTestNeighbourhoodContainer NeighbourhoodContainer;

    coMiniGrid(const vector<int> &minigrid_cells,
               const vector<int> &elem_start_neigh,
               const vector<int> &elem_number_neigh,
               const vector<int> &elem_neighbours, // neighbour cells of a cell
               //as long
               const vector<MagmaUtils::Edge> &edge_neighbours);
    //as elem_neighbours
    virtual ~coMiniGrid();
    int CellSize() const;
    void GetEdgeNeighbours(int cell, coTestNeighbourhoodContainer &nc) const;

private:
    const vector<int> &minigrid_cells_; // contains the cells in the subset
    // the following arrays are the neighbour information
    // for the whole grid
    const vector<int> &elem_start_neigh_;
    const vector<int> &elem_number_neigh_;
    const vector<int> &elem_neighbours_; // neighbour cells of a cell
    //as long
    const vector<MagmaUtils::Edge> &edge_neighbours_;
    //as elem_neighbours_
};
}
#endif
