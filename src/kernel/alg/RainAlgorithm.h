/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  TEMPLATE FUNCTION
//
//  RainAlgorithm classifies cells in a grid tagging each cell
//  with a number, in a such a way that regions separated by a border
//  will get a different tag, which is constant within one region.
//  These tags may be eventually used to generate a group of smaller
//  grids for the cells sharing a common tag.
//
//  As you can see from the code, class Grid should define some
//  typedefs and the involved classes are expected to support some
//  interfaces.
//
//  Initial version: 29.02.2004 Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2004 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes: 24.3.2004, performance was enhanced

#include "covise/covise.h"

namespace covise
{

template <class Grid>
void RainAlgorithm(const Grid &grid,
                   const typename Grid::Border &border,
                   std::vector<int> &tags)
{
    std::vector<int> local_tags(grid.CellSize(), -1);

    int no_cells = grid.CellSize(); // get number of cells
    int tag_label = 0;
    int progress_size = no_cells / 100;
    if (progress_size == 0)
        progress_size = 1;

    vector<int> unclassified_v(no_cells);
    for (int i = 0; i < no_cells; ++i)
    {
        unclassified_v[i] = i;
    }
    std::set<int> unclassified(unclassified_v.begin(), unclassified_v.end());

    while (!unclassified.empty())
    {
        int start = *(unclassified.begin());
        local_tags[start] = tag_label;
        unclassified.erase(unclassified.begin());
        set<int> active;
        active.insert(start);
        while (!active.empty())
        {
            // look for unmarked neighbours
            // of active within the domain
            set<int>::iterator active_it = active.begin();
            set<int>::iterator active_it_end = active.end();
            set<int> new_active;
            for (; active_it != active_it_end; ++active_it)
            {
                std::vector<int> neighbourLabels; // cells touched by this drop of rain
                // get neighbour cells
                typename Grid::NeighbourhoodContainer nc;
                grid.GetEdgeNeighbours(*active_it, nc);
                // filter out neighbours for which the common edge belongs to border
                typename Grid::NeighbourhoodContainer::iterator ncit = nc.begin();
                typename Grid::NeighbourhoodContainer::iterator ncend = nc.end();
                for (; ncit != ncend; ++ncit)
                {
                    // continue if neighbour was already marked
                    if (local_tags[ncit->CellLabel()] != -1)
                    {
                        continue;
                    }
                    // continue if neighbourhood crosses the border
                    if (border.IsIn((*ncit)))
                    {
                        continue;
                    }
                    // add to new_active
                    new_active.insert(ncit->CellLabel());
                }
            }
            // update unclassified and local_tags
            // using new_active -> active
            new_active.swap(active);
            active_it = active.begin();
            active_it_end = active.end();
            for (; active_it != active_it_end; ++active_it)
            {
                local_tags[*active_it] = tag_label;
                unclassified.erase(*active_it);
            }
        }
        ++tag_label;
    }
    local_tags.swap(tags);
}
}
