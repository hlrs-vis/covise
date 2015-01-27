/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS BBoxAdmin
//
//  This class administrates coDoOctTree arrays for the UNSGRDs
//
//  Initial version: 2002-06-?? Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2002 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Changes:

#ifndef _BBOX_ADMIN_H_
#define _BBOX_ADMIN_H_

#include <util/coviseCompat.h>
#include <do/coDistributedObject.h>
#include <map>

#include <vector>

using namespace covise;

class BBoxAdmin
{
public:
    void reload(const coDistributedObject *grid, const coDistributedObject *otree);
    /** assign a new octtree to an input grid or calculate a new one
       * @param grid pointer to input grid
       * @param otree pointer to input octtree or NULL
       */
    void load(const coDistributedObject *grid, const coDistributedObject *otree);
    /** setSurname sets the octSurname_ variable, an name appendage
       *  to ensure that the octtree gets a good name if it has to be created
       */
    void setSurname();
    /// Constructor
    BBoxAdmin();
    /// Destructor
    ~BBoxAdmin();
    /// AssignOctTrees assigns octtrees to grid arrays
    void assignOctTrees(std::vector<std::vector<const coDistributedObject *> > grid_tstep_opt) const;

protected:
private:
    void setGotOctTree(const coDistributedObject *otree);
    void pload(const coDistributedObject *grid, const coDistributedObject *otree);
    void preload(const coDistributedObject *grid, const coDistributedObject *otree);
    int gridSeq_;
    int creationCounter_;
    void clean();
    bool gotOctTrees_;
    // getUsedOctTree looks for a grid with the same name
    // as the argument in the iaUnsGrids_ list. If found,
    // we need not create a brand new octtree.
    const coDistributedObject *getUsedOctTree(const coDistributedObject *);
    std::vector<const coDistributedObject *> iaUnsGrids_;
    std::vector<const coDistributedObject *> iaOctTrees_;
    std::map<string, const coDistributedObject *> namesToOctTrees_;
    std::string octSurname_;
};
#endif
