/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "BBoxAdmin.h"
#include <appl/ApplInterface.h>

#include <do/coDoOctTree.h>
#include <do/coDoPolygons.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoOctTreeP.h>

BBoxAdmin::BBoxAdmin()
{
    creationCounter_ = 0;
    gridSeq_ = 0; // index for oct-tree retrieval when reusing objects in preload
    gotOctTrees_ = false; // true when octtree is available at a port
}

void
BBoxAdmin::setSurname()
{
    // fixme
    octSurname_ = Covise::get_module();
    octSurname_ += "_";
    octSurname_ += Covise::get_instance();
    octSurname_ += "_";
    octSurname_ += Covise::get_host();
    /*
      strcpy(globalOctSurname,Covise::get_module());
      strcat(globalOctSurname,"_");
      strcat(globalOctSurname,Covise::get_instance());
      strcat(globalOctSurname,"_");
      strcat(globalOctSurname,Covise::get_host());
   */
}

BBoxAdmin::~BBoxAdmin()
{
    clean();
}

void
BBoxAdmin::clean()
{
    if (!gotOctTrees_) // Otherwise covise destroys the objects
    {
        int grid;
        for (grid = 0; grid < iaOctTrees_.size(); ++grid)
        {
            int prev;
            for (prev = 0; prev < grid; ++prev)
            {
                if (iaOctTrees_[grid] == iaOctTrees_[prev])
                {
                    break;
                }
            }
            if (prev == grid) // do not destroy an object more than once
            {
                const_cast<coDistributedObject *>(iaOctTrees_[grid])->destroy();
                delete const_cast<coDistributedObject *>(iaOctTrees_[grid]);
            }
        }
    }
    iaOctTrees_.clear();
    iaUnsGrids_.clear();
    namesToOctTrees_.clear();
}

// trivial, isn't it?
void
BBoxAdmin::setGotOctTree(const coDistributedObject *otree)
{
    gotOctTrees_ = (otree != NULL);
}

// loading new input octtrees for the grids or
// starting a calculation for them if otree==NULL
void
BBoxAdmin::load(const coDistributedObject *grid, const coDistributedObject *otree)
{
    clean();
    setGotOctTree(otree);
    pload(grid, otree);
    if (otree == NULL)
    {
        ++creationCounter_;
    }
}

// reusing available object
void
BBoxAdmin::reload(const coDistributedObject *grid, const coDistributedObject *otree)
{
    gridSeq_ = 0;
    if (otree != NULL)
    {
        clean();
    }
    setGotOctTree(otree);
    preload(grid, otree);
}

// used in load
void
BBoxAdmin::pload(const coDistributedObject *grid, const coDistributedObject *otree)
{
    if (grid->isType("SETELE"))
    {
        // calling pload recursively for all elements
        const coDoSet *set = dynamic_cast<const coDoSet *>(grid);
        int no_set_elems;
        const coDistributedObject *const *setList = set->getAllElements(&no_set_elems);
        int elem;
        if (otree == NULL)
        {
            for (elem = 0; elem < no_set_elems; ++elem)
            {
                pload(setList[elem], NULL);
#ifdef _CLEAN_UP_
                if (setList[elem]->isType("SETELE"))
                {
                    delete setList[elem];
                }
#endif
            }
        }
        else
        {
            const coDoSet *oset = dynamic_cast<const coDoSet *>(otree);
            int no_oset_elems;
            const coDistributedObject *const *osetList = oset->getAllElements(&no_oset_elems);
            for (elem = 0; elem < no_oset_elems; ++elem)
            {
                pload(setList[elem], osetList[elem]);
#ifdef _CLEAN_UP_
                if (setList[elem]->isType("SETELE"))
                {
                    delete setList[elem];
                }
                if (osetList[elem]->isType("SETELE"))
                {
                    delete osetList[elem];
                }
#endif
            }
#ifdef _CLEAN_UP_
            delete[] osetList;
#endif
        }
#ifdef _CLEAN_UP_
        delete[] setList;
#endif
    }
    else if (grid->isType("UNSGRD") && otree != NULL && otree->isType("OCTREE"))
    {
        // octtree available as input
        const coDoUnstructuredGrid *unsgrd = dynamic_cast<const coDoUnstructuredGrid *>(grid);
        unsgrd->GetOctTree(otree, NULL);
        iaOctTrees_.push_back(otree);
        iaUnsGrids_.push_back(grid);
        namesToOctTrees_[string(grid->getName())] = otree;
    }
    else if (grid->isType("UNSGRD"))
    {
        // octtree not available as input
        const coDoUnstructuredGrid *unsgrd = dynamic_cast<const coDoUnstructuredGrid *>(grid);
        std::string effectiveSurname_ = octSurname_;
        char buf[256];
        sprintf(buf, "_%d", creationCounter_);
        effectiveSurname_ += buf;
        const coDistributedObject *usedOctTree = getUsedOctTree(unsgrd);
        if (usedOctTree == NULL)
        {
            const coDistributedObject *octTree = unsgrd->GetOctTree(NULL, effectiveSurname_.c_str());
            iaOctTrees_.push_back(octTree);
            iaUnsGrids_.push_back(grid);
            namesToOctTrees_[string(grid->getName())] = octTree;
        }
        else
        {
            unsgrd->GetOctTree((coDoOctTree *)usedOctTree, NULL);
            iaOctTrees_.push_back(usedOctTree);
            iaUnsGrids_.push_back(grid);
            namesToOctTrees_[string(grid->getName())] = usedOctTree;
        }
    }
    else if (grid->isType("POLYGN") && otree != NULL && otree->isType("OCTREP"))
    {
        // octtree available as input
        const coDoPolygons *polgrd = dynamic_cast<const coDoPolygons *>(grid);
        polgrd->GetOctTree(otree, NULL);
        iaOctTrees_.push_back(otree);
        iaUnsGrids_.push_back(grid);
        namesToOctTrees_[string(grid->getName())] = otree;
    }
    else if (grid->isType("POLYGN"))
    {
        // octtree not available as input
        const coDoPolygons *polgrd = dynamic_cast<const coDoPolygons *>(grid);
        std::string effectiveSurname_ = octSurname_;
        char buf[256];
        sprintf(buf, "_%d", creationCounter_);
        effectiveSurname_ += buf;
        const coDistributedObject *usedOctTree = getUsedOctTree(polgrd);
        if (usedOctTree == NULL)
        {
            const coDistributedObject *octTree = polgrd->GetOctTree(NULL, effectiveSurname_.c_str());
            iaOctTrees_.push_back(octTree);
            iaUnsGrids_.push_back(grid);
            namesToOctTrees_[string(grid->getName())] = octTree;
        }
        else
        {
            polgrd->GetOctTree(usedOctTree, NULL);
            iaOctTrees_.push_back(usedOctTree);
            iaUnsGrids_.push_back(grid);
            namesToOctTrees_[string(grid->getName())] = usedOctTree;
        }
    }
}

const coDistributedObject *
BBoxAdmin::getUsedOctTree(const coDistributedObject *grd)
{
    int grid;
    const coDistributedObject *ret = NULL;
    for (grid = 0; grid < iaUnsGrids_.size(); ++grid)
    {
        if (strcmp(grd->getName(), iaUnsGrids_[grid]->getName()) == 0)
        {
            if (grd->isType("UNSGRD"))
            {
                ret = ((const coDoUnstructuredGrid *)iaUnsGrids_[grid])->GetOctTree(NULL, NULL);
            }
            else if (grd->isType("POLYGN"))
            {
                ret = ((const coDoPolygons *)iaUnsGrids_[grid])->GetOctTree(NULL, NULL);
            }
            break;
        }
    }
    return ret;
}

// used in reload
void
BBoxAdmin::preload(const coDistributedObject *grid, const coDistributedObject *otree)
{
    if (grid->isType("SETELE"))
    {
        // calling pload recursively for all elements
        const coDoSet *set = dynamic_cast<const coDoSet *>(grid);
        int no_set_elems;
        const coDistributedObject *const *setList = set->getAllElements(&no_set_elems);
        int elem;
        if (otree == NULL)
        {
            for (elem = 0; elem < no_set_elems; ++elem)
            {
                preload(setList[elem], NULL);
#ifdef _CLEAN_UP_
                if (setList[elem]->isType("SETELE"))
                {
                    delete setList[elem];
                }
#endif
            }
        }
        else
        {
            const coDoSet *oset = dynamic_cast<const coDoSet *>(otree);
            int no_oset_elems;
            const coDistributedObject *const *osetList = oset->getAllElements(&no_oset_elems);
            for (elem = 0; elem < no_oset_elems; ++elem)
            {
                preload(setList[elem], osetList[elem]);
#ifdef _CLEAN_UP_
                if (setList[elem]->isType("SETELE"))
                {
                    delete setList[elem];
                }
                if (osetList[elem]->isType("SETELE"))
                {
                    delete osetList[elem];
                }
#endif
            }
#ifdef _CLEAN_UP_
            delete[] osetList;
#endif
        }
#ifdef _CLEAN_UP_
        delete[] setList;
#endif
    }
    else if (grid->isType("UNSGRD") && otree != NULL && otree->isType("OCTREE"))
    {
        // octtree available as input
        const coDoUnstructuredGrid *unsgrd = dynamic_cast<const coDoUnstructuredGrid *>(grid);
        unsgrd->GetOctTree(otree, NULL);
        iaOctTrees_.push_back(otree);
        iaUnsGrids_.push_back(unsgrd);
        namesToOctTrees_[string(grid->getName())] = otree;
    }
    else if (grid->isType("UNSGRD"))
    {
        // octtree not available as input
        const coDoUnstructuredGrid *unsgrd = dynamic_cast<const coDoUnstructuredGrid *>(grid);
        unsgrd->GetOctTree(iaOctTrees_[gridSeq_], NULL);
        ++gridSeq_;
    }
    else if (grid->isType("POLYGN") && otree != NULL && otree->isType("OCTREP"))
    {
        // octtree available as input
        const coDoPolygons *polgrd = dynamic_cast<const coDoPolygons *>(grid);
        polgrd->GetOctTree(otree, NULL);
        iaOctTrees_.push_back(otree);
        iaUnsGrids_.push_back(grid);
        namesToOctTrees_[string(grid->getName())] = otree;
    }
    else if (grid->isType("POLYGN"))
    {
        // octtree not available as input
        const coDoPolygons *polgrd = dynamic_cast<const coDoPolygons *>(grid);
        polgrd->GetOctTree(iaOctTrees_[gridSeq_], NULL);
        ++gridSeq_;
    }
}

void
BBoxAdmin::assignOctTrees(std::vector<std::vector<const coDistributedObject *> > grid_tstep_opt) const
{
    map<string, const coDistributedObject *>::const_iterator itEnd = namesToOctTrees_.end();
    for (int time = 0; time < grid_tstep_opt.size(); ++time)
    {
        for (int i = 0; i < grid_tstep_opt[time].size(); ++i)
        {
            const coDistributedObject *grid = grid_tstep_opt[time][i];
            map<string, const coDistributedObject *>::const_iterator it = namesToOctTrees_.find(grid->getName());
            if (it != itEnd)
            {
                const coDistributedObject *otree = it->second;
                if (grid->isType("UNSGRD"))
                {
                    coDoUnstructuredGrid *uns_grid = (coDoUnstructuredGrid *)grid;
                    uns_grid->GetOctTree(otree, NULL);
                }
                else if (grid->isType("POLYGN"))
                {
                    coDoPolygons *pol_grid = (coDoPolygons *)grid;
                    pol_grid->GetOctTree(otree, NULL);
                }
            }
        }
    }
}
