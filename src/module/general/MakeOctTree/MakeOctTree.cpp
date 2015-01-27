/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "MakeOctTree.h"
#include <do/coDoOctTree.h>
#include <do/coDoOctTreeP.h>
#include <do/coDoUnstructuredGrid.h>

MakeOctTree::MakeOctTree(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Create Octrees for UNSGRDs")
{
    p_grids_ = addInputPort("inGrid", "UnstructuredGrid|Polygons", "input grid");
    p_octtrees_ = addOutputPort("outOctTree", "OctTree|OctTreeP", "output octtree");
    p_normal_size_ = addInt32Param("normal_size", "normal size of octree population");
    p_normal_size_->setValue(coDoBasisTree::NORMAL_SIZE);
    p_max_no_levels_ = addInt32Param("max_no_levels", "Maximum number of levels in an octree");
    p_max_no_levels_->setValue(coDoBasisTree::MAX_NO_LEVELS);
    p_min_small_enough_ = addInt32Param("min_small_enough", "(minimum) normal size of leaf population");
    p_min_small_enough_->setValue(coDoBasisTree::MIN_SMALL_ENOUGH);
    p_crit_level_ = addInt32Param("crit_level", "critical level for population control");
    p_crit_level_->setValue(coDoBasisTree::CRIT_LEVEL);
    p_limit_fX_ = addInt32Param("limit_fX", "limit number of division in the X direction");
    p_limit_fX_->setValue(INT_MAX);
    p_limit_fY_ = addInt32Param("limit_fY", "limit number of division in the Y direction");
    p_limit_fY_->setValue(INT_MAX);
    p_limit_fZ_ = addInt32Param("limit_fZ", "limit number of division in the Z direction");
    p_limit_fZ_->setValue(INT_MAX);
}

MakeOctTree::~MakeOctTree()
{
}

int
MakeOctTree::compute(const char *)
{
    const coDistributedObject *grid = p_grids_->getCurrentObject();
    // check param values
    if (p_normal_size_->getValue() <= 0)
    {
        sendError("normal_size may not be <= 0");
        return FAIL;
    }
    if (p_max_no_levels_->getValue() < 0)
    {
        sendError("max_no_levels may not be < 0");
        return FAIL;
    }
    if (p_min_small_enough_->getValue() <= 0)
    {
        sendError("min_small_enough may not be <= 0");
        return FAIL;
    }
    if (p_crit_level_->getValue() > p_max_no_levels_->getValue())
    {
        sendError("crit_level may not be > max_no_levels");
        return FAIL;
    }
    if (p_limit_fX_->getValue() <= 0)
    {
        sendError("limit_fX may not be <= 0");
        return FAIL;
    }
    if (p_limit_fY_->getValue() <= 0)
    {
        sendError("limit_fY may not be <= 0");
        return FAIL;
    }
    if (p_limit_fZ_->getValue() <= 0)
    {
        sendError("limit_fZ may not be <= 0");
        return FAIL;
    }

    if (grid->isType("UNSGRD"))
    {
        const coDoUnstructuredGrid *unsgrd = dynamic_cast<const coDoUnstructuredGrid *>(grid);
        int nume, numc, nump;
        unsgrd->getGridSize(&nume, &numc, &nump);
        int *e_l, *c_l;
        float *x_l, *y_l, *z_l;
        unsgrd->getAddresses(&e_l, &c_l, &x_l, &y_l, &z_l);

        coDoOctTree *tree = new coDoOctTree(p_octtrees_->getObjName(),
                                            nume, numc, nump, e_l, c_l,
                                            x_l, y_l, z_l
                                            /*,
               p_normal_size_->getValue(),
               p_max_no_levels_->getValue(),
               p_min_small_enough_->getValue(),
               p_crit_level_->getValue(),
               p_limit_fX_->getValue(),
               p_limit_fY_->getValue(),
                  p_limit_fZ_->getValue()*/
                                            );
        p_octtrees_->setCurrentObject(tree);
    }
    else if (grid->isType("POLYGN"))
    {
        const coDoPolygons *polgrd = dynamic_cast<const coDoPolygons *>(grid);
        int nume, numc, nump;
        nume = polgrd->getNumPolygons();
        numc = polgrd->getNumVertices();
        nump = polgrd->getNumPoints();
        int *e_l, *c_l;
        float *x_l, *y_l, *z_l;
        polgrd->getAddresses(&x_l, &y_l, &z_l, &c_l, &e_l);

        coDoOctTreeP *tree = new coDoOctTreeP(p_octtrees_->getObjName(),
                                              nume, numc, nump, e_l, c_l,
                                              x_l, y_l, z_l,
                                              p_normal_size_->getValue(),
                                              p_max_no_levels_->getValue(),
                                              p_min_small_enough_->getValue(),
                                              p_crit_level_->getValue(),
                                              p_limit_fX_->getValue(),
                                              p_limit_fY_->getValue(),
                                              p_limit_fZ_->getValue());

        p_octtrees_->setCurrentObject(tree);
    }
    else
    {
        p_octtrees_->setCurrentObject(new coDoOctTree(p_octtrees_->getObjName(),
                                                      0, 0, 0, NULL, NULL, NULL, NULL, NULL));
    }
    return SUCCESS;
}

MODULE_MAIN(Tools, MakeOctTree)
