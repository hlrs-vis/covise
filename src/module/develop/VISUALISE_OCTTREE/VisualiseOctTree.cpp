/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "VisualiseOctTree.h"
#include <covise/covise_octtree.h>

int main(int argc, char *argv[])
{
    Application *application = new Application();
    application->start(argc, argv);
    return 0;
}

Application::Application()
    : coSimpleModule("Visualise Oct-tree")
{
    p_oct_ = addInputPort("Grid", "DO_OctTree|DO_OctTreeP", "octtree");
    p_lines_ = addOutputPort("Lines", "coDoLines", "lines");
}

int Application::compute(void)
{
    if (!p_oct_->getCurrentObject()->isType("OCTREE")
        && !p_oct_->getCurrentObject()->isType("OCTREP"))
    {
        p_lines_->setCurrentObject(new coDoLines(p_lines_->getObjName(), 0, 0, 0));
        return SUCCESS;
    }
    else
    {
        DO_BasisTree *octtree = (DO_BasisTree *)(p_oct_->getCurrentObject());
        ia<int> ll, cl;
        ia<float> xi, yi, zi;
        octtree->Visualise(ll, cl, xi, yi, zi);
        coDoLines *lines = new coDoLines(p_lines_->getObjName(), xi.size(),
                                         xi.getArray(), yi.getArray(), zi.getArray(),
                                         cl.size(), cl.getArray(),
                                         ll.size(), ll.getArray());
        p_lines_->setCurrentObject(lines);
    }
    return SUCCESS;
}
