/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CUTRACE_H
#define CUTRACE_H

#include <osg/GraphicsThread>

extern osg::Group *init_cuda(const char *grid, const char *data);

class Cuda : public osg::GraphicsOperation
{

public:
    Cuda(const char *gridName, const char *dataName)
        : osg::GraphicsOperation("cuda", false)
        , root(NULL)
        , grid(gridName)
        , data(dataName)
    {
    }

    virtual void operator()(osg::GraphicsContext *gc)
    {
        root = init_cuda(grid, data);
    }

    osg::Group *root;

    const char *grid;
    const char *data;
};

#endif
