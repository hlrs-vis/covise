/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_UTIL_NODEDUMPVISITOR_H
#define VRUI_UTIL_NODEDUMPVISITOR_H

#include <util/coTypes.h>

#include <osg/NodeVisitor>
#include <osg/Node>

namespace covise
{

class OSGVRUIEXPORT NodeDumpVisitor : public osg::NodeVisitor
{
public:
    NodeDumpVisitor();

    /** Simply traverse using standard NodeVisitor traverse method.*/
    virtual void apply(osg::Node &node);

private:
    int level;
};
}
#endif
