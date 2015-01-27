/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <OpenVRUI/util/NodeDumpVisitor.h>
#include <OpenVRUI/util/vruiLog.h>

#include <string>

using namespace osg;
using namespace std;
using namespace covise;

NodeDumpVisitor::NodeDumpVisitor()
    : NodeVisitor(TRAVERSE_ALL_CHILDREN)
{
    level = 0;
}

void NodeDumpVisitor::apply(osg::Node &node)
{
    string sfill(2 * level, ' ');
    ++level;
    VRUILOG(sfill << "|")
    VRUILOG(sfill << "|-- Type: " << node.className())
    if (!node.getName().empty())
        VRUILOG(sfill << "  | Name: " << node.getName())
    traverse(node);
    --level;
}
