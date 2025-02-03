/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "NodeDumpVisitor.h"
#include <OpenVRUI/util/vruiLog.h>
#include <vsg/all.h>

#include <string>

using namespace vsg;
using namespace std;

namespace vrui
{

    NodeDumpVisitor::NodeDumpVisitor()
    {
        level = 0;
    }

    void NodeDumpVisitor::apply(vsg::Node& node)
    {
        string sfill(2 * level, ' ');
        ++level;
        VRUILOG(sfill << "|")
        VRUILOG(sfill << "|-- Type: " << node.className())
            std::string name;
        if (node.getValue("name", name))
        {
            VRUILOG(sfill << "  | Name: " << name)
        }

        node.traverse(*this);;
        --level;
    }

}
