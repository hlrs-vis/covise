/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <util/coTypes.h>

#include <vsg/core/Visitor.h>
#include <vsg/core/Inherit.h>

namespace vrui
{
class VSGVRUIEXPORT NodeDumpVisitor : public vsg::Inherit<vsg::Visitor, NodeDumpVisitor>
{
public:
    NodeDumpVisitor();

    /** Simply traverse using standard NodeVisitor traverse method.*/
    virtual void apply(vsg::Node &node);

private:
    int level;
};

}

EVSG_type_name(vrui::NodeDumpVisitor);
