/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// **************************************************************************
//
// Description:   Count num of Geodes in the subtree
//
// Author:        Philip Weber
//
// Creation Date: 2006-02-29
//
// **************************************************************************

#ifndef GEODECOUNT_VISITOR_H
#define GEODECOUNT_VISITOR_H

#include <osg/NodeVisitor>

class GeodeCountVisitor : public osg::NodeVisitor
{
private:
    void printNumGeodes();
    unsigned _total;

public:
    GeodeCountVisitor();
    virtual void apply(osg::Geode &);
};
#endif
