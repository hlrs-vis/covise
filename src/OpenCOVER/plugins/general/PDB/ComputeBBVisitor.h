/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _BB_H
#define _BB_H

#include <osg/Geode>
#include <osg/Node>
#include <osg/Matrix>
#include <osg/Transform>
#include <osg/BoundingBox>
#include <osg/MatrixTransform>

class ComputeBBVisitor : public osg::NodeVisitor
{
private:
    osg::BoundingBox m_bb;
    osg::Matrix m_curMatrix;

public:
    ComputeBBVisitor(const osg::Matrix &);
    osg::BoundingBox getBound();
    virtual void apply(osg::Transform &);
    virtual void apply(osg::Geode &);
    virtual void apply(osg::Node &);
};
#endif
