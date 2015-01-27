/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream>
#include "ComputeBBVisitor.h"
#include <osg/Version>

using namespace std;
using namespace osg;

// run from mainNode
ComputeBBVisitor::ComputeBBVisitor(const Matrix &mat)
    : osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
{
    m_curMatrix = mat;
}

void ComputeBBVisitor::apply(osg::Node &node)
{
    //cerr << "node\n";
    traverse(node);
}

/*void ComputeBBVisitor::apply(osg::Geode &geode)
  {
    cerr << "bbgeode\n";
    for(unsigned int i = 0; i < geode.getNumDrawables(); i++)
    {
      osg::BoundingBox bb = geode.getDrawable(i)->getBound();
      Vec3 vec = bb.center();
      cerr << "x: " << vec.x() << " y: " << vec.y() << " z: " << vec.z() << endl;
      m_bb.expandBy(bb.corner(0)*m_curMatrix);
      m_bb.expandBy(bb.corner(1)*m_curMatrix);
      m_bb.expandBy(bb.corner(2)*m_curMatrix);
      m_bb.expandBy(bb.corner(3)*m_curMatrix);
      m_bb.expandBy(bb.corner(4)*m_curMatrix);
      m_bb.expandBy(bb.corner(5)*m_curMatrix);
      m_bb.expandBy(bb.corner(6)*m_curMatrix);
      m_bb.expandBy(bb.corner(7)*m_curMatrix);
    }
    traverse(geode);
  }*/

void ComputeBBVisitor::apply(osg::Geode &geode)
{
    //cerr << "bbgeode\n";
    for (unsigned int i = 0; i < geode.getNumDrawables(); i++)
    {
#if OSG_VERSION_GREATER_OR_EQUAL(3, 3, 3)
        osg::BoundingBox bb = geode.getDrawable(i)->getBoundingBox();
#else
        osg::BoundingBox bb = geode.getDrawable(i)->getBound();
#endif
        //Vec3 vec = bb.center();
        //cerr << "x: " << vec.x() << " y: " << vec.y() << " z: " << vec.z() << endl;
        m_bb.expandBy(bb.corner(0) * m_curMatrix);
        m_bb.expandBy(bb.corner(1) * m_curMatrix);
        m_bb.expandBy(bb.corner(2) * m_curMatrix);
        m_bb.expandBy(bb.corner(3) * m_curMatrix);
        m_bb.expandBy(bb.corner(4) * m_curMatrix);
        m_bb.expandBy(bb.corner(5) * m_curMatrix);
        m_bb.expandBy(bb.corner(6) * m_curMatrix);
        m_bb.expandBy(bb.corner(7) * m_curMatrix);
    }
    traverse(geode);
}

/* void ComputeBBVisitor::apply(osg::Transform& node)
  {
    osg::Matrix matrix;
    node.computeLocalToWorldMatrix(matrix,this);

    osg::Matrix prevMatrix = m_curMatrix;
    m_curMatrix.preMult(matrix);

    traverse(node);

    m_curMatrix = prevMatrix;
  }*/

void ComputeBBVisitor::apply(osg::Transform &node)
{
    // osg::Matrix matrix;
    // node.computeLocalToWorldMatrix(matrix,this);
    //cerr << "bb transform\n";
    osg::Matrix prevMatrix = m_curMatrix;
    m_curMatrix.preMult(node.asMatrixTransform()->getMatrix());
    //m_curMatrix *= node.asMatrixTransform()->getMatrix();

    traverse(node);

    m_curMatrix = prevMatrix;
}

BoundingBox ComputeBBVisitor::getBound()
{
    return m_bb;
}
