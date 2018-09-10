// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VV_BSPTREE_H
#define VV_BSPTREE_H

#include "math/math.h"

#include "vvvisitor.h"
#include "vvinttypes.h"

#include <vector>

class vvVisitor;

class vvBspNode : public vvVisitable
{
public:

    typedef virvo::basic_aabb< ssize_t > box_type;

  vvBspNode(box_type const& aabb);
  virtual ~vvBspNode();

  virtual void accept(vvVisitor* visitor);

  void addChild(vvBspNode* child);
  bool isLeaf() const;
  void setId(size_t id);
  void setAabb(box_type const& aabb);

  size_t getId() const;
  vvBspNode* getChildLeft() const;
  vvBspNode* getChildRight() const;
  box_type const& getAabb() const;

  void clipProbe(virvo::vec3& probeMin, virvo::vec3& probeMax,
                 virvo::vec3& probePosObj, virvo::vec3& probeSizeObj) const;
private:
  size_t _id;
  vvBspNode* _childLeft;
  vvBspNode* _childRight;

  box_type _aabb;
};

/*! \brief Data passed to bsp-tree ctor
 */
struct vvBspData
{
  vvBspData()
    : numLeafs(0)
  {

  }

  size_t numLeafs;
  std::vector<float> loadBalance;
};

class vvBspTree
{
public:

  typedef vvBspNode::box_type box_type;

  vvBspTree(virvo::vector< 3, ssize_t > const& volsize, const vvBspData& data);
  virtual ~vvBspTree();

  void traverse(virvo::vector< 3, ssize_t > const& pos) const;

  const std::vector<vvBspNode*>& getLeafs() const;

  void setVisitor(vvVisitor* visitor);
private:
  std::vector<vvBspNode*> _leafs;
  vvBspNode* _root;
  vvVisitor* _visitor;
  vvBspData _data;

  void buildHierarchy(vvBspNode* node, size_t leafIdx);

  /*!
   by example: load balance == { 0.5, 0.4, 0.1 }
   then the relative fractions are: {0.5, 0.8, 1.0 }
   [0] := 50% of total volume
   [1] := 40% of remaining fraction (0.4 + 0.1) == 0.8
   [2] := 100% of remaining fraction (0.1) == 1.0
  */
  float calcRelativeFraction(size_t leafIdx);
  void traverse(virvo::vector< 3, ssize_t > const& pos, vvBspNode* node) const;
};

#endif // VV_BSPTREE_H
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
