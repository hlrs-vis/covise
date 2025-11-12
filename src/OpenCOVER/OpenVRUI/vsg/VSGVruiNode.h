/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#pragma once

#include <vsg/nodes/Group.h>
#include <OpenVRUI/sginterface/vruiNode.h>
#include <OpenVRUI/sginterface/vruiMatrix.h>

#include <string>

namespace vrui
{

class VSGVRUIEXPORT VSGVruiNode : public virtual vruiNode
{

public:
    VSGVruiNode(vsg::ref_ptr<vsg::Node> node);
    virtual ~VSGVruiNode();

    virtual void addChild(vruiNode *node);
    virtual void removeChild(vruiNode *node);
    virtual void insertChild(int location, vruiNode *node);

    virtual void removeAllParents();
    virtual void removeAllChildren();

    virtual void setName(const std::string &name);
    virtual std::string getName() const;

    virtual int getNumParents() const;
    virtual vruiNode *getParent(int parent = 0);

    virtual void convertToWorld(vruiMatrix *matrix);

    virtual vruiUserData *getUserData(const std::string &name);
    virtual void setUserData(const std::string &name, vruiUserData *data);

    vsg::Node *getNodePtr();
    vsg::ref_ptr<vsg::Node> node;

    void setNodePath(std::vector<const vsg::Node*> hitNodePath);

private:
    VSGVruiNode *parent;
    std::vector<const vsg::Node*> nodePath;
};
}
