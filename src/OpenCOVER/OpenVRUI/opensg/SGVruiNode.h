/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SG_VRUI_NODE_H
#define SG_VRUI_NODE_H

#include <OpenSG/OSGNode.h>
#include <OpenVRUI/sginterface/vruiNode.h>

class SGVRUIEXPORT SGVruiNode : public virtual vruiNode
{

public:
    SGVruiNode(const osg::NodePtr &node);
    virtual ~SGVruiNode();

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

    osg::NodePtr &getNodePtr();

private:
    osg::NodePtr node;
    SGVruiNode *parent;
};
#endif
