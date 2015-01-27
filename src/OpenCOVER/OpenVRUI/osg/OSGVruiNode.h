/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OSG_VRUI_NODE_H
#define OSG_VRUI_NODE_H

#include <osg/Group>
#include <OpenVRUI/sginterface/vruiNode.h>
#include <OpenVRUI/sginterface/vruiMatrix.h>

#include <string>

namespace vrui
{

class OSGVRUIEXPORT OSGVruiNode : public virtual vruiNode
{

public:
    OSGVruiNode(osg::Node *node);
    virtual ~OSGVruiNode();

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

    osg::Node *getNodePtr();

private:
    osg::ref_ptr<osg::Node> node;
    OSGVruiNode *parent;
};
}
#endif
