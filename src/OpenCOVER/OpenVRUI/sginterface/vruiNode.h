/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VRUI_NODE_H
#define VRUI_NODE_H

#include <string>
#include <util/coTypes.h>
#include <string>

namespace vrui
{

class vruiMatrix;
class vruiUserData;

class OPENVRUIEXPORT vruiNode
{

public:
    vruiNode()
    {
    }
    virtual ~vruiNode()
    {
    }

    virtual void addChild(vruiNode *node) = 0;
    virtual void removeChild(vruiNode *node) = 0;
    virtual void insertChild(int location, vruiNode *node) = 0;

    virtual void removeAllParents() = 0;
    virtual void removeAllChildren() = 0;

    virtual void setName(const std::string &name) = 0;
    virtual std::string getName() const = 0;

    virtual int getNumParents() const = 0;
    virtual vruiNode *getParent(int parent = 0) = 0;

    virtual vruiUserData *getUserData(const std::string &name) = 0;
    virtual void setUserData(const std::string &name, vruiUserData *data) = 0;

    virtual void convertToWorld(vruiMatrix *matrix) = 0;
};
}
#endif
