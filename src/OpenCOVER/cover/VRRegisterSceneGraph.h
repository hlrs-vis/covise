/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VR_REGISTER_SG_H
#define VR_REGISTER_SG_H

#include <util/coExport.h>

#include <osg/Node>
#include <string>
#include <iostream>

namespace opencover
{

class COVEREXPORT VRRegisterSceneGraph
{

private:
    static VRRegisterSceneGraph *s_instance;
    int registerId;
    bool blocked;
    std::string whole_message;
    std::string transparency_message;
    std::string sceneGraphAppendixIdString;

protected:
    VRRegisterSceneGraph();

    void createRegisterMessage(osg::Node *node, std::string parent);
    void createUnregisterMessage(osg::Node *node, std::string parent);
    void createTransparencyMessage(osg::Node *node);
    void sendRegisterMessage();
    void sendUnregisterMessage();
    void sendTransparencyMessage();

    std::string createName(std::string);

public:
    virtual ~VRRegisterSceneGraph();
    static VRRegisterSceneGraph *instance();

    void setRegisterStartIndex(int startIndex);

    void registerNode(osg::Node *node, std::string parent);
    void unregisterNode(osg::Node *node, std::string parent);

    void block()
    {
        blocked = true;
    };
    void unblock()
    {
        blocked = false;
    };
    bool isBlocked()
    {
        return blocked;
    };
};
}
#endif
