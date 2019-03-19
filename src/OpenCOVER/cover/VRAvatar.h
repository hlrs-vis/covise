/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
 \brief  avatars for partners in colaborative VR sessions

 \author Uwe Woessner <woessner@hlrs.de>
 \author (C)
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date
 */

#ifndef VR_AVATAR_H
#define VR_AVATAR_H

#include <util/common.h>

#include <osg/ref_ptr>
#include <osg/Matrix>
namespace osg
{
class Node;
class Group;
class MatrixTransform;
}

namespace opencover
{
class COVEREXPORT VRAvatar
{
private:
    //static float rc[10];
    //static float gc[10];
    //static float bc[10];

public:
    int m_clientID;
    osg::MatrixTransform *handTransform;
    osg::Node *handNode;
    osg::MatrixTransform *headTransform;
    osg::Node *brilleNode;
    osg::MatrixTransform *feetTransform;
    osg::Node *schuheNode;
    osg::Node *hostIconNode;
    osg::ref_ptr<osg::Group> avatarNodes;
    VRAvatar(int clientID, const std::string &hostAdress);
    virtual ~VRAvatar();
    void show();
    void hide();
    //osg::Node *genNode();
    //void updateData(VRAvatarData &ad);
};
}
#endif
