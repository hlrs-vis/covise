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
#include <net/tokenbuffer.h>
#include <osg/ref_ptr>
#include <osg/Matrix>
#include "MatrixSerializer.h"
namespace osg
{
class Node;
class Group;
class MatrixTransform;
}

namespace opencover
{
class coVRPartner;


class COVEREXPORT VRAvatar
{
public:
    bool initialized = false;
    osg::MatrixTransform *handTransform;
    osg::Node *handNode;
    osg::MatrixTransform *headTransform;
    osg::Node *brilleNode;
    osg::MatrixTransform *feetTransform;
    osg::Node *schuheNode;
    osg::Node *hostIconNode;
    osg::ref_ptr<osg::Group> avatarNodes;
    ///create an Avatar that only holds the local tramsfom matrices 
    ///Used to collect the data to send it to the partners
    VRAvatar();

    /// initalize avatar if not initialized. Return true if sth. is done
    virtual ~VRAvatar();
    void show();
    void hide();
protected:
    //to skip initialization
    VRAvatar(int dummy){};
    bool init(const std::string &nodeName);
};

///create an Avatar for a remote partner that hold his informations
class COVEREXPORT PartnerAvatar : public VRAvatar
{
public:
    PartnerAvatar(coVRPartner *partner);
    bool init(const std::string &hostAdress);
private:
    coVRPartner *m_partner;
};

///Create an avatar that represents a recorded movement
class COVEREXPORT RecordedAvatar : public VRAvatar
{
public:
    RecordedAvatar();
    bool init();

private:
    const std::string m_icon;
    std::vector<osg::Matrix> m_hand;
    std::vector<osg::Matrix> m_head;
    std::vector<osg::Matrix> m_feet;
};

COVEREXPORT covise::TokenBuffer &operator<<(covise::TokenBuffer &tb, const opencover::VRAvatar &avatar);
COVEREXPORT covise::TokenBuffer &operator>>(covise::TokenBuffer &tb, opencover::VRAvatar &avatar);
}
#endif
