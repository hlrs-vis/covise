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

namespace osg
{
class Node;
class Group;
class MatrixTransform;
}

namespace opencover
{
class COVEREXPORT VRAvatarData
{
public:
    // all data is in object Coordinates
    float handMat[4][4];
    float headMat[4][4];
    float feetMat[4][4];
    VRAvatarData();
    VRAvatarData(const char *buf);
    void convert();
};

class COVEREXPORT VRAvatar
{
private:
    static int num;
    static float rc[10];
    static float gc[10];
    static float bc[10];

public:
    char *hostname;
    int thisnum;
    osg::MatrixTransform *handTransform;
    osg::Node *handNode;
    osg::MatrixTransform *brilleTransform;
    osg::Node *brilleNode;
    osg::MatrixTransform *schuheTransform;
    osg::Node *schuheNode;
    osg::Node *hostIconNode;
    osg::ref_ptr<osg::Group> avatarNodes;
    VRAvatar(const char *name);
    virtual ~VRAvatar();
    void show();
    void hide();
    //osg::Node *genNode();
    void updateData(VRAvatarData &ad);
};

class COVEREXPORT VRAvatarList
{
private:
    VRAvatarList();
    static VRAvatarList *s_instance;
    typedef std::vector<VRAvatar *> Avatars;
    Avatars avatars;
    bool visible;

public:
    ~VRAvatarList();
    static VRAvatarList *instance();
    void receiveMessage(const char *messageData);
    void sendMessage();
    VRAvatar *get(const char *name);
    void add(VRAvatar *a);
    void remove(VRAvatar *a);
    void show();
    void hide();
    bool isVisible();
    size_t getNum() const
    {
        return (int)avatars.size();
    }
    VRAvatar *getAvatar(size_t index)
    {
        return avatars[index];
    }
};
}
#endif
