/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//  Vrml 97 library
//  Copyright (C) 1998 Chris Morley
//
//  %W% %G%
//  ViewerPf.h
//  Class for display of VRML models using Performer.
//

#ifndef _VIEWEROBJECT_H_
#define _VIEWEROBJECT_H_

#ifdef _WIN32
#if (_MSC_VER >= 1300) && !(defined(MIDL_PASS) || defined(RC_INVOKED))
#define POINTER_64 __ptr64
#else
#define POINTER_64
#endif
#include <winsock2.h>
#include <windows.h>
#endif
#include <osg/GL>
#include <osg/Node>
#include <osg/Vec2>
#include <osg/Matrix>
#include <osg/Material>
#include <cover/coBillboard.h>
#include <util/common.h>
#include <vrml97/vrml/config.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlNode.h>

/*
#include <cover/coVRPluginSupport.h>
*/

#include <util/DLinkList.h>
/*
#include <cover/SpinMenu.h>
#include <cover/PointerTooltip.h>
*/

#include <osg/TexGen>
#include <osg/TexEnv>
#include <osg/TexMat>
#include <osg/TexEnvCombine>
#include <osg/Texture>
#include <osg/Texture2D>

class coCubeMap;
class osgViewerObject;
class ViewerOsg;
//class pfViewerBillboard;

using namespace vrml;
using namespace opencover;

typedef covise::DLinkList<osgViewerObject *> objList;
//typedef DLinkList< pfViewerBillboard* > billboardList;

class coSensiveSensor;

class VRML97COVEREXPORT textureData
{
public:
    textureData();
    ~textureData();
    int ntc;
    osg::ref_ptr<osg::Texture> texture; // need to be refptrs, otherwise we are in big trouble if geometry gets replaced
    osg::ref_ptr<osg::TexGen> texGen;
    osg::ref_ptr<osg::TexEnv> tEnv;
    osg::ref_ptr<osg::Image> texImage;
    osg::Matrix newTMat;
    unsigned char mirror; // 1 = vertical 2 = horizontal
};

#define NODE_GENERAL 0
#define NODE_IFS 1
#define NODE_LIGHT 2
class VRML97COVEREXPORT osgViewerObject
{
private:
    int level;
    int refcount;
    objList children;

public:
    ViewerOsg *viewer;
    int ntc;
    int numTextures;
    std::vector<textureData> texData;
    float aI, dC[3], eC[3], sC[3], shin, trans;
    char *modeNames;
    osgViewerObject *parent;
    osg::ref_ptr<osg::Node> pNode;
    osg::ref_ptr<osg::Node> lightedNode;
    void deref();
    void ref();
    osg::ref_ptr<osg::Material> mtl;
    osg::Matrix parentTransform;
    osgViewerObject(VrmlNode *n);
    ~osgViewerObject();
    VrmlNode *node;
    void *sensorObjectToAdd;
    osgViewerObject *getChild(VrmlNode *node);
    bool hasChild(osgViewerObject *o);
    void addChild(osgViewerObject *node);
    osgViewerObject *getParent();
    void incLevel()
    {
        level++;
    };
    int getLevel()
    {
        return level;
    };
    int haveToAdd;
    void addChildrensNodes();
    void updateMaterial();
    void updateTexGen();
    void updateTexData(int numActiveTextures);
    void updateTexture();
    void updateTMat();
    void removeChild(osgViewerObject *rmObj);
    void setRootNode(osg::Node *n);
    void setMaterial(float, float[], float[], float, float[], float);
    osg::Node *getNode()
    {
        if (billBoard.get())
            return billBoard.get();
        else
            return pNode.get();
    };
    coSensiveSensor *sensor;
    const char *MyDoc;
    osg::ref_ptr<coBillboard> billBoard;
    int whichChoice; // if this is != -2 (the default) than this is a switch node
    // and whichChoice is the number of the node to display
    std::map<int, int> choiceMap; // maps ChoiceNumber to number in actual Switch node
    int numSwitchChildren;
    int nodeType;
    coCubeMap *cubeMap;
    osg::ref_ptr<osg::Node> rootNode; // either cover->getScene, head or NULL == objectsRoot
    bool transparent;
    void updateBin(); //set draw bin of all geosets
    void setTexEnv(int environment, int textureNumber, int blendMode, int nc);
    void setTexGen(int environment, int textureNumber, int blendMode);

    static int getBlendModeForVrmlNode(const char *modeString);
    std::string name;
};
#endif
