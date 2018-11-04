/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _FieldOfView_NODE_PLUGIN_H
#define _FieldOfView_NODE_PLUGIN_H

#include <util/common.h>


#include <OpenVRUI/sginterface/vruiActionUserData.h>

#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenu.h>
//#include <cover/coTabletUI.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <osg/Group>
#include <osg/Material>

#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>

#include <cover/coVRMSController.h>
#include <cover/coVRPluginSupport.h>

#include <config/CoviseConfig.h>

#include <util/coTypes.h>

#include <vrml97/vrml/config.h>
#include <vrml97/vrml/VrmlNodeType.h>
#include <vrml97/vrml/coEventQueue.h>
#include <vrml97/vrml/MathUtils.h>
#include <vrml97/vrml/System.h>
#include <vrml97/vrml/Viewer.h>
#include <vrml97/vrml/VrmlScene.h>
#include <vrml97/vrml/VrmlNamespace.h>
#include <vrml97/vrml/VrmlNode.h>
#include <vrml97/vrml/VrmlSFBool.h>
#include <vrml97/vrml/VrmlSFFloat.h>
#include <vrml97/vrml/VrmlSFNode.h>
#include <vrml97/vrml/VrmlSFInt.h>
#include <vrml97/vrml/VrmlSFColor.h>
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

using namespace vrml;
using namespace opencover;

class PLUGINEXPORT FieldOfViewNode : public VrmlNodeChild
{

public:
    // Define the built in VrmlNodeType:: "FieldOfView"
    static VrmlNodeType *defineType(VrmlNodeType *t = 0);
    virtual VrmlNodeType *nodeType() const;

    FieldOfViewNode(VrmlScene *scene);
    virtual ~FieldOfViewNode();

    virtual VrmlNode *cloneMe() const;

    virtual FieldOfViewNode *toFieldOfViewNode() const;

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual ostream &printFields(ostream &os, int indent);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual void setField(const char *fieldName, const VrmlField &fieldValue);

    virtual void render(Viewer *);
    void update();
    void setCoordinates();


private:

    int nseg = 50;
    float coneLength = 10000;
    osg::Vec3Array *vert;
    osg::Geometry *geom;

    osg::Material *material;

    osg::ref_ptr<osg::MatrixTransform> myTransform;
    Viewer::Object d_viewerObject; // move to VrmlNode.h ? ...
    VrmlSFBool d_enabled;
    VrmlSFFloat d_fieldOfView;
    VrmlSFFloat d_transparency;
    VrmlSFColor d_color;
};

class FieldOfView : public coVRPlugin
{
public:
    FieldOfView();
    ~FieldOfView();

	static FieldOfView *instance() { return plugin; };

    // this will be called in PreFrame
    void preFrame();
	bool update();
    bool init();
	std::list<FieldOfViewNode *> nodes;

private:
	static FieldOfView *plugin;
};
#endif
