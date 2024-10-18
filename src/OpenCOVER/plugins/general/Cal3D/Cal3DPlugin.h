/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Cal3D_NODE_PLUGIN_H
#define _Cal3D_NODE_PLUGIN_H

#include <osgCal/CoreModel>
#include <osgCal/Model>

#include <iostream>
#include <list>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>

#include <osg/Group>

#include <cover/coVRPlugin.h>

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
#include <vrml97/vrml/VrmlMFInt.h>
#include <vrml97/vrml/VrmlNodeChild.h>
#include <vrml97/vrml/VrmlScene.h>

using namespace vrml;
using namespace opencover;

class PLUGINEXPORT Cal3dCore : public VrmlNodeChild
{

public:
    // Define the built in VrmlNodeType:: "Cal3D"
    static void initFields(Cal3dCore *node, VrmlNodeType *t);
    static const char *name();

    Cal3dCore(VrmlScene *scene);

    virtual Cal3dCore *toCal3dCore() const;

    virtual void addToScene(VrmlScene *s, const char *relUrl);


    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    osgCal::CoreModel *getCoreModel()
    {
        return coreModel.get();
    };
    osgCal::BasicMeshAdder *getMeshAdder()
    {
        return meshAdder.get();
    };
    float getScale()
    {
        return d_scale.get();
    };

private:
    osg::ref_ptr<osgCal::CoreModel> coreModel;
    osg::ref_ptr<osgCal::BasicMeshAdder> meshAdder;
    osg::ref_ptr<osgCal::MeshParameters> p;
    std::vector<std::string> animationNames;
    int animNum;
    void loadCore(const char *);
    VrmlSFString d_modelName;
    VrmlSFFloat d_scale;
};

class PLUGINEXPORT Cal3dNode : public VrmlNodeChild
{

public:
    static void initFields(Cal3dNode *node, VrmlNodeType *t);
    static const char *name();

    Cal3dNode(VrmlScene *scene);

    virtual Cal3dNode *toCal3dNode() const;

    virtual void addToScene(VrmlScene *s, const char *relUrl);

    virtual void eventIn(double timeStamp,
                         const char *eventName,
                         const VrmlField *fieldValue);

    virtual void render(Viewer *);

	void update(double deltaT)
	{
		model->update(deltaT);
	};

private:
    osgCal::Model *model;
    osg::ref_ptr<osg::MatrixTransform> myTransform;
    int currentAnimation;
    void loadModel(Cal3dCore *core);
    Viewer::Object d_viewerObject; // move to VrmlNode.h ? ...
    VrmlSFNode d_core;
    VrmlSFInt d_materialSet;
    VrmlSFInt d_animationId;
    VrmlSFInt d_executeAction;
    VrmlSFFloat d_animationOffset;
    VrmlSFFloat d_animationWeight;
    VrmlSFFloat d_animationBlendTime;
    VrmlSFFloat d_fadeInTime;
    VrmlSFFloat d_fadeOutTime;
};

class Cal3DPlugin : public coVRPlugin
{
public:
    Cal3DPlugin();
    ~Cal3DPlugin();

	static Cal3DPlugin *instance() { return plugin; };

    // this will be called in PreFrame
    void preFrame();
	bool update();
    bool init();
	std::list<Cal3dNode *> nodes;

private:
	static Cal3DPlugin *plugin;
	time_t oldT;
};
#endif
