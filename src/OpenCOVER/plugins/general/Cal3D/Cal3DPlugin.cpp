/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#ifdef HAVE_CONFIG_H
#undef HAVE_CONFIG_H
#endif

#include "Cal3DPlugin.h"
#include <cal3d/cal3d.h>

#include <osgDB/ReadFile>
#include <osgDB/FileNameUtils>
#include "../Vrml97/ViewerOsg.h"

Cal3DPlugin *Cal3DPlugin::plugin = NULL;

#include <cover/coVRTui.h>
#include <cover/coVRConfig.h>
#include <cover/VRSceneGraph.h>
#ifdef WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

static VrmlNode *creatorCore(VrmlScene *scene)
{
    return new Cal3dCore(scene);
}

void Cal3dCore::loadCore(const char *fn)
{
    coreModel = new osgCal::CoreModel();
    animNum = -1;
    meshAdder = new osgCal::DefaultMeshAdder;
    p = new osgCal::MeshParameters;
    p->useDepthFirstMesh = false;
    p->software = false; // default

    try
    {
        std::string ext = osgDB::getLowerCaseFileExtension(fn);
        std::string dir = osgDB::getFilePath(fn);
        std::string name = osgDB::getStrippedName(fn);

        if (dir == "")
        {
            dir = ".";
        }

        if (ext == "caf")
        {
            coreModel->load(dir + "/cal3d.cfg", p.get());

            for (size_t i = 0; i < coreModel->getAnimationNames().size(); i++)
            {
                if (coreModel->getAnimationNames()[i] == name)
                {
                    animNum = i;
                    break;
                }
            }

            if (animNum == -1)
            {
                // animation is absent in cal3d.cfg, so load it manually
                CalCoreModel *cm = coreModel->getCalCoreModel();

                std::cout << coreModel->getScale() << std::endl;
                if (coreModel->getScale() != 1)
                {
                    // to eliminate scaling of the model by non-scaled animation
                    // we scale model back, load animation, and rescale one more time
                    cm->scale(1.0 / coreModel->getScale());
                }

                animNum = cm->loadCoreAnimation(fn);

                if (coreModel->getScale() != 1)
                {
                    cm->scale(coreModel->getScale());
                }
            }
        }
        else if (ext == "cmf")
        {
            coreModel->load(dir + "/cal3d.cfg", p.get());
            meshAdder = new osgCal::OneMeshAdder(osgDB::getStrippedName(fn));
        }
        else
        {
            coreModel->load(fn, p.get());
        }
    }
    catch (std::exception &e)
    {
        std::cout << "exception during load:" << std::endl
                  << e.what() << std::endl;
        return;
    }
}

// Define the built in VrmlNodeType:: "COVER" fields

VrmlNodeType *Cal3dCore::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("Cal3DCore", creatorCore);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("modelName", VrmlField::SFSTRING);
    t->addExposedField("scale", VrmlField::SFFLOAT);

    return t;
}

VrmlNodeType *Cal3dCore::nodeType() const { return defineType(0); }

Cal3dCore::Cal3dCore(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    fprintf(stderr, "Cal3dCore::Cal3dCore\n");
    d_scale = 1.0;
}

Cal3dCore::~Cal3dCore()
{
}

VrmlNode *Cal3dCore::cloneMe() const
{
    return (vrml::VrmlNode *)this;
}

Cal3dCore *Cal3dCore::toCal3dCore() const
{
    return (Cal3dCore *)this;
}

void Cal3dCore::addToScene(VrmlScene *s, const char *)
{
    d_scene = s;
}

ostream &Cal3dCore::printFields(ostream &os, int indent)
{
    (void)indent;
    return os;
}

void Cal3dCore::eventIn(double timeStamp,
                        const char *eventName,
                        const VrmlField *fieldValue)
{
    if ((strcmp(eventName, "set_modelName") == 0) || (strcmp(eventName, "modelName") == 0))
    {
        loadCore(d_modelName.get());
    }
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }
}

// Set the value of one of the node fields.

void Cal3dCore::setField(const char *fieldName,
                         const VrmlField &fieldValue)
{
    if
        TRY_FIELD(modelName, SFString)
    else if
        TRY_FIELD(modelName, SFString)
    else if
        TRY_FIELD(scale, SFFloat)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
    if (strcmp("modelName", fieldName) == 0)
    {
        loadCore(d_modelName.get());
    }
}

static VrmlNode *creator(VrmlScene *scene)
{
    return new Cal3dNode(scene);
}

void Cal3dNode::loadModel(Cal3dCore *core)
{
    try
    {
      model->load(core->getCoreModel(), core->getMeshAdder());
    }
    catch (std::exception &e)
    {
        std::cout << "exception during load:" << std::endl
                  << e.what() << std::endl;
        myTransform->setMatrix(osg::Matrix::identity());
        return;
    }
    osg::Matrix mat;
    mat.makeScale(core->getScale(), core->getScale(), core->getScale());
    myTransform->setMatrix(mat * osg::Matrix::rotate(-M_PI / 2.0, 1, 0, 0));
}

// Define the built in VrmlNodeType:: "COVER" fields

VrmlNodeType *Cal3dNode::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("Cal3DNode", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("core", VrmlField::SFNODE);
    t->addExposedField("materialSet", VrmlField::SFINT32);
    t->addEventIn("startCycle", VrmlField::SFTIME);
    t->addEventIn("stopCycle", VrmlField::SFTIME);
    t->addEventIn("startAction", VrmlField::SFTIME);
    t->addExposedField("executeAction", VrmlField::SFINT32);
    t->addExposedField("animationId", VrmlField::SFINT32);
    t->addExposedField("animationOffset", VrmlField::SFFLOAT);
    t->addExposedField("animationWeight", VrmlField::SFFLOAT);
    t->addExposedField("animationBlendTime", VrmlField::SFFLOAT);
    t->addExposedField("fadeInTime", VrmlField::SFFLOAT);
    t->addExposedField("fadeOutTime", VrmlField::SFFLOAT);

    return t;
}

VrmlNodeType *Cal3dNode::nodeType() const { return defineType(0); }

Cal3dNode::Cal3dNode(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    fprintf(stderr, "Cal3dNode::Cal3dNode\n");
    model = new osgCal::Model();
    myTransform = new osg::MatrixTransform;
    myTransform->setMatrix(osg::Matrix::rotate(-M_PI / 2.0, 1, 0, 0));
    myTransform->addChild(model);
	model->setAutoUpdate(false); // update callbacks are disabled for performance reasons in VRML loaded scenes 
    d_materialSet = 0;
    d_animationId = -1;
    d_executeAction = -1;
    d_animationOffset = 0.0;
    d_animationWeight = 1.0;
    d_animationBlendTime = 1.0;
    d_fadeInTime = 1.0;
    d_fadeOutTime = 1.0;
    d_viewerObject = 0;
    currentAnimation = -1;
	Cal3DPlugin::instance()->nodes.push_back(this);
}

Cal3dNode::~Cal3dNode()
{
}

VrmlNode *Cal3dNode::cloneMe() const
{
    return new Cal3dNode(*this);
}

void Cal3dNode::render(Viewer *v)
{
    ViewerOsg *viewer = (ViewerOsg *)v;

    if (d_viewerObject)
        viewer->insertReference(d_viewerObject);
    else
    {
        d_viewerObject = viewer->beginObject(name(), 0, this);
        viewer->insertNode(myTransform.get());
        viewer->endObject();
    }

    clearModified();
}

Cal3dNode *Cal3dNode::toCal3dNode() const
{
    return (Cal3dNode *)this;
}

void Cal3dNode::addToScene(VrmlScene *s, const char *)
{
    d_scene = s;
}

ostream &Cal3dNode::printFields(ostream &os, int indent)
{
    (void)indent;
    return os;
}

void Cal3dNode::eventIn(double timeStamp,
                        const char *eventName,
                        const VrmlField *fieldValue)
{

    VrmlNode::eventIn(timeStamp, eventName, fieldValue);

    if ((strcmp(eventName, "set_core") == 0) || (strcmp(eventName, "core") == 0))
    {
        loadModel((Cal3dCore *)d_core.get());
    }
    /* else if ((strcmp(eventName, "set_materialSet") == 0)|| (strcmp(eventName, "materialSet") == 0))
   {
      myNode->materialSet = d_materialSet.get();
   }*/
    else if ((strcmp(eventName, "set_executeAction") == 0) || (strcmp(eventName, "executeAction") == 0))
    {
        model->executeAction(d_executeAction.get(), d_fadeInTime.get(), d_fadeOutTime.get());
    }
    else if ((strcmp(eventName, "set_animationId") == 0) || (strcmp(eventName, "animationId") == 0))
    {
        if (currentAnimation != -1)
        {
            model->clearCycle(currentAnimation, d_animationBlendTime.get()); // clear in 1sec
        }
        currentAnimation = d_animationId.get();
        if (currentAnimation == -1)
        {
            model->clearCycle(currentAnimation, 0.0); // clear now
        }
        model->blendCycle(currentAnimation, d_animationWeight.get(), d_animationBlendTime.get());
    }
    else if ((strcmp(eventName, "set_animationOffset") == 0) || (strcmp(eventName, "animationOffset") == 0))
    {
        osgCal::ModelData *md = (osgCal::ModelData *)model->getModelData();
        md->getCalMixer()->updateAnimation(d_animationOffset.get());
    }
}

// Set the value of one of the node fields.

void Cal3dNode::setField(const char *fieldName,
                         const VrmlField &fieldValue)
{
    if
        TRY_FIELD(core, SFNode)
    else if
        TRY_FIELD(materialSet, SFInt)
    else if
        TRY_FIELD(animationId, SFInt)
    else if
        TRY_FIELD(executeAction, SFInt)
    else if
        TRY_FIELD(animationWeight, SFFloat)
    else if
        TRY_FIELD(animationOffset, SFFloat)
    else if
        TRY_FIELD(animationBlendTime, SFFloat)
    else if
        TRY_FIELD(fadeInTime, SFFloat)
    else if
        TRY_FIELD(fadeOutTime, SFFloat)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
    if (strcmp("core", fieldName) == 0)
    {
        loadModel((Cal3dCore *)d_core.get());
    }
    /* else if ((strcmp(fieldName, "materialSet") == 0))
   {
      myNode->materialSet = d_materialSet.get();
   }*/
    else if ((strcmp(fieldName, "executeAction") == 0))
    {
        model->executeAction(d_executeAction.get(), d_fadeInTime.get(), d_fadeOutTime.get());
    }
    else if ((strcmp(fieldName, "animationId") == 0))
    {
        if (currentAnimation != -1)
        {
            model->clearCycle(currentAnimation, d_animationBlendTime.get()); // clear in 1sec
        }
        currentAnimation = d_animationId.get();
        if (currentAnimation == -1)
        {
            model->clearCycle(currentAnimation, 0.0); // clear now
        }
        model->blendCycle(currentAnimation, d_animationWeight.get(), d_animationBlendTime.get());
    }
    else if ((strcmp(fieldName, "animationOffset") == 0))
    {
        osgCal::ModelData *md = (osgCal::ModelData *)model->getModelData();
        md->getCalMixer()->updateAnimation(d_animationOffset.get());
    }
}

Cal3DPlugin::Cal3DPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "Cal3DPlugin::Cal3DPlugin\n");
    plugin = this;
	oldT = cover->frameTime();
}
/*
void Cal3DPlugin::tabletEvent(coTUIElement* tUIItem)
{
}

void Cal3DPlugin::tabletPressEvent(coTUIElement* )
{
}*/
bool
Cal3DPlugin::update()
{
	for (auto it = nodes.begin(); it != nodes.end(); it++)
	{
		(*it)->update(cover->frameDuration());
	}
	return true;
}

// this is called if the plugin is removed at runtime
Cal3DPlugin::~Cal3DPlugin()
{
    fprintf(stderr, "Cal3DPlugin::~Cal3DPlugin\n");
}

bool Cal3DPlugin::init()
{
    VrmlNamespace::addBuiltIn(Cal3dCore::defineType());
    VrmlNamespace::addBuiltIn(Cal3dNode::defineType());
    return true;
}

void
Cal3DPlugin::preFrame()
{
}

COVERPLUGIN(Cal3DPlugin);
