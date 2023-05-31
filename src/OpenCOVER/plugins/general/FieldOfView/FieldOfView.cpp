/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#ifdef HAVE_CONFIG_H
#undef HAVE_CONFIG_H
#endif

#include "FieldOfView.h"

#include <osgDB/ReadFile>
#include <osgDB/FileNameUtils>
#include "../Vrml97/ViewerOsg.h"

FieldOfView *FieldOfView::plugin = NULL;

#include <cover/coVRConfig.h>
#include <cover/VRSceneGraph.h>
#ifdef WIN32
#include <sys/timeb.h>
#else
#include <sys/time.h>
#endif

static VrmlNode *creator(VrmlScene *scene)
{
    return new FieldOfViewNode(scene);
}


// Define the built in VrmlNodeType:: "COVER" fields

VrmlNodeType *FieldOfViewNode::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st;
        t = st = new VrmlNodeType("FieldOfView", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addExposedField("enabled", VrmlField::SFBOOL);
    t->addExposedField("transparency", VrmlField::SFFLOAT);
    t->addExposedField("fieldOfView", VrmlField::SFFLOAT);
    t->addExposedField("color", VrmlField::SFCOLOR);

    return t;
}

VrmlNodeType *FieldOfViewNode::nodeType() const { return defineType(0); }

FieldOfViewNode::FieldOfViewNode(VrmlScene *scene)
    : VrmlNodeChild(scene)
{
    fprintf(stderr, "FieldOfView::FieldOfView\n");

    d_enabled = true;
    d_color.set(1, 0, 0);
    d_transparency = 0.0;
    d_fieldOfView = 0.6;
    d_viewerObject = 0;

    myTransform = new osg::MatrixTransform;
    myTransform->setMatrix(cover->getViewerMat());
    myTransform->setName("FOV_Trans");
    cover->getScene()->addChild(myTransform);
    osg::Geode *geode = new osg::Geode();
    geode->setName("fovCone");
    geom = new osg::Geometry();
    cover->setRenderStrategy(geom);

    //cover->getObjectsRoot()->addChild(myTransform);
    vert = new osg::Vec3Array;
    setCoordinates();
    osg::DrawElementsUInt *primitives = new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLES);
    for (int i = 0; i < nseg; i++)
    {
        primitives->push_back(i);
        if (i + 1 == nseg)
        {
            primitives->push_back(0);
            primitives->push_back(i + nseg);
            primitives->push_back(i + nseg);
            primitives->push_back(0);
            primitives->push_back(nseg);
        }
        else
        {
            primitives->push_back(i + 1);
            primitives->push_back(i + nseg);
            primitives->push_back(i + nseg);
            primitives->push_back(i + 1);
            primitives->push_back(i + nseg + 1);
        }
    }
    geom->addPrimitiveSet(primitives);
    osg::StateSet *geoState = geode->getOrCreateStateSet();
    material = new osg::Material;
    material->setColorMode(osg::Material::OFF);
    material->setAmbient(osg::Material::FRONT_AND_BACK, osg::Vec4(0.2f, 0.2f, 0.2f, 1.0));
    material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.4f, 0.4f, 0.4f, 1.0));
    material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4(0.0f, 0.0f, 0.0f, 1.0));
    material->setShininess(osg::Material::FRONT_AND_BACK, 16.0f);
    material->setTransparency(osg::Material::FRONT_AND_BACK, d_transparency.get());
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(d_color.get()[0], d_color.get()[1], d_color.get()[2], 1.0));
    geoState->setAttributeAndModes(material, osg::StateAttribute::ON);
        geoState->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        geoState->setMode(GL_BLEND, osg::StateAttribute::ON);
    geoState->setNestRenderBins(false);
    geoState->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    geode->addDrawable(geom);
    myTransform->addChild(geode);
    /*
    osg::AlphaFunc *alphaFunc = new osg::AlphaFunc();
    alphaFunc->setFunction(osg::AlphaFunc::GREATER, AlphaThreshold);
    geoState->setAttributeAndModes(alphaFunc, osg::StateAttribute::ON);
    */



    //myTransform->addChild(model);
	FieldOfView::instance()->nodes.push_back(this);
}

void FieldOfViewNode::setCoordinates()
{
    vert->clear();
    float ri = coVRConfig::instance()->nearClip();
    float ra = sin(d_fieldOfView.get() / 2.0)*coneLength;
    float ya = cos(d_fieldOfView.get() / 2.0)*coneLength;
    for (int i = 0; i < nseg; ++i)
    {
        float angle = (M_PI *2.0 / (float)nseg)*i;
        vert->push_back(osg::Vec3(sin(angle)*ri, -100.0, cos(angle)*ri));
    }
    for (int i = 0; i < nseg; ++i)
    {
        float angle = (M_PI *2.0 / (float)nseg)*i;
        vert->push_back(osg::Vec3(sin(angle)*ra, ya, cos(angle)*ra));
    }
    geom->setVertexArray(vert);
#if OSG_VERSION_GREATER_OR_EQUAL(3, 6, 0)
    geom->dirtyGLObjects();
#else
    geom->dirtyDisplayList();
#endif
}

FieldOfViewNode::~FieldOfViewNode()
{
}

VrmlNode *FieldOfViewNode::cloneMe() const
{
    return new FieldOfViewNode(*this);
}

void FieldOfViewNode::render(Viewer *v)
{
    ViewerOsg *viewer = (ViewerOsg *)v;

  /*  if (d_viewerObject)
        viewer->insertReference(d_viewerObject);
    else
    {
        d_viewerObject = viewer->beginObject(name(), 0, this);
        viewer->insertNode(myTransform.get());
        viewer->endObject();
    }*/

    clearModified();
}

FieldOfViewNode *FieldOfViewNode::toFieldOfViewNode() const
{
    return (FieldOfViewNode *)this;
}

void FieldOfViewNode::addToScene(VrmlScene *s, const char *)
{
    d_scene = s;
}

ostream &FieldOfViewNode::printFields(ostream &os, int indent)
{
    (void)indent;
    return os;
}

void FieldOfViewNode::eventIn(double timeStamp,
                        const char *eventName,
                        const VrmlField *fieldValue)
{

    VrmlNode::eventIn(timeStamp, eventName, fieldValue);

    if ((strcmp(eventName, "set_transparency") == 0) || (strcmp(eventName, "transparency") == 0))
    {
        
    }

    material->setTransparency(osg::Material::FRONT_AND_BACK, d_transparency.get());
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(d_color.get()[0], d_color.get()[1], d_color.get()[2], d_transparency.get()));
    setCoordinates();
}
void FieldOfViewNode::update()
{
    myTransform->setMatrix(cover->getViewerMat());
}

// Set the value of one of the node fields.

void FieldOfViewNode::setField(const char *fieldName,
                         const VrmlField &fieldValue)
{
    if
        TRY_FIELD(enabled, SFBool)
    else if
        TRY_FIELD(transparency, SFFloat)
    else if
        TRY_FIELD(color, SFColor)
    else if
        TRY_FIELD(fieldOfView, SFFloat)
    else
        VrmlNodeChild::setField(fieldName, fieldValue);
    if (strcmp("transparency", fieldName) == 0)
    {
        //loadModel((FieldOfViewCore *)d_core.get());
    }
    setCoordinates();
    material->setTransparency(osg::Material::FRONT_AND_BACK, d_transparency.get());
    material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(d_color.get()[0], d_color.get()[1], d_color.get()[2], d_transparency.get()));
    if (d_enabled.get())
    {
        if (myTransform->getNumParents() == 0)
        {
            cover->getScene()->addChild(myTransform.get());
            fprintf(stderr, "add\n");
        }
    }
    else
    {
        if (myTransform->getNumParents() != 0)
        {
            cover->getScene()->removeChild(myTransform.get());
            fprintf(stderr, "remove\n");
        }
    }
}

FieldOfView::FieldOfView()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "FieldOfView::FieldOfView\n");
    plugin = this;
}
bool
FieldOfView::update()
{
	for (auto it = nodes.begin(); it != nodes.end(); it++)
	{
		(*it)->update();
	}
	return true;
}

// this is called if the plugin is removed at runtime
FieldOfView::~FieldOfView()
{
    fprintf(stderr, "FieldOfView::~FieldOfView\n");
}

bool FieldOfView::init()
{
    VrmlNamespace::addBuiltIn(FieldOfViewNode::defineType());
    return true;
}

void
FieldOfView::preFrame()
{
}

COVERPLUGIN(FieldOfView);
