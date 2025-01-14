/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "vvOnscreenDebug.h"

#include "vvViewer.h"
#include "vvPluginSupport.h"
#include "vvFileManager.h"
#include <iostream>
#include <config/CoviseConfig.h>
#include <vsg/nodes/MatrixTransform.h>

using namespace vsg;
using namespace covise;
using namespace vive;

vvOnscreenDebug *vvOnscreenDebug::singleton = 0;

vvOnscreenDebug *vvOnscreenDebug::instance()
{
    if (!singleton)
        singleton = new vvOnscreenDebug();
    return singleton;
}

vvOnscreenDebug::vvOnscreenDebug()
{
    visible = true;

  /*  geode = new osg::Geode();

    std::string timesFont = vvFileManager::instance()->getFontFile(NULL);

    // turn lighting off for the text and disable depth test to ensure its always ontop.
    osg::StateSet *stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    vsg::vec3 position(320.0f, 200.0f, 0.2f);
    vsg::vec3 delta(0.0f, -40.0f, 0.0f);

    text = new osgText::Text;
    geode->addDrawable(text.get());

    text->setFont(timesFont);
    text->setPosition(position);
    text->setText("debug...", osgText::String::ENCODING_UTF8);
    text->setCharacterSize(20);
    text->setAlignment(osgText::Text::CENTER_BOTTOM_BASE_LINE);
    position += delta;

    {
        osg::BoundingBox bb;
        for (unsigned int i = 0; i < geode->getNumDrawables(); ++i)
        {
            bb.expandBy(geode->getDrawable(i)->getBound());
        }
        vsg::Node *geom = new vsg::Node;

        vsg::vec3Array *vertices = new vsg::vec3Array;
        vertices->push_back(vsg::vec3(0, 480, 0.001));
        vertices->push_back(vsg::vec3(0, 0, 0.001));
        vertices->push_back(vsg::vec3(640, 0, 0.001));
        vertices->push_back(vsg::vec3(640, 480, 0.001));
        geom->setVertexArray(vertices);

        vsg::vec3Array *normals = new vsg::vec3Array;
        normals->push_back(vsg::vec3(0.0f, 0.0f, 1.0f));
        geom->setNormalArray(normals);
        geom->setNormalBinding(vsg::Node::BIND_OVERALL);

        vsg::vec4Array *colors = new vsg::vec4Array;
        colors->push_back(vsg::vec4(0.4f, 0.4f, 0.4f, 0.25f));
        geom->setColorArray(colors);
        geom->setColorBinding(vsg::Node::BIND_OVERALL);

        geom->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));

        osg::StateSet *stateset = geom->getOrCreateStateSet();

        stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
        //stateset->setAttribute(new osg::PolygonOffset(1.0f,1.0f),osg::StateAttribute::ON);
        stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        stateset->setNestRenderBins(false);

        geode->addDrawable(geom);
    }

    camera = new osg::Camera;
    camera->setName("On-screen Debug");

    camera->setProjectionMatrix(vsg::dmat4::ortho2D(0, 1024, 0, 768));
    camera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
    // set the view matrix
    camera->setReferenceFrame(vsg::MatrixTransform::ABSOLUTE_RF);
    camera->setViewMatrix(vsg::dmat4::translate(vsg::vec3(0, 0, 100)));
    camera->setViewMatrix(vsg::dmat4::identity());

    // only clear the depth buffer
    camera->setClearMask(GL_DEPTH_BUFFER_BIT);

    // draw subgraph after main camera view.
    camera->setRenderOrder(osg::Camera::POST_RENDER);

    vsg::MatrixTransform *m = vsg::MatrixTransform::create();
    m->addChild(geode.get());
    m->matrix = (vsg::dmat4::translate((1024 - 640) / 2.0, (768 - 480) / 2.0, 0));
    camera->addChild(m);

    vv->getScene()->addChild(camera.get());*/
}

void vvOnscreenDebug::redraw()
{
    vvViewer::instance()->redrawHUD(0.0);
}

void vvOnscreenDebug::setText(const char *text)
{
    //this->text->setText(text, osgText::String::ENCODING_UTF8);
}

void vvOnscreenDebug::update()
{
}

void vvOnscreenDebug::show()
{
    if (!visible)
    {
        visible = true;
       // vv->getScene()->addChild(camera.get());
    }
}

void vvOnscreenDebug::hide()
{
    if (visible)
    {
        visible = false;
        //vv->getScene()->removeChild(camera.get());
    }
}

void vvOnscreenDebug::toggleVisibility()
{
    if (visible)
        hide();
    else
        show();
}

vvOnscreenDebug::~vvOnscreenDebug()
{
    hide();
}

ostringstream &vvOnscreenDebug::out()
{
    return os;
}

void vvOnscreenDebug::updateString()
{
    std::string str = os.str();
   // text->setText(str.c_str(), osgText::String::ENCODING_UTF8);
    os.str("");
}
