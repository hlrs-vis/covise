/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "coOnscreenDebug.h"

#include "VRViewer.h"
#include "coVRPluginSupport.h"
#include "coVRFileManager.h"
#include <iostream>
#include <config/CoviseConfig.h>
#include <osgDB/ReaderWriter>
#include <osgDB/ReadFile>
#include <osg/Geometry>
#include <osg/MatrixTransform>

using namespace osg;
using namespace covise;
using namespace opencover;

coOnscreenDebug *coOnscreenDebug::singleton = 0;

coOnscreenDebug *coOnscreenDebug::instance()
{
    if (!singleton)
        singleton = new coOnscreenDebug();
    return singleton;
}

coOnscreenDebug::coOnscreenDebug()
{
    visible = true;

    geode = new osg::Geode();

    std::string timesFont = coVRFileManager::instance()->getFontFile(NULL);

    // turn lighting off for the text and disable depth test to ensure its always ontop.
    osg::StateSet *stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    osg::Vec3 position(320.0f, 200.0f, 0.2f);
    osg::Vec3 delta(0.0f, -40.0f, 0.0f);

    text = new osgText::Text;
    text->setDataVariance(Object::DYNAMIC);
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
        osg::Geometry *geom = new osg::Geometry;

        osg::Vec3Array *vertices = new osg::Vec3Array;
        vertices->push_back(osg::Vec3(0, 480, 0.001));
        vertices->push_back(osg::Vec3(0, 0, 0.001));
        vertices->push_back(osg::Vec3(640, 0, 0.001));
        vertices->push_back(osg::Vec3(640, 480, 0.001));
        geom->setVertexArray(vertices);

        osg::Vec3Array *normals = new osg::Vec3Array;
        normals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));
        geom->setNormalArray(normals);
        geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

        osg::Vec4Array *colors = new osg::Vec4Array;
        colors->push_back(osg::Vec4(0.4f, 0.4f, 0.4f, 0.25f));
        geom->setColorArray(colors);
        geom->setColorBinding(osg::Geometry::BIND_OVERALL);

        geom->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));

        osg::StateSet *stateset = geom->getOrCreateStateSet();

        stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
        //stateset->setAttribute(new osg::PolygonOffset(1.0f,1.0f),osg::StateAttribute::ON);
        stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
        stateset->setNestRenderBins(false);

        geode->addDrawable(geom);
    }

    camera = new osg::Camera;

    camera->setProjectionMatrix(osg::Matrix::ortho2D(0, 1024, 0, 768));
    camera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
    // set the view matrix
    camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    camera->setViewMatrix(osg::Matrix::translate(osg::Vec3(0, 0, 100)));
    camera->setViewMatrix(osg::Matrix::identity());

    // only clear the depth buffer
    camera->setClearMask(GL_DEPTH_BUFFER_BIT);

    // draw subgraph after main camera view.
    camera->setRenderOrder(osg::Camera::POST_RENDER);

    osg::MatrixTransform *m = new osg::MatrixTransform;
    m->addChild(geode.get());
    m->setMatrix(osg::Matrix::translate((1024 - 640) / 2.0, (768 - 480) / 2.0, 0));
    camera->addChild(m);

    cover->getScene()->addChild(camera.get());
}

void coOnscreenDebug::redraw()
{
    VRViewer::instance()->redrawHUD(0.0);
}

void coOnscreenDebug::setText(const char *text)
{
    this->text->setText(text, osgText::String::ENCODING_UTF8);
}

void coOnscreenDebug::update()
{
}

void coOnscreenDebug::show()
{
    if (!visible)
    {
        visible = true;
        cover->getScene()->addChild(camera.get());
    }
}

void coOnscreenDebug::hide()
{
    if (visible)
    {
        visible = false;
        cover->getScene()->removeChild(camera.get());
    }
}

void coOnscreenDebug::toggleVisibility()
{
    if (visible)
        hide();
    else
        show();
}

coOnscreenDebug::~coOnscreenDebug()
{
    hide();
}

ostringstream &coOnscreenDebug::out()
{
    return os;
}

void coOnscreenDebug::updateString()
{
    std::string str = os.str();
    text->setText(str.c_str(), osgText::String::ENCODING_UTF8);
    os.str("");
}
