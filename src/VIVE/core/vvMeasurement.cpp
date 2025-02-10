/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "vvMeasurement.h"

#include <config/CoviseConfig.h>

#include "vvVIVE.h"
#include "vvSceneGraph.h"
#include "vvPluginSupport.h"
#include "vvViewer.h"

#include "vvLabel.h"


namespace vive
{

vvMeasurement::vvMeasurement()
{
    // settings
    measureScale_ = covise::coCoviseConfig::getFloat("scale", "VIVE.Measure", 1.0f);
    measureUnit_ = covise::coCoviseConfig::getEntry("unit", "VIVE.Measure", "");

    measureGroup_ = vsg::Group::create();
    vvSceneGraph::instance()->objectsRoot()->addChild(measureGroup_);
    //measureGroup_->setNodeMask(measureGroup_->getNodeMask() & ~Isect::Visible & ~Isect::Intersection & ~Isect::Pick);

    // lines

    //measureGeometry_ = vsg::Node::create();
   // measureGeometry_->getOrCreateStateSet()->setAttributeAndModes(new osg::LineWidth(1.5f), osg::StateAttribute::ON);
    /*vsg::ref_ptr<vsg::vec4Array> pathColor = new vsg::vec4Array();
    pathColor->push_back(vsg::vec4(0.0f, 1.0f, 0.0f, 1.0f));
    measureGeometry_->setColorArray(pathColor.get());
    measureGeometry_->setColorBinding(vsg::Node::BIND_OVERALL);
    measureGeometry_->getStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    measureVertices_ = new vsg::vec3Array();
    for (int i = 0; i < 2 + 2 + 50; ++i)
    {
        measureVertices_->push_back(vsg::vec3(0.0f, 0.0f, 0.0f));
    }
    measureGeometry_->setVertexArray(measureVertices_.get());

    measureGeometry_->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 0, 2));
    measureGeometry_->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 2, 2));
    measureGeometry_->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 4, 50));

    osg::Geode *geode = new osg::Geode();
    geode->addDrawable(measureGeometry_.get());
    measureGroup_->addChild(geode);
    */
    // label
    vsg::vec4 fgcolor(0.0f, 1.0f, 0.0f, 1.0f);
    float fontSize = 0.02f * vv->getSceneSize();
    measureLabel_ = new vvLabel("", fontSize, 0.0f, fgcolor, vvViewer::instance()->getBackgroundColor());
    measureLabel_->keepDistanceFromCamera(true, 50.0f);
    measureLabel_->hide();
    measureOrthoLabel_ = new vvLabel("", fontSize, 0.0f, fgcolor, vvViewer::instance()->getBackgroundColor());
    measureOrthoLabel_->keepDistanceFromCamera(true, 50.0f);
    measureOrthoLabel_->hide();
}

vvMeasurement::~vvMeasurement()
{
    vvPluginSupport::removeChild(vvSceneGraph::instance()->objectsRoot(),measureGroup_);
    delete measureLabel_;
    delete measureOrthoLabel_;
}

void vvMeasurement::start()
{
   // measureStartHitWorld_ = vv->getIntersectionHitPointWorld();
    vsg::dvec3 hit(measureStartHitWorld_);
    hit = vsg::inverse(vv->getXformMat()) * hit;
    hit /= vvSceneGraph::instance()->scaleFactor();

    measureVertices_->at(0) = hit;
    measureVertices_->at(2) = hit;

   // measureGroup_->setNodeMask(measureGroup_->getNodeMask() | Isect::Visible);
    measureLabel_->show();
    measureOrthoLabel_->show();
}

void vvMeasurement::update()
{
    vsg::dvec3 hit(vv->getIntersectionHitPointWorld());
    hit = vsg::inverse(vv->getXformMat()) * hit;
    hit /= vvSceneGraph::instance()->scaleFactor();

    vsg::dvec3 startHit(measureVertices_->at(0));

    // direct line
    measureVertices_->at(1) = hit;

    // orthogonal line
    vsg::dvec3 orthoHit = hit - startHit;
    if ((fabs(orthoHit[0]) > fabs(orthoHit[1])) && (fabs(orthoHit[0]) > fabs(orthoHit[2])))
    {
        orthoHit[1] = 0.0f;
        orthoHit[2] = 0.0f;
    }
    else if (fabs(orthoHit[1]) > fabs(orthoHit[2]))
    {
        orthoHit[0] = 0.0f;
        orthoHit[2] = 0.0f;
    }
    else
    {
        orthoHit[0] = 0.0f;
        orthoHit[1] = 0.0f;
    }
    orthoHit += startHit;
    measureVertices_->at(3) = orthoHit;

    vsg::dvec3 orthoHitWorld(orthoHit);
    orthoHitWorld *= vvSceneGraph::instance()->scaleFactor();
    orthoHitWorld = vv->getXformMat() * orthoHitWorld;

    // circle
    vsg::dvec3 dir1 = hit - orthoHit;
    vsg::dvec3 dir2 = cross(dir1,orthoHit - startHit);
    dir2 *= (length(dir1) / length(dir2));
    double delta = 2.0f * M_PI / 49.0f; // 50-1 because we need the first/last point twice
    int i(0);
    while (i < 50)
    {
        measureVertices_->at(4 + i) = orthoHit + (dir1 * (double)cos(i * delta)) + (dir2 * (double)sin(i * delta));
        ++i;
    }

  /*  measureGeometry_->setVertexArray(measureVertices_.get());
	measureVertices_->dirty();
    measureGeometry_->dirtyDisplayList();
    measureGeometry_->dirtyBound();*/

    // label

    vsg::dvec3 label1Pos = (startHit + hit * 2.0) / 3.0;
    measureLabel_->setPositionInScene(label1Pos);

    vsg::dvec3 label2Pos = (startHit * 2.0 + orthoHit) / 3.0;
    measureOrthoLabel_->setPositionInScene(label2Pos);

    std::stringstream ss1;
    ss1 << int(length(hit - startHit) * measureScale_) << " " << measureUnit_;
    measureLabel_->setString(ss1.str().c_str());

    std::stringstream ss2;
    ss2 << int(length(orthoHit - startHit) * measureScale_) << " " << measureUnit_;
    measureOrthoLabel_->setString(ss2.str().c_str());
}

void vvMeasurement::preFrame()
{
    if (measureLabel_)
        measureLabel_->update();
    if (measureOrthoLabel_)
        measureOrthoLabel_->update();
}
}
