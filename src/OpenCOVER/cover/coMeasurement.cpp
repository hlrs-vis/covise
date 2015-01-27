/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coMeasurement.h"

#include <config/CoviseConfig.h>

#include "OpenCOVER.h"
#include "VRSceneGraph.h"
#include "coVRPluginSupport.h"
#include "VRViewer.h"

#include "coVRLabel.h"

#include <osg/LineWidth>

namespace opencover
{

coMeasurement::coMeasurement()
{
    // settings
    measureScale_ = covise::coCoviseConfig::getFloat("scale", "COVER.Measure", 1.0f);
    measureUnit_ = covise::coCoviseConfig::getEntry("unit", "COVER.Measure", "");

    measureGroup_ = new osg::Group();
    VRSceneGraph::instance()->objectsRoot()->addChild(measureGroup_.get());
    measureGroup_->setNodeMask(measureGroup_->getNodeMask() & ~Isect::Visible & ~Isect::Intersection & ~Isect::Pick);

    // lines

    measureGeometry_ = new osg::Geometry();
    measureGeometry_->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
    measureGeometry_->getOrCreateStateSet()->setAttributeAndModes(new osg::LineWidth(1.5f), osg::StateAttribute::ON);
    osg::ref_ptr<osg::Vec4Array> pathColor = new osg::Vec4Array();
    pathColor->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f));
    measureGeometry_->setColorArray(pathColor.get());
    measureGeometry_->setColorBinding(osg::Geometry::BIND_OVERALL);
    measureGeometry_->getStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    measureVertices_ = new osg::Vec3Array();
    for (int i = 0; i < 2 + 2 + 50; ++i)
    {
        measureVertices_->push_back(osg::Vec3(0.0f, 0.0f, 0.0f));
    }
    measureGeometry_->setVertexArray(measureVertices_.get());

    measureGeometry_->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 0, 2));
    measureGeometry_->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 2, 2));
    measureGeometry_->addPrimitiveSet(new osg::DrawArrays(GL_LINE_STRIP, 4, 50));

    osg::Geode *geode = new osg::Geode();
    geode->addDrawable(measureGeometry_.get());
    measureGroup_->addChild(geode);

    // label
    osg::Vec4 fgcolor(0.0f, 1.0f, 0.0f, 1.0f);
    float fontSize = 0.02f * cover->getSceneSize();
    measureLabel_ = new coVRLabel("", fontSize, 0.0f, fgcolor, VRViewer::instance()->getBackgroundColor());
    measureLabel_->keepDistanceFromCamera(true, 50.0f);
    measureLabel_->hide();
    measureOrthoLabel_ = new coVRLabel("", fontSize, 0.0f, fgcolor, VRViewer::instance()->getBackgroundColor());
    measureOrthoLabel_->keepDistanceFromCamera(true, 50.0f);
    measureOrthoLabel_->hide();
}

coMeasurement::~coMeasurement()
{
    VRSceneGraph::instance()->objectsRoot()->removeChild(measureGroup_.get());
    delete measureLabel_;
    delete measureOrthoLabel_;
}

void coMeasurement::start()
{
    measureStartHitWorld_ = cover->getIntersectionHitPointWorld();
    osg::Vec3 hit = measureStartHitWorld_;
    hit = osg::Matrixd::inverse(cover->getXformMat()).preMult(hit);
    hit /= VRSceneGraph::instance()->scaleFactor();

    measureVertices_->at(0) = hit;
    measureVertices_->at(2) = hit;

    measureGroup_->setNodeMask(measureGroup_->getNodeMask() | Isect::Visible);
    measureLabel_->show();
    measureOrthoLabel_->show();
}

void coMeasurement::update()
{
    osg::Vec3 hit = cover->getIntersectionHitPointWorld();
    hit = osg::Matrixd::inverse(cover->getXformMat()).preMult(hit);
    hit /= VRSceneGraph::instance()->scaleFactor();

    osg::Vec3 startHit = measureVertices_->at(0);

    // direct line
    measureVertices_->at(1) = hit;

    // orthogonal line
    osg::Vec3 orthoHit = hit - startHit;
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

    osg::Vec3 orthoHitWorld = orthoHit;
    orthoHitWorld *= VRSceneGraph::instance()->scaleFactor();
    orthoHitWorld = cover->getXformMat().preMult(orthoHitWorld);

    // circle
    osg::Vec3 dir1 = hit - orthoHit;
    osg::Vec3 dir2 = dir1 ^ (orthoHit - startHit);
    dir2 *= (dir1.length() / dir2.length());
    float delta = 2.0f * M_PI / 49.0f; // 50-1 because we need the first/last point twice
    int i(0);
    while (i < 50)
    {
        measureVertices_->at(4 + i) = orthoHit + (dir1 * cos(i * delta)) + (dir2 * sin(i * delta));
        ++i;
    }

    measureGeometry_->setVertexArray(measureVertices_.get());
    measureGeometry_->dirtyDisplayList();
    measureGeometry_->dirtyBound();

    // label

    osg::Vec3 label1Pos = (startHit + hit * 2.0f) / 3.0f;
    measureLabel_->setPositionInScene(label1Pos);

    osg::Vec3 label2Pos = (startHit * 2.0f + orthoHit) / 3.0f;
    measureOrthoLabel_->setPositionInScene(label2Pos);

    std::stringstream ss1;
    ss1 << int((hit - startHit).length() * measureScale_) << " " << measureUnit_;
    measureLabel_->setString(ss1.str().c_str());

    std::stringstream ss2;
    ss2 << int((orthoHit - startHit).length() * measureScale_) << " " << measureUnit_;
    measureOrthoLabel_->setString(ss2.str().c_str());
}

void coMeasurement::preFrame()
{
    if (measureLabel_)
        measureLabel_->update();
    if (measureOrthoLabel_)
        measureOrthoLabel_->update();
}
}
