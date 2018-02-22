/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CuttingSurfacePlane.h"
#include "CuttingSurfaceInteraction.h"
#include "CuttingSurfacePlugin.h"
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <cover/coVRNavigationManager.h>

#include <cover/coInteractor.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>

#include <config/CoviseConfig.h>

#include <PluginUtil/coPlane.h>
#include <cover/coVRPluginSupport.h>

#include <osg/Vec3>
#include <osg/Matrix>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/ComputeBoundsVisitor>

#include <osg/BlendFunc>
#include <osg/AlphaFunc>
#include <osg/Material>
#include <osg/io_utils>

using namespace vrui;
using namespace opencover;
using covise::coCoviseConfig;

CuttingSurfacePlane::CuttingSurfacePlane(coInteractor *inter, CuttingSurfacePlugin *pl)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "cuttingSurfacePlane::cuttingSurfacePlane\n");
    plugin = pl;
    inter_ = inter;
    newModule_ = false;
    wait_ = false;
    showPickInteractor_ = false;
    showDirectInteractor_ = false;

    // default size for all interactors
    float interSize = -1.f;
    // if defined, COVER.IconSize overrides the default
    interSize = coCoviseConfig::getFloat("COVER.IconSize", interSize);
    // if defined, COVERConfigCuttingSurfacePlugin.IconSize overrides both
    interSize = coCoviseConfig::getFloat("COVER.Plugin.Cuttingsurface.IconSize", interSize);

    // extract parameters from covise
    float *p = NULL, *n = NULL;
    int dummy; // we know that the vector has 3 elements
    point_.set(0., 0., 0.);
    if (inter_->getFloatVectorParam(CuttingSurfaceInteraction::POINT, dummy, p) != -1)
    {
        point_.set(p[0], p[1], p[2]);
    }

    normal_.set(0., 1., 0.);
    if (inter_->getFloatVectorParam(CuttingSurfaceInteraction::VERTEX, dummy, n) != -1)
    {
        //std::cerr << "CuttingSurfacePlane: Normal: " << n[0] << " " << n[1] << " " << n[2] << std::endl;
        normal_.set(n[0], n[1], n[2]);
        normal_.normalize();
    }

    osg::Matrix m;
    osg::Vec3 yaxis(0, 1, 0);
    m.makeRotate(yaxis, normal_);
    m.setTrans(point_);

    // create and position pickinteractor
    planePickInteractor_ = new coVR3DTransRotInteractor(m, interSize, coInteraction::ButtonA, "hand", "Plane_S0", coInteraction::Medium);
    //planePickInteractor_ = new coVR3DTransRotInteractor(m, interSize, coVR3DTransRotInteractor::TwoD, coInteraction::ButtonA, "hand", "Plane_S0", coInteraction::High);
    //planePickInteractor_->updateTransform(m);
    planePickInteractor_->hide();
    planePickInteractor_->disableIntersection();

    // direct interactor
    if (!coVRConfig::instance()->has6DoFInput())
    {
        planeDirectInteractor_ = NULL;
    }
    else
    {
        planeDirectInteractor_ = new coTrackerButtonInteraction(coInteraction::ButtonA, "disc", coInteraction::Medium);
    }

    // create geometry, this is the intersection lines of the plane with the bounding box of the scene
    parent_ = cover->getObjectsScale();
    hasCase_ = false;
    createGeometry();
    showGeometry(false);
}

CuttingSurfacePlane::~CuttingSurfacePlane()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacePlane::~CuttingSurfacePlane\n");

    delete planePickInteractor_;
    if (planeDirectInteractor_)
        delete planeDirectInteractor_;

    deleteGeometry();
}

void
CuttingSurfacePlane::preFrame(int restrictToAxis)
{
    if (cover->debugLevel(5))
        fprintf(stderr, "CuttingSurfacePlane::preFrame\n");

    if (showDirectInteractor_ && planeDirectInteractor_ && planeDirectInteractor_->wasStopped())
    {
        newModule_ = false;
        if (!wait_)
        {
            osg::Matrix w_to_o = cover->getInvBaseMat();
            osg::Matrix pointerMat_o = cover->getPointerMat() * w_to_o;
            if (coVRNavigationManager::instance()->isSnapping())
            {
                if (coVRNavigationManager::instance()->isDegreeSnapping())
                {
                    // snap orientation
                    snapToDegrees(coVRNavigationManager::instance()->snappingDegrees(), &pointerMat_o);
                }
                else
                {
                    // snap orientation to 45 degree
                    snapTo45Degrees(&pointerMat_o);
                }
            }

            //planeDirectInteractor_->getIconWorldPosition(cuttingSurfacePos_w[0], cuttingSurfacePos_w[1], cuttingSurfacePos_w[2]);
            osg::Vec3 cuttingSurfacePos_o = pointerMat_o.getTrans();
            osg::Vec4 yaxis(0, 1, 0, 0), cuttingSurfaceNormal_o;

            if (restrictToAxis == CuttingSurfaceInteraction::RESTRICT_NONE)
            {
                cuttingSurfaceNormal_o = yaxis * pointerMat_o;
                cuttingSurfaceNormal_o.normalize();
            }
            else if (restrictToAxis == CuttingSurfaceInteraction::RESTRICT_X)
            {
                cuttingSurfaceNormal_o = osg::Vec4(1, 0, 0, 0);
            }
            else if (restrictToAxis == CuttingSurfaceInteraction::RESTRICT_Y)
            {
                cuttingSurfaceNormal_o = osg::Vec4(0, 1, 0, 0);
            }
            else if (restrictToAxis == CuttingSurfaceInteraction::RESTRICT_Z)
            {
                cuttingSurfaceNormal_o = osg::Vec4(0, 0, 1, 0);
            }

            //fprintf(stderr,"send normal=[%f %f %f]\n",  cuttingSurfaceNormal_w[0],  cuttingSurfaceNormal_w[1], cuttingSurfaceNormal_w[2]);

            plugin->getSyncInteractors(inter_);
            plugin->setVectorParam(CuttingSurfaceInteraction::POINT, cuttingSurfacePos_o[0], cuttingSurfacePos_o[1], cuttingSurfacePos_o[2]);
            plugin->setVectorParam(CuttingSurfaceInteraction::VERTEX, cuttingSurfaceNormal_o[0], cuttingSurfaceNormal_o[1], cuttingSurfaceNormal_o[2]);
            point_.set(cuttingSurfacePos_o[0], cuttingSurfacePos_o[1], cuttingSurfacePos_o[2]);
            normal_.set(cuttingSurfaceNormal_o[0], cuttingSurfaceNormal_o[1], cuttingSurfaceNormal_o[2]);
            plugin->executeModule();

            wait_ = true;
        }
    }

    // pick interaction mode
    planePickInteractor_->preFrame();
    if (showPickInteractor_)
    {
        osg::Matrix m = planePickInteractor_->getMatrix();
        osg::Vec3 point = m.getTrans();
        //std::cerr << "CuttingSurfacePlane::preFrame(): point=" << point << std::endl;
        osg::Vec4 normal;
        if (restrictToAxis == CuttingSurfaceInteraction::RESTRICT_NONE)
        {
            osg::Vec4 yaxis(0, 1, 0, 0);
            normal = yaxis * m;
            normal.normalize();
            if (normal.length2() < 0.1f)
            {
                std::cerr << "CuttingSurfacePlane: invalid normal" << std::endl;
                normal = yaxis;
            }
        }
        else if (restrictToAxis == CuttingSurfaceInteraction::RESTRICT_X)
        {
            normal = osg::Vec4(1, 0, 0, 0);
        }
        else if (restrictToAxis == CuttingSurfaceInteraction::RESTRICT_Y)
        {
            normal = osg::Vec4(0, 1, 0, 0);
        }
        else if (restrictToAxis == CuttingSurfaceInteraction::RESTRICT_Z)
        {
            normal = osg::Vec4(0, 0, 1, 0);
        }
        else
        {
            std::cerr << "CuttingSurfacePlane: invalid restrictToAxis: " << restrictToAxis << std::endl;
        }

        point_.set(point[0], point[1], point[2]);
        normal_.set(normal[0], normal[1], normal[2]);

        if (planePickInteractor_->wasStarted())
        {
            showGeometry(true);
            updateGeometry();
        }
        if (planePickInteractor_->isRunning())
        {
            updateGeometry();
        }
        if (planePickInteractor_->wasStopped())
        {
            updateGeometry();
            if (!wait_)
            {
                showGeometry(false);
                plugin->getSyncInteractors(inter_);
                plugin->setVectorParam(CuttingSurfaceInteraction::POINT, point_[0], point_[1], point_[2]);
                plugin->setVectorParam(CuttingSurfaceInteraction::VERTEX, normal_[0], normal_[1], normal_[2]);
                plugin->executeModule();
                wait_ = true;
                //fprintf(stderr,"execute\n");
            }
        }
    }
}

void
CuttingSurfacePlane::restrict(int restrictToAxis)
{
    //fprintf(stderr, "CuttingSurfacePlane::restrict\n");
    osg::Vec3 normal = normal_;
    if (restrictToAxis == CuttingSurfaceInteraction::RESTRICT_X)
    {
        normal = osg::Vec3(1, 0, 0);
    }
    else if (restrictToAxis == CuttingSurfaceInteraction::RESTRICT_Y)
    {
        normal = osg::Vec3(0, 1, 0);
    }
    else if (restrictToAxis == CuttingSurfaceInteraction::RESTRICT_Z)
    {
        normal = osg::Vec3(0, 0, 1);
    }
    plugin->getSyncInteractors(inter_);
    plugin->setVectorParam(CuttingSurfaceInteraction::VERTEX, normal[0], normal[1], normal[2]);
    plugin->executeModule();
}

void
CuttingSurfacePlane::update(coInteractor *inter)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacePlane::update\n");
    if (wait_)
    {
        wait_ = false;
    }
    inter_ = inter;

    float *p = NULL, *n = NULL;
    int dummy; // we know that the vector has 3 elements
    if (inter_->getFloatVectorParam(CuttingSurfaceInteraction::POINT, dummy, p) != -1)
    {
        point_.set(p[0], p[1], p[2]);
    }

    if (inter_->getFloatVectorParam(CuttingSurfaceInteraction::VERTEX, dummy, n) != -1)
    {
        normal_.set(n[0], n[1], n[2]);
        normal_.normalize();
    }
    else
    {
        std::cerr << "getFloatVectorParam(" << CuttingSurfaceInteraction::VERTEX << ") failed" << std::endl;
    }

    if (p && n)
    {
        osg::Matrix m;
        osg::Vec3 yaxis(0, 1, 0);
        m.makeRotate(yaxis, normal_);
        m.setTrans(point_);

        planePickInteractor_->updateTransform(m);
        updateGeometry();
    }
    showGeometry(false);
}

void
CuttingSurfacePlane::showPickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\ncuttingSurfacePlane::showPickInteractor\n");

    showPickInteractor_ = true;
    planePickInteractor_->show();
    planePickInteractor_->enableIntersection();
}

void
CuttingSurfacePlane::hidePickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\ncuttingSurfacePlane::hide\n");

    showPickInteractor_ = false;
    planePickInteractor_->hide();
    planePickInteractor_->disableIntersection();
}

void
CuttingSurfacePlane::showDirectInteractor()
{
    showDirectInteractor_ = true;
    if (planeDirectInteractor_ && !planeDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(planeDirectInteractor_);
    }
}

void
CuttingSurfacePlane::hideDirectInteractor()
{

    showDirectInteractor_ = false;

    if (planeDirectInteractor_ && planeDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->unregisterInteraction(planeDirectInteractor_);
    }
}

void
CuttingSurfacePlane::setNew()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "cuttingSurfacePlane::setNew\n");

    newModule_ = true;

    if (planeDirectInteractor_ && !planeDirectInteractor_->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(planeDirectInteractor_);
    }
}

void CuttingSurfacePlane::setInteractorPoint(osg::Vec3 p)
{
    point_ = p;
    osg::Matrix m;
    osg::Vec3 yaxis(0, 1, 0);
    m.makeRotate(yaxis, normal_);
    m.setTrans(point_);

    planePickInteractor_->updateTransform(m);
}

void CuttingSurfacePlane::setInteractorNormal(osg::Vec3 n)
{
    n.normalize();
    normal_ = n;
    osg::Matrix m;
    osg::Vec3 yaxis(0, 1, 0);
    m.makeRotate(yaxis, normal_);
    m.setTrans(point_);

    planePickInteractor_->updateTransform(m);
}

bool CuttingSurfacePlane::sendClipPlane()
{
    return ((showDirectInteractor_ && planeDirectInteractor_
             && planeDirectInteractor_->wasStopped())
            || (showPickInteractor_ && planePickInteractor_->wasStopped() && !wait_));
}

void CuttingSurfacePlane::createGeometry()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "cuttingSurfacePlane::createGeometry\n");

    intersectFlag_ = oldIntersectFlag_ = false;

    osg::Vec4Array *outlineColor;
    outlineColor = new osg::Vec4Array(1);
    (*outlineColor)[0].set(1, 1, 1, 1);
    outlineCoords_ = new osg::Vec3Array(7);
    (*outlineCoords_)[0].set(-0.05, 0.0, -0.05);
    (*outlineCoords_)[1].set(0.05, 0.0, -0.05);
    (*outlineCoords_)[2].set(0.05, 0.0, 0.05);
    (*outlineCoords_)[3].set(-0.05, 0.0, 0.05);
    (*outlineCoords_)[4].set(-0.05, 0.0, -0.05);
    (*outlineCoords_)[5].set(-0.05, 0.0, -0.05);
    (*outlineCoords_)[6].set(-0.05, 0.0, -0.05);

    osg::Vec4Array *polyColor;
    polyColor = new osg::Vec4Array(1);
    (*polyColor)[0].set(1.0, 0.5, 0.5, 0.5);

    polyCoords_ = new osg::Vec3Array(6);
    (*polyCoords_)[0].set(-0.05, 0.0, -0.05);
    (*polyCoords_)[1].set(0.05, 0.0, -0.05);
    (*polyCoords_)[2].set(0.05, 0.0, 0.05);
    (*polyCoords_)[3].set(-0.05, 0.0, 0.05);
    (*polyCoords_)[4].set(-0.05, 0.0, -0.05);
    (*polyCoords_)[5].set(-0.05, 0.0, -0.05);

    polyNormal_ = new osg::Vec3Array(polyCoords_->size());
    for (size_t i = 0; i < polyCoords_->size(); ++i)
    {
        (*polyNormal_)[i].set(normal_[0], normal_[1], normal_[2]);
    }

    testPlane_ = new coPlane(normal_, point_);

    outlineGeometry_ = new osg::Geometry();
    outlineGeometry_->setColorArray(outlineColor);
    outlineGeometry_->setColorBinding(osg::Geometry::BIND_OVERALL);
    outlineGeometry_->setVertexArray(outlineCoords_.get());
    outlineGeometry_->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, 7));
    outlineGeometry_->setUseDisplayList(false);
    outlineGeometry_->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate(osg::Material::AMBIENT_AND_DIFFUSE));

    polyGeometry_ = new osg::Geometry();
    polyGeometry_->setColorArray(polyColor);
    polyGeometry_->setColorBinding(osg::Geometry::BIND_OVERALL);
    polyGeometry_->setVertexArray(polyCoords_.get());
    polyGeometry_->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 6));
    polyGeometry_->setNormalArray(polyNormal_.get());
    polyGeometry_->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    polyGeometry_->setUseDisplayList(false);
    polyGeometry_->setStateSet(VRSceneGraph::instance()->loadTransparentGeostate(osg::Material::AMBIENT_AND_DIFFUSE));

    geode_ = new osg::Geode();
    geode_->addDrawable(outlineGeometry_.get());
    geode_->addDrawable(polyGeometry_.get());
    geode_->setName("CuttingSurfacePlaneOutline");
    geode_->setNodeMask(geode_->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

    parent_->addChild(geode_);

    updateGeometry();
}

void CuttingSurfacePlane::deleteGeometry()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "CuttingSurfacePlane::deleteGeometry\n");

    if (geode_.get() && (geode_->getNumParents() > 0))
    {
        geode_->getParent(0)->removeChild(geode_.get());
    }
    delete testPlane_;
    testPlane_ = nullptr;
}

void CuttingSurfacePlane::updateGeometry()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "CuttingSurfacePlane::updateGeometry\n");

    intersectFlag_ = testIntersection() != 0;

    if (intersectFlag_ != oldIntersectFlag_)
    {
        showGeometry(intersectFlag_);
        oldIntersectFlag_ = intersectFlag_;
    }
}

void CuttingSurfacePlane::showGeometry(bool show)
{

    if (cover->debugLevel(3))
        fprintf(stderr, "showGeometry %d\n", show);
    if (show)
    {
        if (geode_.get() && geode_->getNumParents() == 0)
        {
            parent_->addChild(geode_.get());
        }
    }
    else
    {
        if (geode_.get() && geode_->getNumParents())
        {
            geode_->getParent(0)->removeChild(geode_.get());
        }
    }
}

int
CuttingSurfacePlane::testIntersection()
{

    if (cover->debugLevel(5))
        fprintf(stderr, "cuttingSurfacePlane::testIntersection\n");

    // get bounding box of objectsRoot
    osg::BoundingBox bb = cover->getBBox(cover->getObjectsRoot());

    // NULL objects (which appear when the cuttingsurface was outside) have an inverted bbox
    // if we have only a plane and no other geom in the scenegraph
    if ((bb._min.x() >= bb._max.x())
        || (bb._min.y() >= bb._max.y())
        || (bb._min.z() >= bb._max.z())

            )
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "strange bbox: min=[%f %f %f] max=[%f %f %f]\n", bb._min.x(), bb._min.y(), bb._min.z(), bb._max.x(), bb._max.y(), bb._max.z());
        return 0;
    }

    osg::Vec3 p;
    if (hasCase_)
    {
        p = point_ * ((osg::MatrixTransform *)parent_)->getMatrix();
    }
    else
        p = point_;
    //fprintf(stderr,"point=[%f %f %f]\n", p[0], p[1], p[2]);
    testPlane_->update(normal_, p);

    osg::Vec3 isectPoints[6];
    int n = testPlane_->getBoxIntersectionPoints(bb, isectPoints);

    if (cover->debugLevel(5))
        fprintf(stderr, "%d intersections\n", n);
    if (n > 0)
    {
        //std::cerr << "updating geometry with " << n << " points" << std::endl;

        osg::Matrix m;
        if (hasCase_)
            m.invert(((osg::MatrixTransform *)parent_)->getMatrix());
        else
            m.makeIdentity();
        for (int i = 0; i < n; i++)
        {
            (*outlineCoords_)[i] = isectPoints[i] * m;
            (*polyCoords_)[i] = isectPoints[i] * m;
        }

        for (int i = n; i < 6; i++)
        {
            (*outlineCoords_)[i] = (*outlineCoords_)[n - 1];
            (*polyCoords_)[i] = (*polyCoords_)[n - 1];
        }

        (*outlineCoords_)[6] = (*outlineCoords_)[0];

        (*polyNormal_)[0].set(normal_[0], normal_[1], normal_[2]);

        return n;
    }
    else
    {
        //fprintf(stderr,"NO intersection\n");
        return 0;
    }
}

void
CuttingSurfacePlane::setCaseTransform(osg::MatrixTransform *t)
{
    planePickInteractor_->setCaseTransform(t);
    parent_ = t;
    hasCase_ = true;
    if (geode_.get() && geode_->getNumParents())
    {
        geode_->getParent(0)->removeChild(geode_.get());
        parent_->addChild(geode_.get());
    }
}
