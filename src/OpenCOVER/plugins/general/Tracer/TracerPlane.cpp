/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TracerInteraction.h"
#include "TracerPlane.h"
#include "TracerPlugin.h"
#include <cover/coInteractor.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <PluginUtil/coVR2DTransInteractor.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <PluginUtil/coPlane.h>
#include <cover/RenderObject.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>
#include <cover/coVRPluginSupport.h>
#include <grmsg/coGRObjVisMsg.h>
#include <net/message.h>

#include <config/CoviseConfig.h>

#include <osg/Material>
#include <osg/StateSet>
#include <osg/BlendFunc>
#include <osg/AlphaFunc>

using namespace vrui;
using namespace opencover;
using namespace grmsg;

TracerPlane::TracerPlane(coInteractor *inter, TracerPlugin *p)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nnew TracerPlane\n");
    plugin = p;

    osg::Matrix m0;

    _inter = inter;
    _newModule = false;
    showPickInteractor_ = false;
    showDirectInteractor_ = false;
    keepSquare_ = false;
    wait_ = false;
    geometryLine_ = NULL;
    geometryPoly_ = NULL;

    // default size for all interactors
    _interSize = -1.f;
    // if defined, COVERConfig.ICON_SIZE overrides the default
    _interSize = coCoviseConfig::getFloat("COVER.IconSize", _interSize);
    // if defined, TracerPlugin.SCALEFACTOR overrides both
    _interSize = coCoviseConfig::getFloat("COVER.Plugin.Tracer.IconSize", _interSize);

    _execOnChange = coCoviseConfig::isOn("COVER.ExecuteOnChange", true);

    _cyberclassroom = coCoviseConfig::isOn("CyberClassroom", false);

    // create interactors
    // set some good default values
    // not from parameter values, beccause this is a base class
    // and TracerInteraction::P_XXX could be the wrong index
    // later in update we can be sure that getParameter from the derived class is called if any
    //_pos1=getParameterStartpoint1();
    //_pos2=getParameterStartpoint2();
    //_dir1=getParameterDirection();
    _pos1.set(-0.5, 0.5, 0);
    _pos2.set(0.5, 0.5, 0);
    _dir1.set(0, 1, 0);

    computeQuad12();
    m0.makeIdentity();
    m0(0, 0) = _dir1[0];
    m0(0, 1) = _dir1[1];
    m0(0, 2) = _dir1[2];
    m0(1, 0) = _dir2[0];
    m0(1, 1) = _dir2[1];
    m0(1, 2) = _dir2[2];
    m0(2, 0) = _dir3[0];
    m0(2, 1) = _dir3[1];
    m0(2, 2) = _dir3[2];
    m0(3, 0) = _pos0[0];
    m0(3, 1) = _pos0[1];
    m0(3, 2) = _pos0[2];

    parent = cover->getObjectsScale();
    _s0 = new coVR3DTransRotInteractor(m0, _interSize, coInteraction::ButtonA, "Menu", "Plane_S0", coInteraction::Medium);
    _s1 = new coVR2DTransInteractor(_pos1, _dir2, _interSize, coInteraction::ButtonA, "Menu", "Plane_S1", coInteraction::Medium);
    _s2 = new coVR2DTransInteractor(_pos2, _dir2, _interSize, coInteraction::ButtonA, "Menu", "Plane_S2", coInteraction::Medium);
    _s3 = new coVR2DTransInteractor(_pos3, _dir2, _interSize, coInteraction::ButtonA, "Menu", "Plane_S3", coInteraction::Medium);
    _s4 = new coVR2DTransInteractor(_pos4, _dir2, _interSize, coInteraction::ButtonA, "Menu", "Plane_S4", coInteraction::Medium);

    initialObjectName_ = _inter->getObject()->getName();

    if (!coVRConfig::instance()->has6DoFInput())
    {
        _directInteractor = NULL;
        //fprintf(stderr,"Info: TracerPlugin: no direct interaction for tracer plane possible for TRACKING_SYSTEM MOUSE\n");
    }
    else
    {
        _directInteractor = new coTrackerButtonInteraction(coInteraction::ButtonA, "sphere", coInteraction::Medium);
    }
    createGeometry();
    _plane = new coPlane(_dir3, _pos0);

    hideDirectInteractor();
    hidePickInteractor();
    hideGeometry();

    // hide s1 s2 s3 s4 for cyberclassroom
    if (_cyberclassroom)
    {
        //fprintf(stderr,"\nnew TracerPlane cyberclassroom\n");
        _s1->hide();
        _s1->disableIntersection();
        _s2->hide();
        _s2->disableIntersection();
        _s3->hide();
        _s3->disableIntersection();
        _s4->hide();
        _s4->disableIntersection();
    }
}

TracerPlane::~TracerPlane()
{

    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete TracerPlane\n");

    delete _s0;
    delete _s1;
    delete _s2;
    delete _s3;
    delete _s4;

    if (_directInteractor)
        delete _directInteractor;

    deleteGeometry();
    delete _plane;
}

void
TracerPlane::updatePlane()
{
    computeQuad12();
    osg::Matrix m0;
    m0.makeIdentity();
    m0(0, 0) = _dir1[0];
    m0(0, 1) = _dir1[1];
    m0(0, 2) = _dir1[2];
    m0(1, 0) = _dir2[0];
    m0(1, 1) = _dir2[1];
    m0(1, 2) = _dir2[2];
    m0(2, 0) = _dir3[0];
    m0(2, 1) = _dir3[1];
    m0(2, 2) = _dir3[2];
    m0(3, 0) = _pos0[0];
    m0(3, 1) = _pos0[1];
    m0(3, 2) = _pos0[2];
    _s0->updateTransform(m0);
    _s1->updateTransform(_pos1, _dir2);
    _s2->updateTransform(_pos2, _dir2);
    _s3->updateTransform(_pos3, _dir2);
    _s4->updateTransform(_pos4, _dir2);
    updateGeometry();
    _plane->update(_dir3, _pos0);
}

void
TracerPlane::update(coInteractor *inter)
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlane::update\n");

    osg::Matrix m0;
    _inter = inter;

    _pos1 = getParameterStartpoint1();
    _pos2 = getParameterStartpoint2();
    _dir1 = getParameterDirection();

    updatePlane();

    wait_ = false;
}

void
TracerPlane::setNew()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "TracerLine::setNew\n");

    _newModule = true;

    if (_directInteractor && !_directInteractor->isRegistered())
        coInteractionManager::the()->registerInteraction(_directInteractor);
}
void
TracerPlane::createGeometry()
{

    osg::Vec4Array *colorLine, *colorPoly;
    colorLine = new osg::Vec4Array(1);
    colorPoly = new osg::Vec4Array(1);
    coordLine_ = new osg::Vec3Array(5);
    coordPoly_ = new osg::Vec3Array(4);
    polyNormal_ = new osg::Vec3Array(1);

    updateGeometry();

    (*colorLine)[0].set(1, 0.5, 0.5, 1);
    (*colorPoly)[0].set(1.0, 0.5, 0.5, 0.5);

    geometryLine_ = new osg::Geometry();
    geometryLine_->setColorArray(colorLine);
    geometryLine_->setColorBinding(osg::Geometry::BIND_OVERALL);
    geometryLine_->setVertexArray(coordLine_.get());
    geometryLine_->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINE_STRIP, 0, 5));
    geometryLine_->setUseDisplayList(false);
    geometryLine_->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());

    geometryPoly_ = new osg::Geometry();
    geometryPoly_->setColorArray(colorPoly);
    geometryPoly_->setColorBinding(osg::Geometry::BIND_OVERALL);
    geometryPoly_->setVertexArray(coordPoly_.get());
    geometryPoly_->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::POLYGON, 0, 4));
    geometryPoly_->setUseDisplayList(false);
    osg::StateSet *stateSet = VRSceneGraph::instance()->loadTransparentGeostate();
    stateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    geometryPoly_->setStateSet(stateSet);

    geometryNode = new osg::Geode();
    geometryNode->addDrawable(geometryLine_.get());
    geometryNode->addDrawable(geometryPoly_.get());
    geometryNode->setName("TracerPlaneGeometry");
    geometryNode->setNodeMask(geometryNode->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

    parent->addChild(geometryNode.get());
}
void
TracerPlane::deleteGeometry()
{
    if (geometryNode.get())
    {
        if (geometryNode->getNumParents())
            geometryNode->getParent(0)->removeChild(geometryNode.get());
    }
}

void TracerPlane::updateGeometry()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "TracerPlane::updateGeometry\n");
    (*coordLine_)[0].set(_pos1[0], _pos1[1], _pos1[2]);
    (*coordLine_)[1].set(_pos3[0], _pos3[1], _pos3[2]);
    (*coordLine_)[2].set(_pos2[0], _pos2[1], _pos2[2]);
    (*coordLine_)[3].set(_pos4[0], _pos4[1], _pos4[2]);
    (*coordLine_)[4].set(_pos1[0], _pos1[1], _pos1[2]);

    (*coordPoly_)[0].set(_pos1[0], _pos1[1], _pos1[2]);
    (*coordPoly_)[1].set(_pos3[0], _pos3[1], _pos3[2]);
    (*coordPoly_)[2].set(_pos2[0], _pos2[1], _pos2[2]);
    (*coordPoly_)[3].set(_pos4[0], _pos4[1], _pos4[2]);

    osg::Vec3 v1 = _pos1 - _pos0;
    osg::Vec3 v2 = _pos2 - _pos1;
    osg::Vec3 vn = v1 ^ v2;
    vn.normalize();
	coordLine_->dirty();
	coordPoly_->dirty();
    (*polyNormal_)[0].set(vn[0], vn[1], vn[2]);
	if (geometryLine_ != NULL)
	{
		geometryLine_->dirtyDisplayList();
		geometryLine_->dirtyBound();
	}
    if (geometryPoly_ != NULL)
	{
		geometryPoly_->dirtyDisplayList();
		geometryPoly_->dirtyBound();
	}
}

void
TracerPlane::preFrame()
{
    osg::Vec3 xaxis(1, 0, 0), yaxis(0, 1, 0), zaxis(0, 0, 1);
    osg::Matrix m0, m1;
    osg::Vec3 lp0(0.0, 0.0, 0.0), lp1(0.0, 1.0, 0.0), isectPoint;

    _s0->preFrame();
    _s1->preFrame();
    _s2->preFrame();
    _s3->preFrame();
    _s4->preFrame();

    if (showPickInteractor_ || showDirectInteractor_)
    {

        // register direct interactor if no intersection interactor is registered
        if (_s0->isRegistered() || _s1->isRegistered() || _s2->isRegistered() || _s3->isRegistered() || _s4->isRegistered())
        {
            //fprintf(stderr,"_s0->registered=%d _s2->registered=%d\n", _s0->registered ,_s2->registered);
            if (_directInteractor && _directInteractor->isRegistered())
            {
                coInteractionManager::the()->unregisterInteraction(_directInteractor);
                //fprintf(stderr,"unregister _directInteractor\n");
            }
        }
        else
        {
            if (_directInteractor && showDirectInteractor_ && !_directInteractor->isRegistered())
            {
                coInteractionManager::the()->registerInteraction(_directInteractor);
                //fprintf(stderr,"register _directInteractor\n");
            }
        }

        // direct interaction mode if direct interactor is registered and no intersection interactor is running
        if (_directInteractor && _directInteractor->isRegistered()
            && !_s0->isRunning() && !_s1->isRunning() && !_s2->isRunning()
            && !_s3->isRunning() && !_s4->isRunning())
        {

            if (cover->debugLevel(5))
                fprintf(stderr, "direct interaction mode\n");

            if (_newModule)
            {
                if (showPickInteractor_)
                {

                    hidePickInteractor();
                }
            }

            osg::Matrix currentHandMat, currentHandMat_o;
            osg::Vec3 currentHandPos, currentHandPos_o;

            currentHandMat = cover->getPointerMat();
            currentHandPos = currentHandMat.getTrans();
            currentHandMat_o = currentHandMat * cover->getInvBaseMat();
            currentHandPos_o = currentHandPos * cover->getInvBaseMat();

            if (_directInteractor && _directInteractor->wasStarted())
            {

                // set startpoint1
                _s0->updateTransform(currentHandMat_o);
                if (!_newModule)
                {
                    if (showPickInteractor_)
                    {
                        //hideGeometry();
                        hidePickInteractor();
                    }
                }

                // we need to update the coPlane
                m0 = _s0->getMatrix();
                _pos0 = m0.getTrans();

                //fprintf(stderr,"_pos0=[%f %f %f]\n", _pos0[0], _pos0[1], _pos0[2]);
                _pos2 = _pos0;

                _dir1 = osg::Matrix::transform3x3(xaxis, m0);
                _dir2 = osg::Matrix::transform3x3(yaxis, m0);
                _dir3 = osg::Matrix::transform3x3(zaxis, m0);
                _dir1.normalize();
                _dir2.normalize();

                //fprintf(stderr,"_dir2=[%f %f %f]\n", _dir2[0], _dir2[1], _dir2[2]);
                _dir3.normalize();

                _plane->update(_dir2, _pos0);
            }
            if (_directInteractor && _directInteractor->isRunning())
            {

                showGeometry();

                // pointer direction in world coordinates
                lp0 = lp0 * cover->getPointerMat();
                lp1 = lp1 * cover->getPointerMat();

                // pointer direction in object coordinates
                lp0 = lp0 * cover->getInvBaseMat();
                lp1 = lp1 * cover->getInvBaseMat();

                // get intersection point in object coordinates
                if (_plane->getLineIntersectionPoint(lp0, lp1, isectPoint))
                {
                    _pos2 = isectPoint;
                    //fprintf(stderr,"isectPoint=[%f %f %f]\n", isectPoint[0], isectPoint[1], isectPoint[2]);
                }

                _diag = _pos2 - _pos0;
                _diag.normalize();

                _c = 2.0 * (_pos0 - _pos2).length();
                float cosphi = _dir1 * _diag;
                _a = _c * cosphi;
                _b = sqrt(_c * _c - _a * _a);

                _pos1 = _pos0 - _diag * (0.5 * _c);
                _pos3 = _pos1 + _dir1 * _a;
                _pos4 = _pos2 - _dir1 * _a;

                updateGeometry();

                _s2->updateTransform(_pos2, _dir2);
                _s1->updateTransform(_pos1, _dir2);
            }

            if (_directInteractor && _directInteractor->wasStopped())
            {
                //fprintf(stderr,"_directInteractor->wasStopped\n");
                // pointer direction in world coordinates
                lp0 = lp0 * cover->getPointerMat();
                lp1 = lp1 * cover->getPointerMat();

                // pointer direction in object coordinates
                lp0 = lp0 * cover->getInvBaseMat();
                lp1 = lp1 * cover->getInvBaseMat();

                // get intersection point in object coordinates
                if (_plane->getLineIntersectionPoint(lp0, lp1, isectPoint))
                {
                    _pos2 = isectPoint;
                }

                _diag = _pos2 - _pos0;
                _diag.normalize();

                _c = 2.0 * (_pos0 - _pos2).length();
                float cosphi = _dir1 * _diag;
                _a = _c * cosphi;
                _b = sqrt(_c * _c - _a * _a);

                _pos1 = _pos0 - _diag * (0.5 * _c);
                _pos3 = _pos1 + _dir1 * _a;
                _pos4 = _pos2 - _dir1 * _a;

                _newModule = false;

                _s1->updateTransform(_pos1, _dir2);
                _s2->updateTransform(_pos2, _dir2);
                _s3->updateTransform(_pos3, _dir2);
                _s4->updateTransform(_pos4, _dir2);

                if (showPickInteractor_)
                {
                    showPickInteractor();
                    // geometry is already visible
                }
                else
                {
                    hideGeometry();
                }

                plugin->getSyncInteractors(_inter);
                setParameterStartpoint1(_pos1);
                setParameterStartpoint2(_pos2);
                setParameterDirection(_dir1);

                if (_execOnChange)
                    plugin->executeModule();

                /// hier nicht, weil coInteractionManager coTrackerButtonInteraction::RunningState nicht kennt
                /// und daher nicht auf Idle setzt -> abwarten bis zum n�chsten preFrame
                ///if (_directInteractor->registered)
                ///{
                ///   coInteractionManager::im->unregisterInteraction(_directInteractor);
                ///   fprintf(stderr,"unregister direct interactor\n");
                ///}
            }
        }

        // intersection mode
        else
        {
            if (cover->debugLevel(5))
            {
                fprintf(stderr, "tracer plane intersection mode\n");
                //fprintf(stderr,"s0 interactionState=%d runningState=%d\n", _s0->getState(), _s0->runningState);
            }
            if (_s0->isRunning())
            {
                //fprintf(stderr,"TracerPlane _s0->isRunning");
                // recompute all points
                m0 = _s0->getMatrix();
                _pos0 = m0.getTrans();

                _dir1 = osg::Matrix::transform3x3(xaxis, m0);
                _dir2 = osg::Matrix::transform3x3(yaxis, m0);
                _dir3 = osg::Matrix::transform3x3(zaxis, m0);
                _dir1.normalize();
                _dir2.normalize();
                _dir3.normalize();

                _pos2 = _pos0 + _dir1 * (0.5 * _a) + _dir3 * (0.5 * _b);
                _pos1 = _pos0 - _dir1 * (0.5 * _a) - _dir3 * (0.5 * _b);

                _pos3 = _pos1 + _dir1 * _a;
                _pos4 = _pos1 + _dir3 * _b;

                updateGeometry();
                _s1->updateTransform(_pos1, _dir2);
                _s2->updateTransform(_pos2, _dir2);
                _s3->updateTransform(_pos3, _dir2);
                _s4->updateTransform(_pos4, _dir2);
            }

            if (_s1->isRunning())
            {
                //fprintf(stderr,"_s1 isRunning\n");
                // _s2 is fixed, s1 interactively modified and s0, s3, s4 adjusted
                _pos1 = ((coPlane *)_s1)->getPosition();

                if (keepSquare_)
                    computeQuad1();
                else
                    computeQuad12();

                m0 = _s0->getMatrix();
                m0.setTrans(_pos0);
                _s0->updateTransform(m0);
                _s3->updateTransform(_pos3, _dir2);
                _s4->updateTransform(_pos4, _dir2);

                updateGeometry();
            }
            if (_s2->isRunning())
            {
                // s1 fixed, s2 interactively modified, and s0, s3, s4 adjusted
                _pos2 = ((coPlane *)_s2)->getPosition();

                if (keepSquare_)
                    computeQuad2();
                else
                    computeQuad12();

                m0 = _s0->getMatrix();
                m0.setTrans(_pos0);
                _s0->updateTransform(m0);
                _s3->updateTransform(_pos3, _dir2);
                _s4->updateTransform(_pos4, _dir2);
                updateGeometry();
            }
            if (_s3->isRunning())
            {
                _pos3 = ((coPlane *)_s3)->getPosition();

                if (keepSquare_)
                    computeQuad3();
                else
                    computeQuad34();

                m0 = _s0->getMatrix();
                m0.setTrans(_pos0);
                _s0->updateTransform(m0);
                _s1->updateTransform(_pos1, _dir2);
                _s2->updateTransform(_pos2, _dir2);
                updateGeometry();
            }
            if (_s4->isRunning())
            {
                _pos4 = ((coPlane *)_s4)->getPosition();

                if (keepSquare_)
                    computeQuad4();
                else
                    computeQuad34();

                m0 = _s0->getMatrix();
                m0.setTrans(_pos0);
                _s0->updateTransform(m0);
                _s1->updateTransform(_pos1, _dir2);
                _s2->updateTransform(_pos2, _dir2);
                updateGeometry();
            }
            if (_s0->wasStopped() || _s1->wasStopped() || _s2->wasStopped() || _s3->wasStopped() || _s4->wasStopped())
            {
                plugin->getSyncInteractors(_inter);
                setParameterStartpoint1(_pos1);
                setParameterStartpoint2(_pos2);
                setParameterDirection(_dir1);

                if (_execOnChange && !wait_)
                {
                    plugin->executeModule();
                    wait_ = true;
                }
            }
        }
    }
}

void
TracerPlane::computeQuad12()
{
    //fprintf(stderr,"computeQuad12\n");
    // gegeben: d1, pos1, pos2
    // gesucht pos3, pos4, pos0
    //   _dir3
    //   ^
    //   |
    //  _pos4         _pos2
    //   o-----------o
    //   |           |b
    //   |           |
    //   o-----------o------> _dir1
    //  _pos1  a     _pos3
    //
    _dir1.normalize();
    _diag = _pos2 - _pos1;
    _diag.normalize();
    _dir2 = _dir1 ^ _diag;
    _dir2.normalize();
    _dir2 = -_dir2;

    _c = (_pos1 - _pos2).length();
    float cosphi = _dir1 * _diag;
    _a = _c * cosphi;
    _b = sqrt(_c * _c - _a * _a);

    _pos3 = _pos1 + (_dir1 * _a);
    _pos4 = _pos2 - (_dir1 * _a);

    _dir3 = _pos4 - _pos1;
    _dir3.normalize();

    _pos0 = _pos1 + _diag * 0.5 * _c;

    //fprintf(stderr,"_pos0=[%f %f %f]\n", _pos0[0], _pos0[1], _pos0[2]);
    //fprintf(stderr,"_pos1=[%f %f %f]\n", _pos1[0], _pos1[1], _pos1[2]);
    //fprintf(stderr,"_pos2=[%f %f %f]\n", _pos2[0], _pos2[1], _pos2[2]);
    //fprintf(stderr,"_pos3=[%f %f %f]\n", _pos3[0], _pos3[1], _pos3[2]);
}

void
TracerPlane::computeQuad1()
{
    // gegeben: d1, pos1 (bewegt), pos2 (fix)
    // gesucht pso1, pos3, pos4, pos0
    //   _dir3
    //   ^
    //   |
    //  _pos4         _pos2
    //   o-----------o
    //   |           |b
    //   |           |
    //   o-----------o------> _dir1
    //  _pos1  a     _pos3
    //
    _dir1.normalize();
    _diag = _pos2 - _pos1;
    _diag.normalize();
    _dir2 = _dir1 ^ _diag;
    _dir2.normalize();
    _dir2 = -_dir2;

    _c = (_pos1 - _pos2).length();
    float cosphi = _dir1 * _diag;
    _a = _c * cosphi;
    _b = sqrt(_c * _c - _a * _a);

    if (_a < _b)
    {
        _pos1 = _pos1 + _dir1 * (_a - _b);
        _a = _b;
    }
    else
    {
        _pos1 = _pos1 + _dir3 * (_b - _a);
        _b = _a;
    }
    _pos3 = _pos1 + _dir1 * _a;
    _pos4 = _pos2 - _dir1 * _a;

    _dir3 = _pos4 - _pos1;
    _dir3.normalize();

    _c = sqrt(_a * _a + _b * _b);
    _diag = _pos2 - _pos1;
    _diag.normalize();

    _pos0 = _pos1 + _diag * (0.5 * _c);
}

void
TracerPlane::computeQuad2()
{
    // gegeben: d1, pos1 (fix), pos2 (bewegt)
    // gesucht pso1, pos3, pos4, pos0
    //   _dir3
    //   ^
    //   |
    //  _pos4         _pos2
    //   o-----------o
    //   |           |b
    //   |           |
    //   o-----------o------> _dir1
    //  _pos1  a     _pos3
    //
    _dir1.normalize();
    _diag = _pos2 - _pos1;
    _diag.normalize();
    _dir2 = _dir1 ^ _diag;
    _dir2.normalize();
    _dir2 = -_dir2;

    _c = (_pos1 - _pos2).length();
    float cosphi = _dir1 * _diag;
    _a = _c * cosphi;
    _b = sqrt(_c * _c - _a * _a);

    if (_a < _b)
    {
        _pos2 = _pos2 - _dir1 * (_a - _b);
        _a = _b;
    }
    else
    {
        _pos2 = _pos2 - _dir3 * (_b - _a);
        _b = _a;
    }
    _pos3 = _pos1 + _dir1 * _a;
    _pos4 = _pos2 - _dir1 * _a;

    _dir3 = _pos4 - _pos1;
    _dir3.normalize();

    _c = sqrt(_a * _a + _b * _b);
    _diag = _pos2 - _pos1;
    _diag.normalize();

    _pos0 = _pos1 + _diag * (0.5 * _c);
}

void
TracerPlane::computeQuad3()
{
    // gegeben: dir1, pos3 (bewegt), pos4(fix)
    // gesucht pos1, pos2, pos0
    //   _dir3
    //   ^
    //   |
    //  _pos4         _pos2
    //   o-----------o
    //   |           |b
    //   |           |
    //   o-----------o------> _dir1
    //  _pos1  a     _pos3
    //
    _dir1.normalize();
    //fprintf(stderr,"dir1=[%f %f %f]\n", _dir1[0], _dir1[1], _dir1[2]);
    _diag = _pos4 - _pos3;
    _diag.normalize();
    //fprintf(stderr,"diag=[%f %f %f]\n", _diag[0], _diag[1], _diag[2]);
    _dir2 = _dir1 ^ _diag;
    _dir2.normalize();
    //fprintf(stderr,"dir2=[%f %f %f]\n", _dir2[0], _dir2[1], _dir2[2]);
    ///_dir2=-_dir2;

    _c = (_pos3 - _pos4).length();
    float cosphi = -_dir1 * _diag;
    _a = _c * cosphi;
    _b = sqrt(_c * _c - _a * _a);

    if (_a < _b)
    {
        _pos3 = _pos3 - _dir1 * (_a - _b);
        _a = _b;
    }
    else
    {
        _pos3 = _pos3 + _dir3 * (_b - _a);
        _b = _a;
    }
    //fprintf(stderr,"a=%f b=%f c=%f\n", _a, _b, _c);
    _pos1 = _pos3 - _dir1 * _a;
    _pos2 = _pos4 + _dir1 * _a;

    _dir3 = _pos4 - _pos1;
    _dir3.normalize();

    _c = sqrt(_a * _a + _b * _b);
    _diag = _pos4 - _pos3;
    _diag.normalize();

    _pos0 = _pos3 + _diag * (0.5 * _c);
}

void
TracerPlane::computeQuad4()
{
    // gegeben: dir1, pos3(fix), pos4(bewegt)
    // gesucht pos1, pos2, pos0
    //   _dir3
    //   ^
    //   |
    //  _pos4         _pos2
    //   o-----------o
    //   |           |b
    //   |           |
    //   o-----------o------> _dir1
    //  _pos1  a     _pos3
    //
    _dir1.normalize();
    //fprintf(stderr,"dir1=[%f %f %f]\n", _dir1[0], _dir1[1], _dir1[2]);
    _diag = _pos4 - _pos3;
    _diag.normalize();
    //fprintf(stderr,"diag=[%f %f %f]\n", _diag[0], _diag[1], _diag[2]);
    _dir2 = _dir1 ^ _diag;
    _dir2.normalize();
    //fprintf(stderr,"dir2=[%f %f %f]\n", _dir2[0], _dir2[1], _dir2[2]);
    ///_dir2=-_dir2;

    _c = (_pos3 - _pos4).length();
    float cosphi = -_dir1 * _diag;
    _a = _c * cosphi;
    _b = sqrt(_c * _c - _a * _a);

    if (_a < _b)
    {
        _pos4 = _pos4 + _dir1 * (_a - _b);
        _a = _b;
    }
    else
    {
        _pos4 = _pos4 - _dir3 * (_b - _a);
        _b = _a;
    }
    //fprintf(stderr,"a=%f b=%f c=%f\n", _a, _b, _c);
    _pos1 = _pos3 - _dir1 * _a;
    _pos2 = _pos4 + _dir1 * _a;

    _dir3 = _pos4 - _pos1;
    _dir3.normalize();

    _c = sqrt(_a * _a + _b * _b);
    _diag = _pos4 - _pos3;
    _diag.normalize();

    _pos0 = _pos3 + _diag * 0.5 * _c;
}

void
TracerPlane::computeQuad34()
{
    // gegeben: dir1, pos3, pos4
    // gesucht pos1, pos2, pos0
    //   _dir3
    //   ^
    //   |
    //  _pos4         _pos2
    //   o-----------o
    //   |           |b
    //   |           |
    //   o-----------o------> _dir1
    //  _pos1  a     _pos3
    //
    _dir1.normalize();
    //fprintf(stderr,"dir1=[%f %f %f]\n", _dir1[0], _dir1[1], _dir1[2]);
    _diag = _pos4 - _pos3;
    _diag.normalize();
    //fprintf(stderr,"diag=[%f %f %f]\n", _diag[0], _diag[1], _diag[2]);
    _dir2 = _dir1 ^ _diag;
    _dir2.normalize();
    //fprintf(stderr,"dir2=[%f %f %f]\n", _dir2[0], _dir2[1], _dir2[2]);
    ///_dir2=-_dir2;

    _c = (_pos3 - _pos4).length();
    float cosphi = -_dir1 * _diag;
    _a = _c * cosphi;
    _b = sqrt(_c * _c - _a * _a);

    //fprintf(stderr,"a=%f b=%f c=%f\n", _a, _b, _c);
    _pos1 = _pos3 - _dir1 * _a;
    _pos2 = _pos4 + _dir1 * _a;

    _dir3 = _pos4 - _pos1;
    _dir3.normalize();

    _pos0 = _pos3 + _diag * 0.5 * _c;
}

void
TracerPlane::showPickInteractor()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nTracerPlane::show\n");

    showPickInteractor_ = true;

    _s0->show();
    _s0->enableIntersection();

    if (!_cyberclassroom)
    {
        _s1->show();
        _s2->show();
        _s3->show();
        _s4->show();

        _s1->enableIntersection();
        _s2->enableIntersection();
        _s3->enableIntersection();
        _s4->enableIntersection();
    }
    showGeometry();
}

void
TracerPlane::hidePickInteractor()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nTracerPlane::hide\n");

    showPickInteractor_ = false;

    _s0->hide();
    _s0->disableIntersection();
    _s1->hide();
    _s1->disableIntersection();
    _s2->hide();
    _s2->disableIntersection();
    _s3->hide();
    _s3->disableIntersection();
    _s4->hide();
    _s4->disableIntersection();

    hideGeometry();
}

void
TracerPlane::showDirectInteractor()
{
    showDirectInteractor_ = true;
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlane::showDirectInteractor\n");

    if (_directInteractor && !_directInteractor->isRegistered())
    {
        //fprintf(stderr,"register direct interactor\n");
        coInteractionManager::the()->registerInteraction(_directInteractor);
    }
}

void
TracerPlane::hideDirectInteractor()
{
    showDirectInteractor_ = false;

    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlane::hideDirectInteractor\n");

    if (!showPickInteractor_)
        hideGeometry();

    if (_directInteractor && _directInteractor->isRegistered())
    {
        //fprintf(stderr,"unregister direct interactor\n");
        coInteractionManager::the()->unregisterInteraction(_directInteractor);
    }
}

void
TracerPlane::showGeometry()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlane::showGeometry\n");

    if (geometryNode.get() && geometryNode->getNumParents() == 0)
    {
        parent->addChild(geometryNode.get());
    }
}

void
TracerPlane::hideGeometry()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlane::hideGeometry\n");

    if (geometryNode.get() && geometryNode->getNumParents())
    {
        geometryNode->getParent(0)->removeChild(geometryNode.get());
    }
}

bool
TracerPlane::wasStopped()
{
    if (_s1->wasStopped() || _s2->wasStopped() || _s0->wasStopped() || _s3->wasStopped() || _s4->wasStopped())
        return true;
    else
        return false;
}
bool
TracerPlane::wasStarted()
{
    if (_s1->wasStarted() || _s2->wasStarted() || _s0->wasStarted() || _s3->wasStarted() || _s4->wasStarted())
        return true;
    else
        return false;
}

bool
TracerPlane::isRunning()
{
    if (_s1->isRunning() || _s2->isRunning() || _s0->isRunning() || _s3->isRunning() || _s4->isRunning())
        return true;
    else
        return false;
}

osg::Vec3
TracerPlane::getParameterStartpoint1()
{
    float *sp1 = NULL;
    int dummy; // we know that the vector has 3 elements
    _inter->getFloatVectorParam(TracerInteraction::P_STARTPOINT1, dummy, sp1);
    osg::Vec3 startpoint1;
    if (sp1)
    {
        startpoint1.set(sp1[0], sp1[1], sp1[2]);
    }
    //fprintf(stderr,"TracerPlane::getParameterStartpoint1=[%f %f %f]\n", startpoint1[0], startpoint1[1], startpoint1[2]);
    return startpoint1;
}

void
TracerPlane::setParameterStartpoint1(osg::Vec3 sp1)
{
    plugin->setVectorParam(TracerInteraction::P_STARTPOINT1, 3, sp1._v);
}

osg::Vec3
TracerPlane::getParameterStartpoint2()
{
    float *sp2 = NULL;
    int dummy; // we know that the vector has 3 elements
    _inter->getFloatVectorParam(TracerInteraction::P_STARTPOINT2, dummy, sp2);
    osg::Vec3 startpoint2;
    if (sp2)
    {
        startpoint2.set(sp2[0], sp2[1], sp2[2]);
    }
    //fprintf(stderr,"TracerPlane::getParameterStartpoint2=[%f %f %f]\n", startpoint2[0], startpoint2[1], startpoint2[2]);

    return startpoint2;
}

void
TracerPlane::setParameterStartpoint2(osg::Vec3 sp2)
{
    plugin->setVectorParam(TracerInteraction::P_STARTPOINT2, 3, sp2._v);
}

osg::Vec3
TracerPlane::getParameterDirection()
{
    float *d = NULL;
    int dummy; // we know that the vector has 3 elements
    _inter->getFloatVectorParam(TracerInteraction::P_DIRECTION, dummy, d);
    osg::Vec3 direction;
    if (d)
    {
        direction.set(d[0], d[1], d[2]);
    }
    //fprintf(stderr,"TracerPlane::getParameterDirection=[%f %f %f]\n", direction[0], direction[1], direction[2]);

    return direction;
}

void
TracerPlane::setParameterDirection(osg::Vec3 d)
{
    plugin->setVectorParam(TracerInteraction::P_DIRECTION, 3, d._v);
}

void
TracerPlane::setStartpoint1(osg::Vec3 aVector)
{
    if (cover->debugLevel(3))
        std::cerr
            << "TracerPlane::setStartpoint1(osg::Vec3 aVector==("
            << aVector[0] << ", " << aVector[1] << ", " << aVector[2] << ")";
    _pos1 = aVector; //.set(sp1[0], sp1[1], sp1[2]);
    updatePlane();
}

void
TracerPlane::setStartpoint2(osg::Vec3 aVector)
{
    if (cover->debugLevel(3))
        std::cerr
            << "TracerPlane::setStartpoint2(osg::Vec3 aVector==("
            << aVector[0] << ", " << aVector[1] << ", " << aVector[2] << ")";
    _pos2 = aVector;
    updatePlane();
}

void
TracerPlane::setDirection(osg::Vec3 aVector)
{
    if (cover->debugLevel(3))
        std::cerr
            << "TracerPlane::setDirection(osg::Vec3 aVector==("
            << aVector[0] << ", " << aVector[1] << ", " << aVector[2] << ")";
    _dir1 = aVector;
    updatePlane();
}

void
TracerPlane::sendShowPickInteractorMsg()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlane::sendShowPickInteractorMsg\n");

    //fprintf(stderr,"in show COVER SEND INTERACTOR VISIBLE 1 object=%s\n", initialObjectName_);
    if (coVRMSController::instance()->isMaster())
    {
        coGRObjVisMsg visMsg(coGRMsg::INTERACTOR_VISIBLE, initialObjectName_.c_str(), 1);
        Message grmsg{ Message::UI , DataHandle{(char*)(visMsg.c_str()),strlen(visMsg.c_str()) + 1, false } };;
        cover->sendVrbMessage(&grmsg);
        //fprintf(stderr,"msg sent!\n");
    }
}

void
TracerPlane::sendHidePickInteractorMsg()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlane::sendHidePickInteractorMsg\n");

    //fprintf(stderr,"in show COVER SEND INTERACTOR VISIBLE 1 object=%s\n", initialObjectName_);
    if (coVRMSController::instance()->isMaster())
    {
        coGRObjVisMsg visMsg(coGRMsg::INTERACTOR_VISIBLE, initialObjectName_.c_str(), 0);
        Message grmsg{ Message::UI , DataHandle{(char*)(visMsg.c_str()),strlen(visMsg.c_str()) + 1, false } };;
        cover->sendVrbMessage(&grmsg);
        //fprintf(stderr,"msg sent!\n");
    }
}

void
TracerPlane::setCaseTransform(osg::MatrixTransform *t)
{
    parent = t;
    _s0->setCaseTransform(t);
    _s1->setCaseTransform(t);
    _s2->setCaseTransform(t);
    _s3->setCaseTransform(t);
    _s4->setCaseTransform(t);
}
// local variables:
// c-basic-offset: 3
// end:
