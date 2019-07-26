/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TracerInteraction.h"
#include "TracerLine.h"
#include "TracerPlugin.h"
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <cover/coVRPluginSupport.h>
#include <PluginUtil/coVR3DTransInteractor.h>
#include <PluginUtil/coVR3DTransRotInteractor.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRConfig.h>
#include <cover/coInteractor.h>
#include <cover/RenderObject.h>
#include <cover/coVRMSController.h>
#include <grmsg/coGRObjVisMsg.h>
#include <net/message.h>
#include <net/message_types.h>

#include <osg/MatrixTransform>
#include <osg/Geometry>

#include <config/CoviseConfig.h>

using namespace vrui;
using namespace grmsg;
using namespace opencover;

TracerLine::TracerLine(coInteractor *inter, TracerPlugin *p)
{

    if (cover->debugLevel(2))
        fprintf(stderr, "\nnew TracerLine\n");
    int dummy; // we know that the vector has 3 elements
    plugin = p;
    _inter = inter;
    _newModule = false;
    showPickInteractor_ = false;
    showDirectInteractor_ = false;
    _wait = false;

    // default size for all interactors
    _interSize = -1.f;
    // if defined, COVERConfig.ICON_SIZE overrides the default
    _interSize = coCoviseConfig::getFloat("COVER.IconSize", _interSize);
    // if defined, TracerPlugin.SCALEFACTOR overrides both
    _interSize = coCoviseConfig::getFloat("COVER.Plugin.Tracer.IconSize", _interSize);
    _execOnChange = coCoviseConfig::isOn("COVER.ExecuteOnChange", true);

    // create _s1, _s2
    float *sp1=NULL, *sp2=NULL;
    _inter->getFloatVectorParam(TracerInteraction::P_STARTPOINT1, dummy, sp1);
    _inter->getFloatVectorParam(TracerInteraction::P_STARTPOINT2, dummy, sp2);

    if (sp1)
        _pos1.set(sp1[0], sp1[1], sp1[2]);
    if (sp2)
        _pos2.set(sp2[0], sp2[1], sp2[2]);

    parent = cover->getObjectsScale();
    _s0 = new coVR3DTransRotInteractor(computeM0(), _interSize, coInteraction::ButtonA, "Menu", "Line_s0", coInteraction::Medium);
    _s1 = new coVR3DTransInteractor(_pos1, _interSize, coInteraction::ButtonA, "Menu", "Line_s1", coInteraction::Medium);
    _s2 = new coVR3DTransInteractor(_pos2, _interSize, coInteraction::ButtonA, "Menu", "Line_s2", coInteraction::Medium);

    initialObjectName_ = _inter->getObject()->getName();

    if (!coVRConfig::instance()->has6DoFInput())
    {
        _directInteractor = NULL;
        //fprintf(stderr,"no direct interaction with tracer line possible for TRACKING_SYSTEM MOUSE\n");
    }
    else
    {
        _directInteractor = new coTrackerButtonInteraction(coInteraction::ButtonA, "sphere", coInteraction::Medium);
    }
    createGeometry();

    hidePickInteractor();
    hideDirectInteractor();
    hideGeometry();
}

TracerLine::~TracerLine()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\ndelete TracerLine\n");

    delete _s1;
    delete _s2;
    delete _s0;

    if (_directInteractor)
        delete _directInteractor;
    deleteGeometry();
}

void
TracerLine::update(coInteractor *inter)
{
    float *sp1 = NULL, *sp2 = NULL, *d = NULL;
    int dummy; // we know that the vector has 3 elements
    _inter = inter;

    _inter->getFloatVectorParam(TracerInteraction::P_STARTPOINT1, dummy, sp1);
    _inter->getFloatVectorParam(TracerInteraction::P_STARTPOINT2, dummy, sp2);
    _inter->getFloatVectorParam(TracerInteraction::P_DIRECTION, dummy, d);
    if (sp1)
        _pos1.set(sp1[0], sp1[1], sp1[2]);
    if (sp2)
        _pos2.set(sp2[0], sp2[1], sp2[2]);

    // update
    _s1->updateTransform(_pos1);
    _s2->updateTransform(_pos2);
    _s0->updateTransform(computeM0());

    updateGeometry();

    _wait = false;
}

void
TracerLine::setNew()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "TracerLine::setNew\n");

    _newModule = true;

    if (_directInteractor && !_directInteractor->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(_directInteractor);
    }
}

void
TracerLine::createGeometry()
{
    osg::Vec4Array *color;
    coord = new osg::Vec3Array(2);
    updateGeometry();

    color = new osg::Vec4Array(1);
    (*color)[0].set(1, 0.5, 0.5, 1);

    geometry = new osg::Geometry();

    geometry->setVertexArray(coord.get());
    geometry->setColorArray(color);
    geometry->setColorBinding(osg::Geometry::BIND_OVERALL);
    geometry->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 2));
    geometry->setUseDisplayList(false);

    geometryNode = new osg::Geode();
    state = VRSceneGraph::instance()->loadUnlightedGeostate();
    geometryNode->setStateSet(state.get());
    geometryNode->addDrawable(geometry.get());
    geometryNode->setName("TracerLineGeometry");
    geometryNode->setNodeMask(geometryNode->getNodeMask() & (~Isect::Intersection) & (~Isect::Pick));

    parent->addChild(geometryNode.get());
}
void
TracerLine::deleteGeometry()
{
    if (geometryNode.get() && geometryNode->getNumParents())
    {
        parent->removeChild(geometryNode.get());
    }
}

void
TracerLine::updateGeometry()
{
    (*coord)[0].set(_pos1[0], _pos1[1], _pos1[2]);
    (*coord)[1].set(_pos2[0], _pos2[1], _pos2[2]);
	coord->dirty();
}

void
TracerLine::preFrame()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "TraceLine::preFrame\n");

    _s1->preFrame();
    _s2->preFrame();
    _s0->preFrame();

    if (showPickInteractor_ || showDirectInteractor_)
    {
        bool exec = false;

        // check if we are in direct interaction or intersection mode

        // direct interaction
        // is allowed if a new module was started
        // then the direct interactor was already registered

        // or if no interactor is intersected
        // then we have to register it
        if (_s1->isRegistered() || _s2->isRegistered() || _s0->isRegistered())
        {
            if (_directInteractor && showDirectInteractor_ && _directInteractor->isRegistered())
            {
                coInteractionManager::the()->unregisterInteraction(_directInteractor);
                //fprintf(stderr,"direct interactor unregistered\n");
            }
        }
        else
        {
            if (_directInteractor && showDirectInteractor_ && !_directInteractor->isRegistered())
            {
                coInteractionManager::the()->registerInteraction(_directInteractor);
                //fprintf(stderr,"direct interactor registered\n");
            }
        }

        // direct interaction mode
        // if button was pressed, interaction is started = line startpoint defined
        // s1 and s2 are disabled
        // s1 is movd to startpoint
        // if button is pressed, interaction is ongoing = line is drawn
        // if button was released, interaction is stopped = line endpoint defined
        // s2 is moved to line endpoint
        // s1 and s2 are enabled
        if (_directInteractor && _directInteractor->isRegistered()
                && !_s0->isRunning() && !_s1->isRunning() && !_s2->isRunning())
        {
            if (cover->debugLevel(5))
                fprintf(stderr, "tracer line direct interaction mode\n");

            if (_newModule)
            {
                hidePickInteractor();
            }

            osg::Matrix currentHandMat;
            osg::Vec3 currentHandPos, currentHandPos_o;

            currentHandMat = cover->getPointerMat();
            currentHandPos = currentHandMat.getTrans();
            currentHandPos_o = currentHandPos * cover->getInvBaseMat();

            osg::Vec3 iconPos, iconPos_o;
            //TODO_directInteractor->getIconWorldPosition(iconPos[0], iconPos[1], iconPos[2]);
            iconPos = currentHandPos;
            iconPos_o = iconPos * cover->getInvBaseMat();

            if (_directInteractor->wasStarted()) // button was pressed
            {
                if (cover->debugLevel(5))
                    fprintf(stderr, "tracer line direct interaction started\n");
                _pos1 = iconPos_o;
                _s1->updateTransform(_pos1); // move s1 to line startpoint
                if (!_newModule)
                {
                    hidePickInteractor();
                }

                showGeometry();
            }
            if (_directInteractor->isRunning()) // button pressed
            {
                if (cover->debugLevel(5))
                    fprintf(stderr, "tracer line direct interaction running\n");
                _pos2 = iconPos_o;
                showGeometry();
                updateGeometry();
            }
            if (_directInteractor->wasStopped()) // button released
            {
                if (cover->debugLevel(5))
                    fprintf(stderr, "tracer line direct interaction stopped\n");

                _s2->updateTransform(_pos2); // move s2 to line endpoint
                _s0->updateTransform(computeM0());

                _newModule = false;

                if (showPickInteractor_)
                {
                    showPickInteractor();
                }
                else
                {
                    hideGeometry();
                }

                plugin->getSyncInteractors(_inter);
                plugin->setVectorParam(TracerInteraction::P_STARTPOINT1, 3, _pos1._v);
                plugin->setVectorParam(TracerInteraction::P_STARTPOINT2, 3, _pos2._v);
                exec = true;

                /// hier nicht, weil coInteractionManager coTrackerButtonInteraction::RunningState nicht kennt
                /// und daher nicht auf Idle setzt -> abwarten bis zum n�chsten preFrame
                //if (_directInteractor->registered)
                //{
                //   coInteractionManager::im->unregisterInteraction(_directInteractor);
                //}
            }
        }

        // intersection mode
        else
        {
            if (cover->debugLevel(5))
                fprintf(stderr, "tracer line intersection mode\n");

            if (_s1->isRunning())
            {
                if (cover->debugLevel(5))
                    fprintf(stderr, "tracer line s1 intersection interaction running\n");

                _pos1 = _s1->getPos();
                updateGeometry();

                _s0->updateTransform(computeM0());
            }

            if (_s2->isRunning())
            {

                if (cover->debugLevel(5))
                    fprintf(stderr, "s2 intersection interaction running\n");

                _pos2 = _s2->getPos();
                updateGeometry();

                _s0->updateTransform(computeM0());
            }

            if (_s0->isRunning())
            {
                //fprintf(stderr,"line _s0 is running\n");
                osg::Matrix m0 = _s0->getMatrix();
                osg::Vec3 pos0;
                pos0 = m0.getTrans();
                osg::Vec3 xaxis(1, 0, 0), dir;
                dir = osg::Matrix::transform3x3(xaxis, m0);
                dir.normalize();
                float d = (_pos1 - _pos2).length();
                _pos1 = pos0 - dir * 0.5 * d;
                _pos2 = pos0 + dir * 0.5 * d;
                _s1->updateTransform(_pos1);
                _s2->updateTransform(_pos2);
                updateGeometry();
            }

            if (_s1->wasStopped())
            {
                if (cover->debugLevel(5))
                    fprintf(stderr, "s1 intersection interaction stopped\n");

                _pos1 = _s1->getPos();
                updateGeometry();

                _s0->updateTransform(computeM0());
                plugin->getSyncInteractors(_inter);
                plugin->setVectorParam(TracerInteraction::P_STARTPOINT1, 3, _pos1._v);
                exec = true;
            }

            if (_s2->wasStopped())
            {
                if (cover->debugLevel(5))
                    fprintf(stderr, "s2 intersection interaction stopped\n");

                _pos2 = _s2->getPos();
                updateGeometry();

                _s0->updateTransform(computeM0());
                plugin->getSyncInteractors(_inter);
                plugin->setVectorParam(TracerInteraction::P_STARTPOINT2, 3, _pos2._v);
                exec = true;
            }

            if (_s0->wasStopped())
            {
                osg::Matrix m0 = _s0->getMatrix();
                osg::Vec3 pos0;
                pos0 = m0.getTrans();
                osg::Vec3 xaxis(1, 0, 0), dir;
                dir = osg::Matrix::transform3x3(xaxis, m0);
                dir.normalize();
                float d = (_pos1 - _pos2).length();
                _pos1 = pos0 - dir * 0.5 * d;
                _pos2 = pos0 + dir * 0.5 * d;
                _s1->updateTransform(_pos1);
                _s2->updateTransform(_pos2);
                updateGeometry();

                if (cover->debugLevel(5))
                    fprintf(stderr, "s0 intersection interaction stopped\n");

                plugin->getSyncInteractors(_inter);
                plugin->setVectorParam(TracerInteraction::P_STARTPOINT1, 3, _pos1._v);
                plugin->setVectorParam(TracerInteraction::P_STARTPOINT2, 3, _pos2._v);
                exec = true;
            }
        }

        if (exec && _execOnChange && !_wait)
        {
            plugin->executeModule();
            _wait = true;
        }
    }
}

void
TracerLine::showPickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerLine::showPickInteractor\n");

    showPickInteractor_ = true;

    showGeometry();

    _s1->show();
    _s2->show();
    _s0->show();
    _s1->enableIntersection();
    _s2->enableIntersection();
    _s0->enableIntersection();
}

void
TracerLine::hidePickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerLine::hide\n");

    showPickInteractor_ = false;

    hideGeometry();

    _s0->hide();
    _s1->hide();
    _s2->hide();

    _s0->disableIntersection();
    _s1->disableIntersection();
    _s2->disableIntersection();
}

void
TracerLine::showDirectInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerLine::showDirectInteractor\n");

    showDirectInteractor_ = true;
    if (_directInteractor && !_directInteractor->isRegistered())
    {
        //fprintf(stderr,"register direct interactor\n");
        coInteractionManager::the()->registerInteraction(_directInteractor);
    }
}

void
TracerLine::hideDirectInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerLine::hideDirectInteractor\n");

    showDirectInteractor_ = false;

    if (!showPickInteractor_)
        hideGeometry();

    if (_directInteractor && _directInteractor->isRegistered())
    {
        //fprintf(stderr,"unregister direct interactor\n");
        coInteractionManager::the()->unregisterInteraction(_directInteractor);
    }
}

void
TracerLine::showGeometry()
{
    if (geometryNode.get() && geometryNode->getNumParents() == 0)
    {
        parent->addChild(geometryNode.get());
    }
}

void
TracerLine::hideGeometry()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerLine::hideGeometry\n");

    if (geometryNode.get() && geometryNode->getNumParents())
    {
        parent->removeChild(geometryNode.get());
    }
}

bool
TracerLine::wasStopped()
{
    if (_s1->wasStopped() || _s2->wasStopped() || _s0->wasStopped())
        return true;
    else
        return false;
}
bool
TracerLine::wasStarted()
{
    if (_s1->wasStarted() || _s2->wasStarted() || _s0->wasStarted())
        return true;
    else
        return false;
}

bool
TracerLine::isRunning()
{
    if (_s1->isRunning() || _s2->isRunning() || _s0->isRunning())
        return true;
    else
        return false;
}

osg::Matrix
TracerLine::computeM0()
{
    float d = (_pos1 - _pos2).length();
    osg::Vec3 dir = _pos2 - _pos1;
    dir.normalize();

    osg::Vec3 pos0 = _pos1 + dir * 0.5 * d;
    //fprintf(stderr,"_pos0=[%f %f %f]\n", _pos0[0], _pos0[1], _pos0[2]);
    osg::Vec3 xaxis(1, 0, 0);
    osg::Matrix m;
    m.makeRotate(xaxis, dir);
    m.setTrans(pos0);
    return (m);
}

void
TracerLine::setStartpoint(osg::Vec3 sp1)
{
    //fprintf(stderr, "TracerLine::setStartpoint(osg::Vec3 sp1[0]%f, sp1[1]%f, sp1[2]%f)", sp1[0], sp1[1], sp1[2]);

    _pos1.set(sp1[0], sp1[1], sp1[2]);
    _s1->updateTransform(_pos1);

    _s0->updateTransform(computeM0());

    updateGeometry();
}

void
TracerLine::setEndpoint(osg::Vec3 sp2)
{
    _pos2.set(sp2[0], sp2[1], sp2[2]);
    _s2->updateTransform(_pos2);

    _s0->updateTransform(computeM0());

    updateGeometry();
}

void
TracerLine::sendShowPickInteractorMsg()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerLine::sendShowPickInteractorMsg\n");

    if (coVRMSController::instance()->isMaster())
    {

        //fprintf(stderr,"in show COVER SEND INTERACTOR VISIBLE 1 object=%s\n", initialObjectName_.c_str());

        coGRObjVisMsg visMsg(coGRMsg::INTERACTOR_VISIBLE, initialObjectName_.c_str(), 1);
        Message grmsg{ COVISE_MESSAGE_VISENSO_UI , DataHandle{(char*)(visMsg.c_str()),strlen(visMsg.c_str()) + 1, false } };
        cover->sendVrbMessage(&grmsg);
        //fprintf(stderr,"msg sent!\n");
    }
}

void
TracerLine::sendHidePickInteractorMsg()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "\nTracerPlane::sendHidePickInteractorMsg\n");

    if (coVRMSController::instance()->isMaster())
    {

        //fprintf(stderr,"in show COVER SEND INTERACTOR VISIBLE 1 object=%s\n", initialObjectName_.c_str());

        coGRObjVisMsg visMsg(coGRMsg::INTERACTOR_VISIBLE, initialObjectName_.c_str(), 0);
        Message grmsg{ COVISE_MESSAGE_VISENSO_UI , DataHandle{(char*)(visMsg.c_str()),strlen(visMsg.c_str()) + 1, false } };
        grmsg.type = COVISE_MESSAGE_VISENSO_UI;
        cover->sendVrbMessage(&grmsg);
        //fprintf(stderr,"msg sent!\n");
    }
}

void
TracerLine::setCaseTransform(osg::MatrixTransform *t)
{
    parent = t;
    _s0->setCaseTransform(t);
    _s1->setCaseTransform(t);
    _s2->setCaseTransform(t);
}
