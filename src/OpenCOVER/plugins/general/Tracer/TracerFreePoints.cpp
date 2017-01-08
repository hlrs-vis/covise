/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TracerInteraction.h"
#include "TracerFreePoints.h"
#include "TracerPlugin.h"
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <PluginUtil/coVR3DTransInteractor.h>
#include <cover/coInteractor.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <config/CoviseConfig.h>

#include <osg/MatrixTransform>

using namespace vrui;
using namespace opencover;

TracerFreePoints::TracerFreePoints(coInteractor *inter, TracerPlugin *p)
{
    if (cover->debugLevel(2))
        fprintf(stderr, "TracerFreePoints::TracerFreePoints\n");
    plugin = p;
    _inter = inter;
    _newModule = false;
    showPickInteractor_ = false;
    showDirectInteractor_ = false;
    _waitForNewPoint = false;

    osg::Matrix currentHandMat;
    osg::Vec3 currentHandPos_o;

    // default size for all interactors
    _interSize = -2.f;
    // if defined, COVERConfig.ICON_SIZE overrides the default
    _interSize = coCoviseConfig::getFloat("COVER.IconSize", _interSize);
    // if defined, TracerPlugin.SCALEFACTOR overrides both
    _interSize = coCoviseConfig::getFloat("COVER.Plugin.Tracer.IconSize", _interSize);
    // use only half of the size for free start points
    _interSize = _interSize * 0.5;

    // get string
    ///_inter->getStringParam(TracerInteraction::P_FREESTARTPOINTS, str);
    // get string with already parsed points
    const char *str = _inter->getString(0);

    if (str && strlen(str) > 0)
    {

        istringstream tmp(str);
        tmp >> _numPoints;

        for (int i = 0; i < _numPoints; i++)
        {
            // parse string
            osg::Vec3 pos;
            tmp >> pos[0];
            tmp >> pos[1];
            tmp >> pos[2];

            // create interactor
            char iname[1024];
            sprintf(iname, "3dtransinteractor_%d", i);
            _pointsList.push_back(new coVR3DTransInteractor(currentHandPos_o, _interSize, coInteraction::ButtonA, "Menu", iname, coInteraction::Medium));
        }
    }
    else
    {
        //     fprintf(stderr,"\t INFO: freePointsList is empty\n");
    }

    if (!coVRConfig::instance()->has6DoFInput())
    {
        _directInteractor = NULL;
        //fprintf(stderr,"no direct interaction for tracer freepoints possible for TRACKING_SYSTEM MOUSE\n");
    }
    else
    {
        _directInteractor = new coTrackerButtonInteraction(coInteraction::ButtonA, "sphere", coInteraction::Medium);
    }
    hidePickInteractor();
    hideDirectInteractor();
}

TracerFreePoints::~TracerFreePoints()
{
    _pointsList.clear();

    if (_directInteractor)
        delete _directInteractor;
}

void
TracerFreePoints::update(coInteractor *inter)
{
    _inter = inter;

    const char *str = _inter->getString(0);
    if (!str)
        return;

    istringstream tmp(str);

    // check if the number of points changed
    int oldNumPoints = _numPoints;

    tmp >> _numPoints;

    if (_numPoints == oldNumPoints)
    {
        for (int i = 0; i < _numPoints; i++)
        {
            // parse string
            osg::Vec3 pos;
            tmp >> pos[0];
            tmp >> pos[1];
            tmp >> pos[2];

            // update interactor
            _pointsList[i]->updateTransform(pos);
        }
    }
    else
    {
        // delete old list
        _pointsList.clear();

        // append all viewpoints
        for (int i = 0; i < _numPoints; i++)
        {
            // parse string
            osg::Vec3 pos;
            tmp >> pos[0];
            tmp >> pos[1];
            tmp >> pos[2];

            // create interactor
            char iname[1024];
            sprintf(iname, "3dtransinteractor_%d", i);
            _pointsList.push_back(new coVR3DTransInteractor(pos, _interSize, coInteraction::ButtonA, "hand", iname, coInteraction::Medium));

            if (showPickInteractor_)
            {
                _pointsList.at(_pointsList.size() - 1)->show();
                _pointsList.at(_pointsList.size() - 1)->enableIntersection();
            }
            else
            {
                _pointsList.at(_pointsList.size() - 1)->hide();
                _pointsList.at(_pointsList.size() - 1)->disableIntersection();
            }
        }
    }
    _waitForNewPoint = false;
}

void
TracerFreePoints::setNew()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "TracerFreePoints::setNew\n");

    // Modul wurde kopiert
    // alte Interatoren sollen erhalten bleiben
    // der neue Modul soll keine Interaktoren in der Liste haben
    /// wird in preframe sowieso neu gesetzt_newPoint = true;

    _newModule = true;

    _pointsList.clear();
    _numPoints = 0;

    //fprintf(stderr,"pointslist now empty\n");
    if (_directInteractor && !_directInteractor->isRegistered())
    {
        coInteractionManager::the()->registerInteraction(_directInteractor);
    }
}

void
TracerFreePoints::preFrame()
{

    // call preFrame for all interactors
    for (std::vector<coVR3DTransInteractor *>::iterator it = _pointsList.begin(); it != _pointsList.end(); it++)
        (*it)->preFrame();

    if (showDirectInteractor_ || showPickInteractor_)
    {
        // direct interaction
        osg::Matrix currentHandMat;
        osg::Vec3 currentHandPos, currentHandPos_o;

        currentHandMat = cover->getPointerMat();
        currentHandPos = currentHandMat.getTrans();
        currentHandPos_o = currentHandPos * cover->getInvBaseMat();

        if (!_waitForNewPoint) // preframe is called several times before addObject
        { // and we don't want to start more than one point
            // set new point
            if (_directInteractor && _directInteractor->wasStopped())
            {

                char iname[1024];
                sprintf(iname, "3dtransinteractor_%d", _numPoints);
                _pointsList.push_back(new coVR3DTransInteractor(currentHandPos_o, _interSize, coInteraction::ButtonA, "hand", iname, coInteraction::Medium));

                _numPoints++;
                if (showPickInteractor_)
                {
                    _pointsList.at(_pointsList.size() - 1)->show();
                    _pointsList.at(_pointsList.size() - 1)->enableIntersection();
                }
                else
                {
                    _pointsList.at(_pointsList.size() - 1)->hide();
                    _pointsList.at(_pointsList.size() - 1)->disableIntersection();
                }
                char *tmpstr = new char[_pointsList.size() * 1024 + 1];
                tmpstr[0] = '\0';
                for (std::vector<coVR3DTransInteractor *>::iterator it = _pointsList.begin(); it != _pointsList.end(); it++)
                {
                    char tmp[1024];
                    osg::Vec3 p;
                    p = (*it)->getPos();

                    sprintf(tmp, "[%f,%f,%f]", p[0], p[1], p[2]);
                    //fprintf(stderr,"tmpstr=%s\n", tmpstr);
                    //fprintf(stderr,"tmp=%s\n", tmp);
                    strcat(tmpstr, tmp);
                }
                plugin->getSyncInteractors(_inter);
                plugin->setStringParam(TracerInteraction::P_FREE_STARTPOINTS, tmpstr);
                plugin->executeModule();
                _waitForNewPoint = true;
                delete[] tmpstr;

                _newModule = false;
            }
        }

        // intersection interaction
        for (std::vector<coVR3DTransInteractor *>::iterator it = _pointsList.begin(); it != _pointsList.end(); it++)
        {
            if ((*it)->wasStopped())
            {
                char *tmpstr = new char[_pointsList.size() * 1024 + 1];
                tmpstr[0] = '\0';
                for (std::vector<coVR3DTransInteractor *>::iterator it2 = _pointsList.begin(); it2 != _pointsList.end(); it2++)
                {
                    char tmp[1024];
                    osg::Vec3 p;
                    p = (*it2)->getPos();
                    sprintf(tmp, "[%f,%f,%f]", p[0], p[1], p[2]);
                    strcat(tmpstr, tmp);
                }
                plugin->getSyncInteractors(_inter);
                plugin->setStringParam(TracerInteraction::P_FREE_STARTPOINTS, tmpstr);
                //fprintf(stderr, "3dtransinter execute module\n");
                plugin->executeModule();
                //delete []tmpstr;
                break;
            }
        }
    }
}

void
TracerFreePoints::showPickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "TracerFreePoints::showPickInteractor\n");
    showPickInteractor_ = true;
    for (std::vector<coVR3DTransInteractor *>::iterator it = _pointsList.begin(); it != _pointsList.end(); it++)
    {
        (*it)->show();
        (*it)->enableIntersection();
    }
}

void
TracerFreePoints::hidePickInteractor()
{
    if (cover->debugLevel(3))
        fprintf(stderr, "TracerFreePoints::hidePickInteractor\n");
    showPickInteractor_ = false;
    for (std::vector<coVR3DTransInteractor *>::iterator it = _pointsList.begin(); it != _pointsList.end(); it++)
    {
        (*it)->hide();
        (*it)->disableIntersection();
    }
}

void
TracerFreePoints::showDirectInteractor()
{
    showDirectInteractor_ = true;
    if (_directInteractor && !_directInteractor->isRegistered())
        coInteractionManager::the()->registerInteraction(_directInteractor);
}

void
TracerFreePoints::hideDirectInteractor()
{
    if (_directInteractor && _directInteractor->isRegistered())
        coInteractionManager::the()->unregisterInteraction(_directInteractor);
}
void
TracerFreePoints::setCaseTransform(osg::MatrixTransform *t)
{
    //fprintf(stderr,"TracerFreePoints::setCaseTransform\n");
    for (std::vector<coVR3DTransInteractor *>::iterator it = _pointsList.begin(); it != _pointsList.end(); it++)
    {
        //fprintf(stderr,"+++++++++setting case for interactor(s)\n");
        (*it)->setCaseTransform(t);
    }
}
