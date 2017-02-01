/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "TracerInteraction.h"
#include "TracerLine.h"
#include "TracerPlane.h"
#include "TracerFreePoints.h"
#include "TracerPlugin.h"
#ifdef USE_COVISE
#include "../../covise/COVISE/SmokeGeneratorSolutions.h"
#include "alg/coUniTracer.h"
#endif

#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coPotiMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coButtonMenuItem.h>

#include <cover/coVRPluginSupport.h>
#include <cover/coInteractor.h>
#include <cover/VRSceneGraph.h>

#include <PluginUtil/ColorBar.h>

#include <config/CoviseConfig.h>

#include <cover/RenderObject.h>

#include <osg/MatrixTransform>

using namespace vrui;
using namespace opencover;

using namespace osg;

const char *TracerInteraction::P_NO_STARTPOINTS = "no_startp";
const char *TracerInteraction::P_STARTPOINT1 = "startpoint1";
const char *TracerInteraction::P_STARTPOINT2 = "startpoint2";
const char *TracerInteraction::P_DIRECTION = "direction";
const char *TracerInteraction::P_TDIRECTION = "tdirection";
const char *TracerInteraction::P_TASKTYPE = "taskType";
const char *TracerInteraction::P_STARTSTYLE = "startStyle";
const char *TracerInteraction::P_TRACE_LEN = "trace_len";
const char *TracerInteraction::P_FREE_STARTPOINTS = "FreeStartPoints";

TracerInteraction::TracerInteraction(const RenderObject *container, coInteractor *inter, const char *pluginName, TracerPlugin *p)
    : ModuleInteraction(container, inter, pluginName)
{
    // so far the base class created
    // the item in pinboard submenu COVISE
    // and the tracer menu without contents (menu_)
    plugin = p;

    isComplex = false;
    if (strcmp(inter->getModuleName(), "TracerComp") == 0)
    {
        isComplex = true;
    }

    interDCS_ = NULL;

    _containerName = NULL;

    smokeCheckbox_ = NULL;
    smokeInMenu_ = false;
    newObject_ = false;
    interactorUsed_ = true;
    showSmoke_ = false;
    //_colorBar=NULL;

    if (container)
    {
        _containerName = new char[strlen(container->getName()) + 1];
        strcpy(_containerName, container->getName());
    }

    // get the parameter values
    getParameters();

    _oldStartStyle = _selectedStartStyle;

    // create interactors
    _tLine = new TracerLine(inter, plugin);
    _tPlane = new TracerPlane(inter, plugin);
    _tFree = new TracerFreePoints(inter, plugin);

    // initialize smoke generator
    smokeRoot = new osg::Group();
    smokeRoot->setName("SmokeRoot");
    cover->getObjectsScale()->addChild(smokeRoot.get());

    smokeGeode_ = NULL;
    smokeGeometry_ = NULL;
    smokeColor_ = NULL;

    // create the menu contents
    createMenuContents();
    updateMenuContents();

    debugSmoke_ = coCoviseConfig::isOn("COVER.Plugin.Tracer.DebugSmoke", false);
}

TracerInteraction::~TracerInteraction()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "\nTracerInteraction::~TracerInteraction\n");

    deleteMenuContents();

    delete _tPlane;
    delete _tLine;
    delete _tFree;
    if (smokeRoot->getNumParents())
        cover->getObjectsScale()->removeChild(smokeRoot.get());
}

void
TracerInteraction::update(const RenderObject *container, coInteractor *inter)
{

    // base class updates the item in the COVISE menu
    // and the title of the Tracer menu
    ModuleInteraction::update(container, inter);

    if (container)
    {
        if (_containerName)
            delete[] _containerName;
        _containerName = new char[strlen(container->getName()) + 1];
        strcpy(_containerName, container->getName());
    }

    // get the current parameter values
    _oldStartStyle = _selectedStartStyle;
    getParameters();

    // update the submenu contents
    updateMenuContents();

    _tPlane->update(inter);
    _tLine->update(inter);
    _tFree->update(inter);
    if (interDCS_)
    {
        _tFree->setCaseTransform(interDCS_);
    }

    // we can't set the visisbility (node trav mask),
    // because in coVRNewInteractor the geometry is not yet added to the sg,
    // therefore we delay it tp next preFrame (or we should introduce coVRAddNode)
    newObject_ = true;
}

void
TracerInteraction::addSmoke(const RenderObject *grid, const RenderObject *velo)
{
    //fprintf(stderr,"TracerInteraction::addSmoke...");

    if (grid && velo)
    {
        if (debugSmoke_)
            fprintf(stderr, "grid and velo available\n");
        smoke_ = true;

        const RenderObject *ugrid = grid;
        const RenderObject *uvelo = velo;
        int nx, ny, nz;
        float xmin, xmax;
        float ymin, ymax;
        float zmin, zmax;
        ugrid->getSize(nx, ny, nz);
        ugrid->getMinMax(xmin, xmax, ymin, ymax, zmin, zmax);

        const float *u = velo->getFloat(Field::X);
        const float *v = velo->getFloat(Field::Y);
        const float *w = velo->getFloat(Field::Z);
        int nvx, nvy, nvz;
        uvelo->getSize(nvx, nvy, nvz);
        //assert(nx==nvx && ny==nvy && nz==nvz);

        xmin_ = xmin;
        xmax_ = xmax;
        ymin_ = ymin;
        ymax_ = ymax;
        zmin_ = zmin;
        zmax_ = zmax;
        nx_ = nx;
        ny_ = ny;
        nz_ = nz;

        u_.resize(nx_ * ny_ * nz_);
        v_.resize(nx_ * ny_ * nz_);
        w_.resize(nx_ * ny_ * nz_);
        if (nx_ * ny_ * nz_ > 0)
        {
            std::copy(u, u + nx_ * ny_ * nz_, &u_[0]);
            std::copy(v, v + nx_ * ny_ * nz_, &v_[0]);
            std::copy(w, w + nx_ * ny_ * nz_, &w_[0]);
        }

        if (!smokeCheckbox_)
        {
            if (cover->debugLevel(4))
                fprintf(stderr, "adding checkbox Smoke to menu\n");
            smokeCheckbox_ = new coCheckboxMenuItem("Smoke", 0);
            smokeCheckbox_->setMenuListener(this);
            if (_lineCheckbox->getState() || _planeCheckbox->getState())
            {
                menu_->add(smokeCheckbox_);
                smokeInMenu_ = true;
            }
        }
    }
    else
    {
        if (debugSmoke_)
            fprintf(stderr, "grid and velo not available\n");

        if ((_lineCheckbox->getState() || _planeCheckbox->getState()) && smokeCheckbox_ && smokeInMenu_)
            menu_->remove(smokeCheckbox_);
        delete smokeCheckbox_;
        smokeCheckbox_ = NULL;
    }

    if (hideCheckbox_->getState())
    {
        hideGeometry(true);
        hideSmoke();
        updatePickInteractors(false);
        updateDirectInteractors(false);
    }
}

void
TracerInteraction::getParameters()
{
    inter_->getIntSliderParam(P_NO_STARTPOINTS, _numStartPointsMin, _numStartPointsMax, _numStartPoints);
    inter_->getFloatScalarParam(P_TRACE_LEN, traceLen_);
    inter_->getChoiceParam(P_TASKTYPE, _numTaskTypes, _taskTypeNames, _selectedTaskType);
    inter_->getChoiceParam(P_STARTSTYLE, _numStartStyles, _startStyleNames, _selectedStartStyle);
    inter_->getChoiceParam(P_TDIRECTION, _numTimeDirections, _timeDirectionNames, _selectedTimeDirection);
}

void
TracerInteraction::createMenuContents()
{

    if (cover->debugLevel(3))
        fprintf(stderr, "TracerInteraction::createMenuContents\n");

    bool hide = coCoviseConfig::isOn("COVER.Plugin.Tracer.Hide", false);
    if (hide)
    {
        hideCheckbox_->setState(hide, true);
    }

    _numStartPointsPoti = new coPotiMenuItem(P_NO_STARTPOINTS, _numStartPointsMin, _numStartPointsMax, _numStartPoints);
    _numStartPointsPoti->setInteger(true);
    _numStartPointsPoti->setMenuListener(this);
    menu_->add(_numStartPointsPoti);

    traceLenPoti_ = new coPotiMenuItem(P_TRACE_LEN, 0, 5 * traceLen_, traceLen_);
    traceLenPoti_->setMenuListener(this);
    menu_->add(traceLenPoti_);

    _taskTypeButton = NULL;
    _taskTypeMenu = NULL;
    _taskTypeGroup = NULL;
    _streamlinesCheckbox = NULL;
    _particlesCheckbox = NULL;
    _pathlinesCheckbox = NULL;
    _streaklinesCheckbox = NULL;

    if (isComplex)
    {
        _taskTypeButton = new coSubMenuItem("TaskType:---");
        _taskTypeButton->setMenuListener(this);
        _taskTypeMenu = new coRowMenu("TaskType", menu_);
        _taskTypeGroup = new coCheckboxGroup(false);

        _streamlinesCheckbox = new coCheckboxMenuItem("Streamlines", false, _taskTypeGroup);
        _streamlinesCheckbox->setMenuListener(this);

        _particlesCheckbox = new coCheckboxMenuItem("Moving Points", false, _taskTypeGroup);
        _particlesCheckbox->setMenuListener(this);

        _pathlinesCheckbox = new coCheckboxMenuItem("Pathlines", false, _taskTypeGroup);
        _pathlinesCheckbox->setMenuListener(this);

        _streaklinesCheckbox = new coCheckboxMenuItem("StreakLines", false, _taskTypeGroup);
        _streaklinesCheckbox->setMenuListener(this);

        _taskTypeButton->setMenu(_taskTypeMenu);
        menu_->add(_taskTypeButton);

        _taskTypeMenu->add(_streamlinesCheckbox);
        _taskTypeMenu->add(_particlesCheckbox);
        _taskTypeMenu->add(_pathlinesCheckbox);
        _taskTypeMenu->add(_streaklinesCheckbox);
    }
    _startStyleButton = new coSubMenuItem("StartStyle:---");
    _startStyleButton->setMenuListener(this);
    _startStyleMenu = new coRowMenu("StartStyle", menu_);
    _startStyleGroup = new coCheckboxGroup(false);

    _planeCheckbox = new coCheckboxMenuItem("Plane", false, _startStyleGroup);
    _planeCheckbox->setMenuListener(this);

    _lineCheckbox = new coCheckboxMenuItem("Line", false, _startStyleGroup);
    _lineCheckbox->setMenuListener(this);
    _cylinderCheckbox = NULL;
    _freeCheckbox = NULL;
    if (isComplex)
    {
        _freeCheckbox = new coCheckboxMenuItem("Free", false, _startStyleGroup);
        _freeCheckbox->setMenuListener(this);
    }
    else
    {
        _cylinderCheckbox = new coCheckboxMenuItem("Cylinder", false, _startStyleGroup);
        _cylinderCheckbox->setMenuListener(this);
    }

    _startStyleButton->setMenu(_startStyleMenu);
    menu_->add(_startStyleButton);

    _startStyleMenu->add(_planeCheckbox);
    _startStyleMenu->add(_lineCheckbox);
    if (_freeCheckbox)
        _startStyleMenu->add(_freeCheckbox);
    if (_cylinderCheckbox)
        _startStyleMenu->add(_cylinderCheckbox);
}

void
TracerInteraction::updateMenuContents()
{
    if (cover->debugLevel(4))
        fprintf(stderr, "TracerInteraction::updateMenuContents \n");

    _numStartPointsPoti->setMin(_numStartPointsMin);
    _numStartPointsPoti->setMax(_numStartPointsMax);
    _numStartPointsPoti->setValue(_numStartPoints);

    traceLenPoti_->setMin(0);
    traceLenPoti_->setMax(5 * traceLen_);
    traceLenPoti_->setValue(traceLen_);
    if (isComplex)
    {
        switch (_selectedTaskType)
        {
        case TASKTYPE_STREAMLINES:
            _streamlinesCheckbox->setState(true);
            _particlesCheckbox->setState(false);
            _pathlinesCheckbox->setState(false);
            _streaklinesCheckbox->setState(false);
            _taskTypeButton->setName("TaskType: Streamlines");
            break;

        case TASKTYPE_PARTICLES:
            _particlesCheckbox->setState(true);
            _streamlinesCheckbox->setState(false);
            _pathlinesCheckbox->setState(false);
            _streaklinesCheckbox->setState(false);

            _taskTypeButton->setName("TaskType: Moving Points");
            break;

        case TASKTYPE_PATHLINES:
            _pathlinesCheckbox->setState(true);
            _streamlinesCheckbox->setState(false);
            _particlesCheckbox->setState(false);
            _streaklinesCheckbox->setState(false);

            _taskTypeButton->setName("TaskType: Pathlines");
            break;

        case TASKTYPE_STREAKLINES:
            _streaklinesCheckbox->setState(true);
            _streamlinesCheckbox->setState(false);
            _particlesCheckbox->setState(false);
            _pathlinesCheckbox->setState(false);
            _taskTypeButton->setName("TaskType: Streaklines");
            break;
        }
    }

    switch (_selectedStartStyle)
    {
    case STARTSTYLE_PLANE:
        _planeCheckbox->setState(true);
        _lineCheckbox->setState(false);
        if (_freeCheckbox)
            _freeCheckbox->setState(false);
        if (_cylinderCheckbox)
            _cylinderCheckbox->setState(false);
        _startStyleButton->setName("StartStyle: Plane");

        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);

        if (smokeCheckbox_ && !smokeInMenu_)
        {
            menu_->add(smokeCheckbox_);
            smokeInMenu_ = true;
        }
        if (smokeCheckbox_ && !smokeInMenu_)
        {
            menu_->add(smokeCheckbox_);
            smokeInMenu_ = true;
        }
        break;

    case STARTSTYLE_LINE:

        _lineCheckbox->setState(true);
        _planeCheckbox->setState(false);
        if (_freeCheckbox)
            _freeCheckbox->setState(false);
        if (_cylinderCheckbox)
            _cylinderCheckbox->setState(false);
        _startStyleButton->setName("StartStyle: Line");
        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);

        if (smokeCheckbox_ && !smokeInMenu_)
        {
            menu_->add(smokeCheckbox_);
            smokeInMenu_ = true;
        }
        break;

    case STARTSTYLE_FREE:

        if (isComplex)
        {
            if (_freeCheckbox)
                _freeCheckbox->setState(true);
            _planeCheckbox->setState(false);
            _lineCheckbox->setState(false);
            if (_cylinderCheckbox)
                _cylinderCheckbox->setState(false);
            _startStyleButton->setName("StartStyle: Free");
            updatePickInteractors(showPickInteractor_);
            updateDirectInteractors(showDirectInteractor_);

            if (smokeCheckbox_ && smokeInMenu_)
            {
                menu_->remove(smokeCheckbox_);
                smokeInMenu_ = false;
            }

            break;
        }
        else
        {

            if (_freeCheckbox)
                _freeCheckbox->setState(false);
            _planeCheckbox->setState(false);
            _lineCheckbox->setState(false);
            if (_cylinderCheckbox)
                _cylinderCheckbox->setState(true);
            _startStyleButton->setName("StartStyle: Cylinder");
            updatePickInteractors(showPickInteractor_);
            updateDirectInteractors(showDirectInteractor_);

            if (smokeCheckbox_ && smokeInMenu_)
            {
                menu_->remove(smokeCheckbox_);
                smokeInMenu_ = false;
            }

            break;
        }
    }
}

void
TracerInteraction::deleteMenuContents()
{

    delete _numStartPointsPoti;
    delete traceLenPoti_;

    delete _taskTypeButton;
    delete _taskTypeMenu;
    delete _streamlinesCheckbox;
    delete _particlesCheckbox;
    delete _pathlinesCheckbox;
    delete _streaklinesCheckbox;
    delete _taskTypeGroup;

    delete _startStyleButton;
    delete _startStyleMenu;
    delete _planeCheckbox;
    delete _lineCheckbox;
    delete _freeCheckbox;
    delete _cylinderCheckbox;
    delete _startStyleGroup;
}

void
TracerInteraction::preFrame()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "\nTracerInteraction::preFrame\n");

    // update visibility of new traces
    // in update the new geometry is not in the sg, either use addNode or delay it to preFrame
    if (newObject_ && hideCheckbox_ != NULL)
    {
        menuEvent(hideCheckbox_);
        newObject_ = false;
    }
    _tPlane->preFrame();
    _tLine->preFrame();
    _tFree->preFrame();

    if (smokeCheckbox_ && smokeCheckbox_->getState())
    {
        if (_tLine->wasStarted() || _tPlane->wasStarted())
        {
            if (showSmoke_)
                showSmoke();
            hideGeometry(true);
        }
        if (_tLine->wasStopped() || _tPlane->wasStopped())
        {
            hideSmoke();
            hideGeometry(false);
        }
        if (_tLine->isRunning())
        {
            updateSmokeLine();
        }
        if (_tPlane->isRunning())
        {
            updateSmokePlane();
        }
    }
}

void
TracerInteraction::showSmoke()
{
    //fprintf(stderr,"--------TracerInteraction::showSmoke\n");
    if (smokeRoot->getNumParents() == 0)
        cover->getObjectsScale()->addChild(smokeRoot.get());
}
void
TracerInteraction::hideSmoke()
{
    //fprintf(stderr,"--------TracerInteraction::hideSmoke\n");

    if (smokeRoot->getNumParents())
        cover->getObjectsScale()->removeChild(smokeRoot.get());
}

void
TracerInteraction::updateSmokeLine()
{
#ifdef USE_COVISE
    if (debugSmoke_)
        fprintf(stderr, "TracerInteraction::updateSmokeLine\n");

    // update solutions_
    vector<vector<coUniState> > solus;

    osg::Vec3 s1, s2, dir;
    float d;

    s1 = _tLine->getStartpoint();
    s2 = _tLine->getEndpoint();
    dir = s2 - s1;
    dir.normalize();
    d = (s1 - s2).length() / (_numStartPoints - 1);

    for (int i = 0; i < _numStartPoints; ++i)
    {
        float yarrini[3];
        yarrini[0] = s1[0] + i * d * dir[0];
        yarrini[1] = s1[1] + i * d * dir[1];
        yarrini[2] = s1[2] + i * d * dir[2];
        //fprintf(stderr,"startpoint: %f %f %f\n", yarrini[0], yarrini[1], yarrini[2]);
        coUniTracer unitracer(xmin_, xmax_, ymin_, ymax_, zmin_, zmax_, nx_, ny_, nz_,
                              &u_[0], &v_[0], &w_[0]);
        vector<coUniState> solution;
        if (_selectedTimeDirection == 0)
        {
            unitracer.solve(yarrini, solution, 0.0001, traceLen_);
            solus.push_back(solution);
        }
        else if (_selectedTimeDirection == 1)
        {
            unitracer.solve(yarrini, solution, 0.0001, traceLen_, 0.005, -1);
            solus.push_back(solution);
        }
        else if (_selectedTimeDirection == 2)
        {
            unitracer.solve(yarrini, solution, 0.0001, traceLen_ / 2);
            solus.push_back(solution);
            unitracer.solve(yarrini, solution, 0.0001, traceLen_ / 2, 0.005, -1);
            solus.push_back(solution);
        }
    }
    solutions_.set(solus);

    displaySmoke();
#endif
}

void
TracerInteraction::updateSmokePlane()
{
#ifdef USE_COVISE

    if (debugSmoke_)
        fprintf(stderr, "TracerInteraction::updateSmokePlane\n");

    // update solutions_
    vector<vector<coUniState> > solus;

    osg::Vec3 s1, s2, s3, s4, dir1, dir2;
    float d0, d1; // distance between two startpoints
    int n0, n1;
    float r, s;

    s1 = _tPlane->getStartpoint();
    s2 = _tPlane->getEndpoint();
    s3 = _tPlane->getPos3();
    s4 = _tPlane->getPos4();
    ///dir1=_tPlane->getDirection1();
    dir1 = s3 - s1;
    dir1.normalize();
    ///dir2=_tPlane->getDirection2();
    dir2 = s4 - s1;
    dir2.normalize();

    // side lengths
    r = (s3 - s1).length();
    s = (s3 - s2).length();

    if (r > s)
    {
        n1 = int(sqrt(_numStartPoints * s / r)) + 1;
        if (n1 <= 1)
            n1 = 2;
        n0 = _numStartPoints / n1;
        if (n0 <= 1)
            n0 = 2;
    }
    else
    {
        n0 = int(sqrt(_numStartPoints * r / s)) + 1;
        if (n0 <= 1)
            n0 = 2;
        n1 = _numStartPoints / n0;
        if (n1 <= 1)
            n1 = 2;
    }

    d0 = r / (n0 - 1);
    d1 = s / (n1 - 1);

    //fprintf(stderr,"n0=%d n1=%d d0=%f d1=%f\n", n0, n1, d0, d1);
    //fprintf(stderr,"dir1=[%f %f %f]\n", dir1[0], dir1[1], dir1[2]);
    //fprintf(stderr,"dir2=[%f %f %f]\n", dir2[0], dir2[1], dir2[2]);
    for (int i = 0; i < n0; ++i)
    {
        for (int j = 0; j < n1; ++j)
        {
            float yarrini[3];
            yarrini[0] = s1[0] + i * d0 * dir1[0] + j * d1 * dir2[0];
            yarrini[1] = s1[1] + i * d0 * dir1[1] + j * d1 * dir2[1];
            yarrini[2] = s1[2] + i * d0 * dir1[2] + j * d1 * dir2[2];
            //fprintf(stderr,"startpoint at i=%d j=%d: %f %f %f\n", i, j, yarrini[0], yarrini[1], yarrini[2]);
            coUniTracer unitracer(xmin_, xmax_, ymin_, ymax_, zmin_, zmax_, nx_, ny_, nz_,
                                  &u_[0], &v_[0], &w_[0]);
            vector<coUniState> solution;
            unitracer.solve(yarrini, solution, 0.0001, traceLen_);
            solus.push_back(solution);
        }
    }
    solutions_.set(solus);

    displaySmoke();
#endif
}

void
TracerInteraction::menuEvent(coMenuItem *item)
{

    if (cover->debugLevel(4))
        fprintf(stderr, "TracerInteraction::menuEvent %s\n", item->getName());
    if (item == _numStartPointsPoti)
    {
        _numStartPoints = (int)_numStartPointsPoti->getValue();
        inter_->setSliderParam(P_NO_STARTPOINTS, _numStartPointsMin, _numStartPointsMax, _numStartPoints);
        if (cover->getPointerButton()->wasReleased())
            inter_->executeModule();
    }

    else if (item == traceLenPoti_)
    {
        traceLen_ = traceLenPoti_->getValue();
        inter_->setScalarParam(P_TRACE_LEN, traceLen_);
        if (cover->getPointerButton()->wasReleased())
            inter_->executeModule();
    }

    else if (item == _streamlinesCheckbox)
    {
        _taskTypeButton->setName("TaskType: Streamlines");
        _taskTypeButton->closeSubmenu();
        inter_->setChoiceParam(P_TASKTYPE, _numTaskTypes, _taskTypeNames, TASKTYPE_STREAMLINES);

        inter_->executeModule();
    }
    else if (item == _particlesCheckbox)
    {
        _taskTypeButton->setName("TaskType: Moving Point");
        _taskTypeButton->closeSubmenu();
        inter_->setChoiceParam(P_TASKTYPE, _numTaskTypes, _taskTypeNames, TASKTYPE_PARTICLES);

        inter_->executeModule();
    }

    else if (item == _pathlinesCheckbox)
    {
        _taskTypeButton->setName("TaskType: Pathlines");
        _taskTypeButton->closeSubmenu();
        inter_->setChoiceParam(P_TASKTYPE, _numTaskTypes, _taskTypeNames, TASKTYPE_PATHLINES);

        inter_->executeModule();
    }

    else if (item == _streaklinesCheckbox)
    {
        _taskTypeButton->setName("TaskType: Streaklines");
        _taskTypeButton->closeSubmenu();
        inter_->setChoiceParam(P_TASKTYPE, _numTaskTypes, _taskTypeNames, TASKTYPE_STREAKLINES);

        inter_->executeModule();
    }

    else if (item == _planeCheckbox)
    {
        _startStyleButton->setName("StartStyle: Plane");
        _startStyleButton->closeSubmenu();
        inter_->setChoiceParam(P_STARTSTYLE, _numStartStyles, _startStyleNames, STARTSTYLE_PLANE);
        _selectedStartStyle = STARTSTYLE_PLANE;
        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);
        if (smokeCheckbox_ && !smokeInMenu_)
        {
            menu_->add(smokeCheckbox_);
            smokeInMenu_ = true;
        }

        if (smokeCheckbox_ && !smokeInMenu_)
        {
            menu_->add(smokeCheckbox_);
            smokeInMenu_ = true;
        }

        inter_->executeModule();
    }

    else if (item == _lineCheckbox)
    {
        _startStyleButton->setName("StartStyle: Line");
        _startStyleButton->closeSubmenu();
        inter_->setChoiceParam(P_STARTSTYLE, _numStartStyles, _startStyleNames, STARTSTYLE_LINE);
        _selectedStartStyle = STARTSTYLE_LINE;
        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);
        if (smokeCheckbox_ && !smokeInMenu_)
        {
            menu_->add(smokeCheckbox_);
            smokeInMenu_ = true;
        }

        inter_->executeModule();
    }

    else if (item == _freeCheckbox)
    {
        _startStyleButton->setName("StartStyle: Free");
        _startStyleButton->closeSubmenu();
        inter_->setChoiceParam(P_STARTSTYLE, _numStartStyles, _startStyleNames, STARTSTYLE_FREE);
        _selectedStartStyle = STARTSTYLE_FREE;
        updatePickInteractors(showPickInteractor_);
        updateDirectInteractors(showDirectInteractor_);

        if (smokeCheckbox_ && smokeInMenu_)
        {
            menu_->remove(smokeCheckbox_);
            smokeInMenu_ = false;
        }
        inter_->executeModule();
    }

    else if (smokeCheckbox_)
    {
        if (smokeCheckbox_->getState())
        {
            showSmoke_ = true;
        }
        else
        {
            showSmoke_ = false;
        }
    }

    else // other menu actions are treated by the base class
    {
        ModuleInteraction::menuEvent(item);
    }
}

void
TracerInteraction::menuReleaseEvent(coMenuItem *item)
{

    if (item == _numStartPointsPoti)
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "TracerInteraction::menuReleaseEvent for _numStartPointsPoti\n");

        _numStartPoints = (int)_numStartPointsPoti->getValue();
        inter_->setSliderParam(P_NO_STARTPOINTS, _numStartPointsMin, _numStartPointsMax, _numStartPoints);
        inter_->executeModule();
    }

    if (item == traceLenPoti_)
    {
        if (cover->debugLevel(3))
            fprintf(stderr, "TracerInteraction::menuReleaseEvent for traceLenPointsPoti\n");

        traceLen_ = traceLenPoti_->getValue();
        inter_->setScalarParam(P_TRACE_LEN, traceLen_);
        inter_->executeModule();
    }
}

/*
void TracerInteraction::setNew()
{
   if (_selectedStartStyle==STARTSTYLE_LINE)
{
      _tLine->setNew();
}
   else if (_selectedStartStyle==STARTSTYLE_PLANE)
{
      _tPlane->setNew();
}
   else if (_selectedStartStyle==STARTSTYLE_FREE)
{
      _tFree->setNew();
}
}
*/

void
TracerInteraction::displaySmoke()
{
    if (debugSmoke_)
        fprintf(stderr, "TracerInteraction::displaySmoke\n");

    if (!smokeGeode_) //firsttime
    {
        smokeGeode_ = new osg::Geode();
        smokeRoot->addChild(smokeGeode_.get());
    }

    if (!smokeGeometry_)
    {
        smokeGeometry_ = new osg::Geometry();
        smokeGeometry_->setStateSet(VRSceneGraph::instance()->loadDefaultGeostate());
        smokeGeode_->addDrawable(smokeGeometry_.get());
    }

    // remove last primitives
    for (unsigned int i = 0; i < smokeGeometry_->getNumPrimitiveSets(); i++)
        smokeGeometry_->removePrimitiveSet(i);

    // set color
    smokeColor_ = new osg::Vec4Array();
    smokeColor_->push_back(Vec4(1, 0.5, 0.5, 1));
    smokeGeometry_->setColorArray(smokeColor_.get());
    smokeGeometry_->setColorBinding(osg::Geometry::BIND_OVERALL);
    smokeGeometry_->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    int numPoints = 0;
#ifdef USE_COVISE
    for (int i = 0; i < solutions_.size(); i++)
    {
        //fprintf(stderr,"line %d has %d points\n", i, (const_cast<int *>(solutions_.lengths())[i]));
        numPoints += solutions_.lengths()->at(i);
    }
    if (debugSmoke_)
        fprintf(stderr, "TracerInteraction::displaySmoke %d points\n", numPoints);

    if (numPoints != 0)
    {
        smokeGeometry_->addPrimitiveSet(solutions_.lengths());
        smokeGeometry_->setVertexArray(solutions_.linepoints());
    }
    else
    {
        // draw one point
        Vec3 s0 = _tPlane->getPos0();
        ref_ptr<Vec3Array> points = new Vec3Array();
        points->push_back(Vec3(s0[0], s0[1], s0[2]));
        smokeGeometry_->setVertexArray(points.get());
        ref_ptr<DrawArrayLengths> primitives = new DrawArrayLengths(PrimitiveSet::POINTS);
        primitives->push_back(1);
        smokeGeometry_->addPrimitiveSet(primitives.get());
    }
#endif
}

void TracerInteraction::updatePickInteractorVisibility()
{

    // if geometry is hidden, hide also interactor
    updatePickInteractors(!hideCheckbox_->getState() && showPickInteractorCheckbox_->getState());
}

void
TracerInteraction::setStartpoint1FromGui(float x, float y, float z)
{
    //fprintf(stderr,"TracerInteraction::setStartpoint1 %f %f %f\n", x, y, z);
    //showInteractor(true);
    if (_selectedStartStyle == STARTSTYLE_LINE)
    {
        Vec3 pos(x, y, z);
        _tLine->setStartpoint(pos);
    }
    if (STARTSTYLE_PLANE == _selectedStartStyle)
    {
        Vec3 pos(x, y, z);
        _tPlane->setStartpoint1(pos);
        //updateSmokePlane();
    }
}

void
TracerInteraction::setStartpoint2FromGui(float x, float y, float z)
{
    //fprintf(stderr,"TracerInteraction::setStartpoint2 %f %f %f\n", x, y, z);

    //showInteractor(true);
    if (_selectedStartStyle == STARTSTYLE_LINE)
    {
        Vec3 pos(x, y, z);
        _tLine->setEndpoint(pos);
    }

    if (STARTSTYLE_PLANE == _selectedStartStyle)
    {
        Vec3 pos(x, y, z);
        _tPlane->setStartpoint2(pos);
        //updateSmokePlane();
    }
}

void
TracerInteraction::setDirectionFromGui(float x, float y, float z)
{
    //fprintf(stderr,"TracerInteraction::setDirection %f %f %f\n", x, y, z);

    //showInteractor(true);
    if (STARTSTYLE_PLANE == _selectedStartStyle)
    {
        Vec3 pos(x, y, z);
        _tPlane->setDirection(pos);
        if (smokeCheckbox_ && smokeCheckbox_->getState())
        {
            if (guiSliderFirsttime_)
            {
                //fprintf(stderr,"guiSliderFirsttime_\n");
                //showSmoke(true);
                ModuleFeedbackManager::hideGeometry(true);
                guiSliderFirsttime_ = false;
            }
            //updateSmokePlane();
        }
    }
}

/*
void 
TracerInteraction::setUseInteractorFromGui(bool use)
{
 
   if (use && !interactorUsed_)
{
      if (showDirectInteractorCheckbox_)
{
         menu_->insert(showDirectInteractorCheckbox_, 0);
}

      if (smokeCheckbox_ && !smokeInMenu_)
{
         menu_->add(smokeCheckbox_);
         smokeInMenu_=true;
}

      menu_->insert(showPickInteractorCheckbox_, 0);

      interactorUsed_=true;
}
   if (!use && interactorUsed_)
{
      if (showDirectInteractorCheckbox_)
{
         ////showDirectInteractorCheckbox_->setState(false, true);
         showDirectInteractorCheckbox_->setState(false);
         _tPlane->hideDirectInteractor();

         menu_->remove(showDirectInteractorCheckbox_);
}

      ////showPickInteractorCheckbox_->setState(false, true);
      showPickInteractorCheckbox_->setState(false);
      _tPlane->hidePickInteractor();
      
      menu_->remove(showPickInteractorCheckbox_);

      if (smokeCheckbox_ && smokeInMenu_)
{
         menu_->remove(smokeCheckbox_);
         smokeInMenu_=false;
}

      interactorUsed_=false;
}
}
*/

void
TracerInteraction::setShowSmokeFromGui(bool state)
{
    //fprintf(stderr,"\n\n\n *** TracerInteraction::setShowSmokeCheckbox %d ***\n\n\n", state);
    if (smokeCheckbox_)
    {
        if (state)
        {
            if (!smokeCheckbox_->getState())
            {
                ////smokeCheckbox_->setState(true, true);
                smokeCheckbox_->setState(true);
            }
        }
        else
        {
            if (smokeCheckbox_->getState())
            {
                smokeCheckbox_->setState(false);
            }
        }
    }
    else
    {
        if (cover->debugLevel(5))
            fprintf(stderr, "showSmokeCheckbox not yet created\n!");
        //       smokeVisibleAtStart_=true;
    }

    updatePickInteractorVisibility();
}

void
TracerInteraction::interactorSetCaseFromGui(const char *caseName)
{
    //fprintf(stderr,"TracerInteraction::interactorSetCaseFromGui case=%s \n", caseName);

    string interactorCaseName(caseName);
    interactorCaseName += "_INTERACTOR";
    interDCS_ = VRSceneGraph::instance()->findFirstNode<osg::MatrixTransform>(interactorCaseName.c_str(), false, cover->getObjectsScale());
    if (!interDCS_)

    {
        // firsttime we create also a case DCS
        //fprintf(stderr,"ModuleFeedbackManager::setCaseFromGui create case DCS\n");
        interDCS_ = new osg::MatrixTransform();
        interDCS_->setName(interactorCaseName.c_str());
        cover->getObjectsScale()->addChild(interDCS_);
    }

    if (interDCS_)
    {
        _tPlane->setCaseTransform(interDCS_);
        _tLine->setCaseTransform(interDCS_);
        _tFree->setCaseTransform(interDCS_);
    }
    //   else
    //fprintf(stderr,"TracerInteraction::interactorSetCaseFromGui didn't find case dcs\n");
}

void TracerInteraction::updatePickInteractors(bool show)
{
    if (show && !hideCheckbox_->getState())
    {
        if (_selectedStartStyle == STARTSTYLE_LINE)
        {
            _tLine->showPickInteractor();
            _tPlane->hidePickInteractor();
            _tFree->hidePickInteractor();
        }
        else if (_selectedStartStyle == STARTSTYLE_PLANE)
        {
            _tLine->hidePickInteractor();
            _tPlane->showPickInteractor();
            _tFree->hidePickInteractor();
        }
        else if (_selectedStartStyle == STARTSTYLE_FREE)
        {
            _tLine->hidePickInteractor();
            _tPlane->hidePickInteractor();
            _tFree->showPickInteractor();
        }
    }
    else
    {
        _tPlane->hidePickInteractor();
        _tLine->hidePickInteractor();
        _tFree->hidePickInteractor();
    }
}

void TracerInteraction::updateDirectInteractors(bool show)
{

    if (show)
    {
        if (_selectedStartStyle == STARTSTYLE_LINE)
        {
            _tLine->showDirectInteractor();
            _tPlane->hideDirectInteractor();
            _tFree->hideDirectInteractor();
        }
        else if (_selectedStartStyle == STARTSTYLE_PLANE)
        {
            _tLine->hideDirectInteractor();
            _tPlane->showDirectInteractor();
            _tFree->hideDirectInteractor();
        }
        else if (_selectedStartStyle == STARTSTYLE_FREE)
        {
            _tLine->hideDirectInteractor();
            _tPlane->hideDirectInteractor();
            _tFree->showDirectInteractor();
        }
    }
    else
    {
        _tPlane->hideDirectInteractor();
        _tLine->hideDirectInteractor();
        _tFree->hideDirectInteractor();
    }
}
