/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifdef WIN32
#pragma warning(disable : 4996)
#endif

#include "TracerInteraction.h"
#include "TracerLine.h"
#include "TracerPlane.h"
#include "TracerFreePoints.h"
#include "TracerPlugin.h"
#ifdef USE_COVISE
#include <CovisePluginUtil/SmokeGeneratorSolutions.h>
#include <alg/coUniTracer.h>
#endif

#include <cover/ui/Button.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Slider.h>
#include <cover/ui/SelectionList.h>

#include <cover/coVRPluginSupport.h>
#include <cover/coInteractor.h>
#include <cover/VRSceneGraph.h>

#include <PluginUtil/colors/ColorBar.h>

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
    , _numStartPointsPoti(NULL)
    , _numStartPointsMin(1)
    , _numStartPointsMax(1)
    , _numStartPoints(1)
    , traceLenPoti_(NULL)
    , traceLenMin_(0.f)
    , traceLenMax_(1.f)
    , traceLen_(1.f)
    , _numTaskTypes(0)
    , _taskTypeNames(NULL)
    , _selectedTaskType(0)
    , _numStartStyles(0)
    , _startStyleNames(NULL)
    , _selectedStartStyle(0)
    , _oldStartStyle(0)
    , _numTimeDirections(0)
    , _timeDirectionNames(NULL)
    , _selectedTimeDirection(0)
    , _tPlane(NULL)
    , _tLine(NULL)
    , _tFree(NULL)
    , smokeCheckbox_(NULL)
    , smokeInMenu_(false)
    , _containerName(NULL)
    , isComplex(false)
    , interDCS_(NULL)
    , plugin(NULL)
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
            smokeCheckbox_ = new ui::Button("Smoke", this);
            smokeCheckbox_->setState(false);
            smokeCheckbox_->setCallback([this](bool state){
                showSmoke_ = state;
            });
            if (_startStyle->selectedIndex() == STARTSTYLE_LINE || _startStyle->selectedIndex() == STARTSTYLE_PLANE)
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

        if ((_startStyle->selectedIndex() == STARTSTYLE_LINE || _startStyle->selectedIndex() == STARTSTYLE_PLANE)
                && smokeCheckbox_ && smokeInMenu_)
        {
            menu_->remove(smokeCheckbox_);
        }
        delete smokeCheckbox_;
        smokeCheckbox_ = NULL;
    }

    if (hideCheckbox_->state())
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
    hideCheckbox_->setState(hide);
    hideCheckbox_->trigger();

    _numStartPointsPoti = new ui::Slider(menu_, "NumStartPoints");
    _numStartPointsPoti->setPresentation(ui::Slider::AsDial);
    _numStartPointsPoti->setIntegral(true);
    _numStartPointsPoti->setCallback([this](double value, bool released){
        _numStartPoints = (int)value;
        inter_->setSliderParam(P_NO_STARTPOINTS, _numStartPointsMin, _numStartPointsMax, _numStartPoints);
        if (released)
            inter_->executeModule();
    });

    traceLenPoti_ = new ui::Slider(menu_, P_TRACE_LEN);
    traceLenPoti_->setCallback([this](double value, bool released){
        traceLen_ = value;
        inter_->setScalarParam(P_TRACE_LEN, traceLen_);
        if (released)
            inter_->executeModule();
    });

    _taskType = new ui::SelectionList(menu_, "TaskType");
    _taskType->setText("Task type");
    _taskType->append("Streamlines");
    _taskType->append("Moving points");
    _taskType->append("Pathlines");
    _taskType->append("Streaklines");
    _taskType->select(0);
    _taskType->setCallback([this](int idx){
        inter_->setChoiceParam(P_TASKTYPE, _numTaskTypes, _taskTypeNames, idx);
        inter_->executeModule();
    });
    _startStyle = new ui::SelectionList(menu_, "StartStyle");
    _startStyle->setText("Start style");
    _startStyle->append("Line");
    _startStyle->append("Plane");
    if (isComplex)
    {
        _startStyle->append("Free");
    }
    else
    {
        _startStyle->append("Cylinder");
    }
    _startStyle->setCallback([this](int idx){
        if (idx == STARTSTYLE_PLANE)
        {
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
        else if (idx == STARTSTYLE_LINE)
        {
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
        else if (idx == STARTSTYLE_FREE && isComplex)
        {
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
    });

    updateMenuContents();
}

void
TracerInteraction::updateMenuContents()
{
    if (cover->debugLevel(4))
        fprintf(stderr, "TracerInteraction::updateMenuContents \n");

    _numStartPointsPoti->setBounds(_numStartPointsMin, _numStartPointsMax);
    _numStartPointsPoti->setValue(_numStartPoints);

    if (traceLen_ > 1e-6)
        traceLenPoti_->setBounds(0., 5*traceLen_);
    else
        traceLenPoti_->setBounds(0., 1.);
    traceLenPoti_->setValue(traceLen_);
    if (_taskType)
        _taskType->select(_selectedTaskType);
    _startStyle->select(_selectedStartStyle);

    updatePickInteractors(showPickInteractor_);
    updateDirectInteractors(showDirectInteractor_);

    if (_selectedStartStyle == STARTSTYLE_LINE || _selectedStartStyle == STARTSTYLE_PLANE)
    {
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
    }
    else
    {
        if (smokeCheckbox_ && smokeInMenu_)
        {
            menu_->remove(smokeCheckbox_);
            smokeInMenu_ = false;
        }
    }
}

void
TracerInteraction::deleteMenuContents()
{
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
        hideCheckbox_->trigger();
        newObject_ = false;
    }
    _tPlane->preFrame();
    _tLine->preFrame();
    _tFree->preFrame();

    if (smokeCheckbox_ && smokeCheckbox_->state())
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
    updatePickInteractors(!hideCheckbox_->state() && showPickInteractorCheckbox_->state());
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
        if (smokeCheckbox_ && smokeCheckbox_->state())
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
            if (!smokeCheckbox_->state())
            {
                ////smokeCheckbox_->setState(true, true);
                smokeCheckbox_->setState(true);
            }
        }
        else
        {
            if (smokeCheckbox_->state())
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
    if (show && !hideCheckbox_->state())
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
