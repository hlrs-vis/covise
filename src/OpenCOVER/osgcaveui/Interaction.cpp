/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// C++:
#include <iostream>

// OSG:
#include <osg/LineSegment>
#include <osgUtil/IntersectVisitor>

// Local:
#include "CUI.h"
#include "Interaction.h"
#include "PickBox.h"

using namespace osg;
using namespace cui;
using namespace std;

Interaction::Interaction(Group *worldRoot, Group *objectRoot, LogFile *logFile)
{
    _worldRoot = worldRoot;
    _objectRoot = objectRoot;
    _head = new InputDevice(this, _worldRoot, 3);
    _wandR = new InputDevice(this, _worldRoot, 3);
    _wandL = new InputDevice(this, _worldRoot, 3);
    _mouse = new InputDevice(this, _worldRoot, 3);
    _logFile = logFile;

    _gazeInteraction = false;
}

Interaction::~Interaction()
{
    delete _head;
    delete _wandR;
    delete _wandL;
    delete _mouse;
    _wandR = NULL;
    _head = NULL;
    _wandL = NULL;
    _mouse = NULL;
}

Matrix Interaction::getW2O()
{
    Matrix o2w = CUI::computeLocal2Root(_objectRoot);
    Matrix w2o = Matrix::inverse(o2w);
    return w2o;
}

/** Called every frame to process user interaction.
  @return true if any input device action returned true
*/
bool Interaction::action()
{
    bool ret = false;
    if (_gazeInteraction)
    {
        if (_head->action())
            ret = true;
    }
    if (_wandR != NULL)
    {
        if (_wandR->action())
            ret = true;
    }
    //  if (_wandL->action()) ret = true;
    //  if (_mouse->action()) ret = true;
    return ret;
}

void Interaction::addListener(Events *events, Widget *widget)
{
    _widgetInfoList.push_back(new WidgetInfo(events, widget));
}

void Interaction::addListener(PickBox *box)
{
    _boxList.push_back(new WidgetInfo(box));
}

void Interaction::addAnyButtonListener(Events *events, Widget *widget)
{
    _anyButtonListeners.push_back(new WidgetInfo(events, widget));
}

void Interaction::addAnyTrackballListener(Events *events, Widget *widget)
{
    _anyTrackballListeners.push_back(new WidgetInfo(events, widget));
}

void Interaction::removeListener(Widget *widget)
{
    std::list<WidgetInfo *>::const_iterator iter;
    for (iter = _widgetInfoList.begin(); iter != _widgetInfoList.end(); iter++)
    {
        if ((*iter)->_widget == widget)
        {
            _widgetInfoList.remove(*iter);
            delete (*iter);
            return;
        }
    }
}

void Interaction::removeListener(PickBox *box)
{
    std::list<WidgetInfo *>::const_iterator iter;
    for (iter = _boxList.begin(); iter != _boxList.end(); iter++)
    {
        if ((*iter)->_box == box)
        {
            WidgetInfo *w = (*iter);
            _boxList.remove(*iter);
            delete w;
            return;
        }
    }
}

void Interaction::removeAnyButtonListener(Widget *widget)
{
    if (_anyButtonListeners.empty())
        return;

    std::list<WidgetInfo *>::const_iterator iter;
    for (iter = _anyButtonListeners.begin(); iter != _anyButtonListeners.end(); iter++)
    {
        if ((*iter)->_widget == widget)
        {
            _anyButtonListeners.remove(*iter);
            delete (*iter);
            return;
        }
    }
}

void Interaction::removeAnyTrackballListener(Widget *widget)
{
    if (_anyTrackballListeners.empty())
        return;

    std::list<WidgetInfo *>::const_iterator iter;
    for (iter = _anyButtonListeners.begin(); iter != _anyButtonListeners.end(); iter++)
    {
        if ((*iter)->_widget == widget)
        {
            _anyButtonListeners.remove(*iter);
            delete (*iter);
            return;
        }
    }
}

/** If NULL is returned, the geode has not registered. This is not a problem,
  but it should not happen. Check the traversal mask of the respective nodes.
  @return true if a registered object was found
*/
bool Interaction::findGeodeWidget(Geode *geode, WidgetInfo &widgetInfo)
{
    std::list<WidgetInfo *>::const_iterator iter;
    std::list<Geode *>::const_iterator iterGeode;
    for (iter = _widgetInfoList.begin(); iter != _widgetInfoList.end(); iter++)
    {
        for (iterGeode = (*iter)->_geodeList.begin(); iterGeode != (*iter)->_geodeList.end(); iterGeode++)
        {
            if ((*iterGeode) == geode)
            {
                widgetInfo._geodeList = (*iter)->_geodeList;
                widgetInfo._isectGeode = geode;
                widgetInfo._widget = (*iter)->_widget;
                widgetInfo._events = (*iter)->_events;
                widgetInfo._box = 0;
                return true;
            }
        }
    }
    widgetInfo.reset();
    return false;
}

/** Find first intersection of pointer with bounding box object.
  Due to the lack of a more sophisticated algorithm, the closest object
  to the wand is considered the one that is intersected and has the closest
  midpoint to the wand.
  @return isect
*/
void Interaction::getFirstBoxIntersection(Vec3 &wStart, Vec3 &wEnd, IsectInfo &isect)
{
    Matrix b2w; // box to world coordinates
    Matrix w2b; // world to box coordinates
    Vec3 bStart, bEnd; // laser pointer in box coordinate system
    Vec3 diff;
    Vec3 bCenter, wCenter;
    float dist;
    float minDist = FLT_MAX;

    ref_ptr<LineSegment> line = new LineSegment();

    // Intersect line with all bounding box widgets:
    isect.found = false;
    std::list<WidgetInfo *>::const_iterator iter;
    //if (_boxList.size()==0) cerr << "boxlist empty" << endl;
    for (iter = _boxList.begin(); iter != _boxList.end(); iter++)
    {
        //cerr << "checking box" << endl;
        // Transform pointer line into box coordinate system:
        b2w = (*iter)->_box->getB2W();
        w2b = Matrix::inverse(b2w);
        bStart = wStart * w2b;
        bEnd = wEnd * w2b;
        line->set(bStart, bEnd);

        // Intersection test:
        if (line->intersect(((*iter)->_box->_bbox)))
        {
            bCenter = (*iter)->_box->_bbox.center();
            diff = bStart - bCenter;
            dist = diff.length();
            if (dist < minDist) // is this box closer to the wand than the ones before?
            {
                isect.found = true;
                minDist = dist;
                isect.widget._events = (*iter)->_box;
                isect.widget._box = (*iter)->_box;
                wCenter = bCenter * b2w;
                isect.point = wCenter;
            }
        }
    }
}

/** Compares the first geode and the first bounding box object and returns
  the one that is closer to the wand.
  @return isect
*/
void Interaction::getFirstIntersection(Vec3 &wStart, Vec3 &wEnd, IsectInfo &isect)
{
    Vec3 diffGeode, diffBox;
    IsectInfo isectGeode;
    IsectInfo isectBox;

    isect.found = false;

    IsectType itype = getFirstGeodeIntersection(wStart, wEnd, isectGeode);
    if (itype == ISECT_OTHER)
        return;

    getFirstBoxIntersection(wStart, wEnd, isectBox);

    if (!isectGeode.found) // no geode intersects: box wins
    {
        if (isectBox.found)
        {
            isect = isectBox;
        }
    }
    else if (!isectBox.found) // no box: geode wins
    {
        isect = isectGeode;
    }
    else
    {
        // If geode is a child of box, return geode no matter what:
        if (CUI::isChild(isectGeode.widget._isectGeode, isectBox.widget._box->getNode()))
        {
            isect = isectGeode;
        }
        else // else return the one which is closer to the wand:
        {
            diffGeode = wStart - isectGeode.point;
            diffBox = wStart - isectBox.point;
            if (diffGeode.length() > diffBox.length())
            {
                isect = isectBox;
            }
            else
            {
                isect = isectGeode;
            }
        }
    }
}

/** Find first intersection of pointer with intersectable OSG object.
  @param wStart and wEnd: world coordinates
  @return isect
*/
Interaction::IsectType Interaction::getFirstGeodeIntersection(Vec3 &wStart, Vec3 &wEnd, IsectInfo &isect)
{
    // Compute intersections of viewing ray with pick objects:
    osgUtil::IntersectVisitor iv;
    osg::ref_ptr<osg::LineSegment> testSegment = new LineSegment;
    testSegment->set(wStart, wEnd);
    iv.addLineSegment(testSegment.get());
    iv.setTraversalMask(2);

    // Traverse the whole scenegraph.
    // Non-Interactive objects have been marked with setNodeMask(~2):
    _worldRoot->accept(iv);

    isect.found = false;
    if (iv.hits())
    {
        osgUtil::IntersectVisitor::HitList &hitList = iv.getHitList(testSegment.get());
        if (!hitList.empty())
        {
            isect.point = hitList.front().getWorldIntersectPoint();
            isect.normal = hitList.front().getWorldIntersectNormal();
            if (findGeodeWidget(hitList.front()._geode.get(), isect.widget))
            {
                isect.found = true;
                return ISECT_OSG;
            }
            else
            {
                return ISECT_OTHER;
            }
        }
    }
    return ISECT_NONE;
}

void Interaction::setGazeInteraction(bool gaze)
{
    _gazeInteraction = gaze;
}

bool Interaction::getGazeInteraction()
{
    return _gazeInteraction;
}

void Interaction::widgetDeleted(Node *node)
{
    _head->widgetDeleted(node);
    _wandR->widgetDeleted(node);
    _wandL->widgetDeleted(node);
    _mouse->widgetDeleted(node);
}

void Interaction::widgetDeleted()
{
    _head->widgetDeleted();
    _wandR->widgetDeleted();
    _wandL->widgetDeleted();
    _mouse->widgetDeleted();
}

LogFile *Interaction::getLogFile()
{
    return _logFile;
}

void Interaction::setLogFile(LogFile *lf)
{
    _logFile = lf;
}
