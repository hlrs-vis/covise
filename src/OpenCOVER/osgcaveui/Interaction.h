/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_INTERACTION_H_
#define _CUI_INTERACTION_H_

// Local:
#include "InputDevice.h"
#include "LogFile.h"

namespace cui
{

/** Base class of user interaction implementations.
    This class manages all input devices in the system.
  */
class CUIEXPORT Interaction
{
public:
    enum IsectType
    {
        ISECT_NONE,
        ISECT_OSG,
        ISECT_OTHER
    };

    InputDevice *_head;
    InputDevice *_wandR;
    InputDevice *_wandL;
    InputDevice *_mouse;

    std::list<WidgetInfo *> _anyButtonListeners;
    std::list<WidgetInfo *> _anyTrackballListeners;

    Interaction(osg::Group *, osg::Group *, LogFile * = NULL);
    virtual ~Interaction();
    virtual osg::Matrix getW2O();
    virtual void addListener(Events *, Widget *);
    virtual void addListener(PickBox *);
    virtual void addAnyButtonListener(Events *, Widget *);
    virtual void addAnyTrackballListener(Events *, Widget *);
    virtual void removeListener(Widget *);
    virtual void removeListener(PickBox *);
    virtual void removeAnyButtonListener(Widget *);
    virtual void removeAnyTrackballListener(Widget *);
    virtual bool action();
    virtual bool findGeodeWidget(osg::Geode *, WidgetInfo &);
    virtual IsectType getFirstGeodeIntersection(osg::Vec3 &, osg::Vec3 &, IsectInfo &);
    virtual void getFirstBoxIntersection(osg::Vec3 &, osg::Vec3 &, IsectInfo &);
    virtual void getFirstIntersection(osg::Vec3 &, osg::Vec3 &, IsectInfo &);
    virtual void setGazeInteraction(bool);
    virtual bool getGazeInteraction();
    virtual void widgetDeleted(osg::Node *);
    virtual void widgetDeleted();
    virtual LogFile *getLogFile();
    virtual void setLogFile(LogFile *);

protected:
    std::list<WidgetInfo *> _widgetInfoList; ///< list of all registered geode widgets
    std::list<WidgetInfo *> _boxList; ///< list of all registered PickBox widgets
    osg::Group *_worldRoot;
    osg::Group *_objectRoot;
    bool _gazeInteraction; ///< false = gaze directed interaction
    cui::LogFile *_logFile;
};
}
#endif
