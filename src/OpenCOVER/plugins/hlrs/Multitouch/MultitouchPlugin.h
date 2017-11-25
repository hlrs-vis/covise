/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Multitouch_PLUGIN_H
#define _Multitouch_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2012 HLRS  **
**                                                                          **
** Description: Multitouch Plugin											**
**                                                                          **
**                                                                          **
** Author:																	**
**         Jens Dehlke														**
**                                                                          **
** History:  								                                **
** Sep-12  v1.0	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;
#include <cover/coVRPlugin.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRConfig.h>
#include <osg/io_utils>
#include "MultitouchNavigation.h"
#include "export.h"

class TouchContact
{
public:
    TouchContact(float cx, float cy, int ident)
        : x(cx)
        , y(cy)
        , id(ident)
    {
    }
    ~TouchContact(){};
    float x;
    float y;
    int id;
};

class MultitouchNavigation;

class MULTITOUCHEXPORT MultitouchPlugin : public coVRPlugin
{
private:
    enum interactionMode
    {
        NONE = 0,
        TBD,
        MOUSE,
        ROTATEXY,
        MOVEXY,
        C_MOVEXY,
        WALKXY,
        SCALEXYZ,
        C_SCALEXYZ,
        ROTATEZ,
        MOVEZ,
        C_MOVEZ,
        WALKZ,
        FLY
    };
    interactionMode _mode, _prevMode;
    std::list<TouchContact> _contacts;
    MultitouchNavigation *_navigation;

    void preFrame();
    void determineInteractionMode();
    void reset()
    {
        _prevMode = _mode;
        _counter = _scale = _move = _rotate = _skippedFrames = 0;
        _initialDistance = 0.;
    };
    virtual void recognizeGesture();
    void addMouseContact();
    double angleBetween2DVectors(osg::Vec2 v1, osg::Vec2 v2);

    osg::Vec2f _prev2DVec1, _prev2DVec2;
    int _counter, _scale, _move, _rotate, _skippedFrames; // counters
    int _buttonState, _mouseID, _navMode;
    double _initialDistance;

    TouchContact getContactCenter();
    std::list<TouchContact> getContacts();

public:
    MultitouchPlugin();
    ~MultitouchPlugin();

    virtual void addContact(TouchContact &c);
    virtual void updateContact(TouchContact &c);
    virtual void removeContact(TouchContact &c);
};
#endif
