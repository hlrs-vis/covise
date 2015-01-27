/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _Surface_PLUGIN_H
#define _Surface_PLUGIN_H
/****************************************************************************\ 
**                                                            (C)2012 HLRS  **
**                                                                          **
** Description: Surface Plugin (does Surface)								**
**                                                                          **
**                                                                          **
** Author:																	**
**			Jens Dehlke														**
**			U. Woessner														**
**                                                                          **
** History:																	**
** Sep-12  v2 (updated to work with Multitouch Plugin)						**
** xxx-08  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <cover/coVRPlugin.h>
#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <cover/coVRTouchTable.h>
#include "plugins/hlrs/Multitouch/MultitouchPlugin.h"

class SurfaceContact
{
public:
    SurfaceContact(float cx, float cy, float ang, float ar, uint64_t id, bool f, bool t, int ident)
        : CenterX(cx)
        , CenterY(cy)
        , Orientation(ang)
        , Area(ar)
        , Identity(id)
        , finger(f)
        , tag(t)
        , Id(ident)
    {
    }
    ~SurfaceContact(){};
    float CenterX;
    float CenterY;
    float Orientation;
    float Area;
    uint64_t Identity; // for Tags, 0 if Finger
    bool finger;
    bool tag;
    int Id; // Contact ID
};

class MotionEvent
{
public:
    MotionEvent(float av, float ce, float cr, float cs, float ctx,
                float cty, float dx, float dy, float ed, float ev, float mox,
                float moy, float rd, float sd, float vx, float vy)
        : AngularVelocity(av)
        , CumulativeExpansion(ce)
        , CumulativeRotation(cr)
        , CumulativeScale(cs)
        , CumulativeTranslationX(ctx)
        , CumulativeTranslationY(cty)
        , DeltaX(dx)
        , DeltaY(dy)
        , ExpansionDelta(ed)
        , ExpansionVelocity(ev)
        , ManipulationOriginX(mox)
        , ManipulationOriginY(moy)
        , RotationDelta(rd)
        , ScaleDelta(sd)
        , VelocityX(vx)
        , VelocityY(vy){};
    ~MotionEvent(){};
    float AngularVelocity;
    float CumulativeExpansion;
    float CumulativeRotation;
    float CumulativeScale;
    float CumulativeTranslationX;
    float CumulativeTranslationY;
    float DeltaX;
    float DeltaY;
    float ExpansionDelta;
    float ExpansionVelocity;
    float ManipulationOriginX;
    float ManipulationOriginY;
    float RotationDelta;
    float ScaleDelta;
    float VelocityX;
    float VelocityY;
};

class SurfacePlugin : public coVRPlugin, public coTUIListener, coVRTouchTableInterface
{
private:
    std::list<SurfaceContact> _otherContacts, _fingerContacts;
    MultitouchPlugin *multitouchPlugin;

public:
    SurfacePlugin();
    ~SurfacePlugin();

    virtual void manipulation(MotionEvent &me);
    //! Notify that a finger has just been made active.
    virtual void addedContact(SurfaceContact &c);
    //! Notify that a finger has been updated.
    virtual void changedContact(SurfaceContact &c);
    //! A finger is no longer active.
    virtual void removedContact(SurfaceContact &c);

    void key(int type, int keySym, int mod);
    virtual bool isPlanar()
    {
        return true;
    };
    virtual void tabletEvent(coTUIElement *tUIItem);
    virtual void tabletPressEvent(coTUIElement *tUIItem);

    // coVRTouchTableInterface
    virtual int getMarker(std::string name);
    virtual bool isVisible(int);
    virtual osg::Vec2 getPosition(int); // Ursprung ist links unten X rechts X nach oben
    virtual float getOrientation(int);
};
#endif
