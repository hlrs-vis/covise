/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CUI_EVENTS_H_
#define _CUI_EVENTS_H_

// OSG:
#include "Widget.h"
#include <osg/Vec3>

namespace cui
{

class InputDevice;

/** Derive from this class to listen to input device events. 
  Currently the cursor is at the intersection of a ray from the wand
  to the first intersection with listening widgets.
*/
class CUIEXPORT Events
{
public:
    virtual ~Events()
    {
    }

    /** Called when cursor enters the widget.
      @param event event class
    */
    virtual void cursorEnter(InputDevice *) = 0;

    /** Called once for each frame while cursor is in widget.
      @param event event class
    */
    virtual void cursorUpdate(InputDevice *) = 0;

    /** Called when cursor leaves the widget.
      @param event event class
    */
    virtual void cursorLeave(InputDevice *) = 0;

    /** Called on changes of button status.
      @param event event class
      @param button ID of button that changed its value
    */
    virtual void buttonEvent(InputDevice *, int) = 0;

    /** Called on joystick/trackball events.
      @param event event class
    */
    virtual void joystickEvent(InputDevice *) = 0;

    /** Called on mouse wheel events.
      @param event event class
      @param direction 1=wheel up, -1=wheel down
    */
    virtual void wheelEvent(InputDevice *, int) = 0;
};
}

#endif
