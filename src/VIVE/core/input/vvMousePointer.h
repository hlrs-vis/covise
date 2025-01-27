/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *									*
 *          								*
 *                            (C) 2001					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			vvMousePointer.h (Performer 2.0)	*
 *									*
 *	Description		Mouse support for COVER
 *									*
 *	Author		Uwe Woessner				*
 *									*
 *	Date			19.09.2001				*
 *									*
 *	Status			none					*
 *									*
 ************************************************************************/

#pragma once

#include <util/common.h>
#include <vsg/maths/mat4.h>

namespace vive
{

class ButtonDevice;
class TrackingBody;

struct MouseEvent
{
    int type, state, code;
};

class VVCORE_EXPORT vvMousePointer
{
    friend class Input;

private:
    vvMousePointer();
    ~vvMousePointer();

    float width, height;
    int xres, yres, xori, yori;
    float screenX, screenY, screenZ;
    float screenH, screenP, screenR;

    int wheelCounter[2], newWheelCounter[2]; // vertical and horizontal
    float mouseX, mouseY;
    typedef std::deque<MouseEvent> EventQueue;
    EventQueue eventQueue;
    double mouseTime, mouseButtonTime;
    void queueEvent(int type, int state, int code);
    void processEvents();

    ButtonDevice *buttons = nullptr;
    TrackingBody *body = nullptr;
    bool buttonPressed = false;

public:
    // frame time of last mouse event
    double eventTime() const;

    // process mouse events
    void handleEvent(int type, int state, int code, bool queue = true);

    void update();
#if 0
    void setMatrix(const vsg::dmat4 &mat);
#endif
    const vsg::dmat4 &getMatrix() const;

    //! current mouse screen x coordinate
    float x() const;

    //! current mouse screen y coordinate
    float y() const;

    //! width of window where mouse events are tracked
    float winWidth() const;

    //! height of window where mouse events are tracked
    float winHeight() const;

    //! width of screen where mouse events are tracked
    float screenWidth() const;

    //! height of screen where mouse events are tracked
    float screenHeight() const;

    int wheel(size_t num = 0) const;

    unsigned int buttonState() const;
};
}
