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
 *	File			coMousePointer.h (Performer 2.0)	*
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

#ifndef MOUSE_POINTER_H
#define MOUSE_POINTER_H

#include <util/common.h>
#include <osg/Matrix>

namespace opencover
{

class ButtonDevice;

struct MouseEvent
{
    int type, state, code;
};

class COVEREXPORT coMousePointer
{
    friend class Input;

private:
    coMousePointer();
    ~coMousePointer();

    osg::Matrix matrix;

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

    ButtonDevice *buttons;

public:
    // frame time of last mouse event
    double eventTime() const;

    // process mouse events
    void handleEvent(int type, int state, int code, bool queue = true);

    void update();
    void setMatrix(const osg::Matrix &mat);
    const osg::Matrix &getMatrix() const;

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
#endif
