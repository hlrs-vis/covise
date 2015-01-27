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
#include <OpenVRUI/osg/mathUtils.h>
#define CO_MOUSE_LEFT_DOWN 1
#define CO_MOUSE_MIDDLE_DOWN 2
#define CO_MOUSE_RIGHT_DOWN 4
namespace opencover
{

class ButtonDevice;

struct MouseEvent
{
    int type, state, code;
};

class INPUT_LEGACY_EXPORT coMousePointer
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

    int wheel(size_t num = 0) const;

    unsigned int buttonState() const;
};
}
#endif
