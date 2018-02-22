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
*	File			coMousePointer.cpp (Performer 2.0)	*
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

#include <math.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <OpenVRUI/sginterface/vruiButtons.h>
#include <cover/VRViewer.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>

#include "coMousePointer.h"
#include "buttondevice.h"
#include "input.h"

using namespace opencover;

/*______________________________________________________________________*/
coMousePointer::coMousePointer()
{
    if (cover->debugLevel(2))
        fprintf(stderr, "new coMousePointer\n");

    buttons = Input::instance()->getButtons("Mouse");

    wheelCounter[0] = wheelCounter[1] = newWheelCounter[0] = newWheelCounter[1] = 0;

    matrix = osg::Matrix::identity();

    mouseTime = cover->frameRealTime();
    mouseButtonTime = mouseTime - 1.0;
    mouseX = mouseY = 0;

    width = 1;
    height = 1;
    screenX = 0.;
    screenY = 0.;
    screenZ = 0.;
    screenH = 0.;
    screenP = 0.;
    screenR = 0.;

    if (coVRConfig::instance()->numWindows() > 0 && coVRConfig::instance()->numScreens() > 0)
    {
        width = coVRConfig::instance()->screens[0].hsize;
        height = coVRConfig::instance()->screens[0].vsize;
        if ((coVRConfig::instance()->viewports[0].viewportXMax - coVRConfig::instance()->viewports[0].viewportXMin) == 0)
        {
            xres = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sx;
            yres = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sy;
        }
        else
        {
            xres = (int)(coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sx * (coVRConfig::instance()->viewports[0].viewportXMax - coVRConfig::instance()->viewports[0].viewportXMin));
            yres = (int)(coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sy * (coVRConfig::instance()->viewports[0].viewportYMax - coVRConfig::instance()->viewports[0].viewportYMin));
        }

        //xres=coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sx;
        //yres=coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sy;
        xori = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].ox;
        yori = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].oy;

        //fprintf(stderr,"width: %f, height %f, xres %d , yres %d, xori %d, yori %d\n", width, height, xres,yres,xori,yori);

        screenX = coVRConfig::instance()->screens[0].xyz[0];
        screenY = coVRConfig::instance()->screens[0].xyz[1];
        screenZ = coVRConfig::instance()->screens[0].xyz[2];
        screenH = coVRConfig::instance()->screens[0].hpr[0];
        screenP = coVRConfig::instance()->screens[0].hpr[1];
        screenR = coVRConfig::instance()->screens[0].hpr[2];
    }
}

/*______________________________________________________________________*/
coMousePointer::~coMousePointer()
{
}

double coMousePointer::eventTime() const
{
    return mouseTime;
}

void coMousePointer::queueEvent(int type, int state, int code)
{
    MouseEvent me = { type, state, code };
    std::cerr << "queueEvent " << type << " " << state << " " << code << std::endl;
    eventQueue.push_back(me);
}

void coMousePointer::processEvents()
{
    while (!eventQueue.empty())
    {
        MouseEvent me = eventQueue.front();
        eventQueue.pop_front();
        handleEvent(me.type, me.state, me.code, false);
        if (me.type == osgGA::GUIEventAdapter::PUSH
            || me.type == osgGA::GUIEventAdapter::RELEASE
            || me.type == osgGA::GUIEventAdapter::SCROLL)
            break;
    }
}

void coMousePointer::handleEvent(int type, int state, int code, bool queue)
{
    mouseTime = cover->frameRealTime();

    if (queue && !eventQueue.empty())
    {
        queueEvent(type, state, code);
        return;
    }

    switch(type)
    {
    case osgGA::GUIEventAdapter::DRAG:
        mouseX = state;
        mouseY = code;
        break;
    case osgGA::GUIEventAdapter::MOVE:
        mouseX = state;
        mouseY = code;
        break;
    case osgGA::GUIEventAdapter::SCROLL:
        if (!buttonPressed)
        {
            if (state == osgGA::GUIEventAdapter::SCROLL_UP)
                ++newWheelCounter[0];
            else if (state == osgGA::GUIEventAdapter::SCROLL_DOWN)
                --newWheelCounter[0];
            else if (state == osgGA::GUIEventAdapter::SCROLL_RIGHT)
                ++newWheelCounter[1];
            else if (state == osgGA::GUIEventAdapter::SCROLL_LEFT)
                --newWheelCounter[1];
        }
        break;
    case osgGA::GUIEventAdapter::PUSH:
        buttonPressed = bool(state);
        if (mouseTime == mouseButtonTime)
            queueEvent(type, state, code);
        else
        {
            buttons->setButtonState(state, true);
            mouseButtonTime = cover->frameRealTime();
        }
        break;
    case osgGA::GUIEventAdapter::RELEASE:
        if (mouseTime == mouseButtonTime)
            queueEvent(type, state, code);
        else
        {
            buttons->setButtonState(state, true);
            mouseButtonTime = cover->frameRealTime();
        }
        buttonPressed = bool(state);
        break;
    case osgGA::GUIEventAdapter::DOUBLECLICK:
        handleEvent(osgGA::GUIEventAdapter::PUSH, state, code, queue);
        handleEvent(osgGA::GUIEventAdapter::RELEASE, state, code, true);
        break;
    }
}

/*______________________________________________________________________*/
void
coMousePointer::update()
{
    if (cover->debugLevel(5))
        fprintf(stderr, "coMousePointer::update\n");

    if (coVRConfig::instance()->numWindows() <= 0 || coVRConfig::instance()->numScreens() <= 0)
        return;

    unsigned state = buttons->getButtonState();
    state &= ~(vrui::vruiButtons::WHEEL);
    buttons->setButtonState(state);

    processEvents();

    wheelCounter[0] = newWheelCounter[0];
    wheelCounter[1] = newWheelCounter[1];
    newWheelCounter[0] = newWheelCounter[1] = 0;

    static int oldWidth = -1, oldHeight = -1;
    int currentW, currentH;
    const osg::GraphicsContext::Traits *traits = NULL;
    if (coVRConfig::instance()->windows[0].window)
        traits = coVRConfig::instance()->windows[0].window->getTraits();
    if (!traits)
        return;

    currentW = traits->width;
    currentH = traits->height;
    if (oldWidth != currentW || oldHeight != currentH)
    {
        coVRConfig::instance()->windows[0].sx = currentW;
        coVRConfig::instance()->windows[0].sy = currentH;
        oldWidth = currentW;
        oldHeight = currentH;
        if ((coVRConfig::instance()->viewports[0].viewportXMax - coVRConfig::instance()->viewports[0].viewportXMin) == 0)
        {
            xres = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sx;
            yres = coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sy;
        }
        else
        {
            xres = (int)(coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sx * (coVRConfig::instance()->viewports[0].viewportXMax - coVRConfig::instance()->viewports[0].viewportXMin));
            yres = (int)(coVRConfig::instance()->windows[coVRConfig::instance()->viewports[0].window].sy * (coVRConfig::instance()->viewports[0].viewportYMax - coVRConfig::instance()->viewports[0].viewportYMin));
        }
    }

    float mx = x();
    float my = y();
    float wc = 0.0;
    // mouse coordinates are relative to original window size, even if window has been resized... so don´t use current window size here.
    // but physical size might have been adjusted, if aspect ratio changed
    width = coVRConfig::instance()->screens[0].hsize;
    height = coVRConfig::instance()->screens[0].vsize;
    if(coVRConfig::instance()->channels[0].stereoMode == osg::DisplaySettings::HORIZONTAL_SPLIT)
    {
        mx *=2;
        if(mx > xres)
            mx -= xres;
    }
    if(coVRConfig::instance()->channels[0].stereoMode == osg::DisplaySettings::VERTICAL_SPLIT)
    {
        my *=2;
        if(my > yres)
            my -= yres;
    }

    osg::Vec3 mouse2D;
    osg::Vec3 mouse3D;
    osg::Matrix transMat;

    mouse2D[0] = ((mx - (xres / 2.0)) / xres) * width;
    if(mouse2D[0] > width/2.0) // work around for twho viewports in one window
    {
        mouse2D[0] -= width;
    }
    mouse2D[1] = 0;
    mouse2D[2] = (my / yres - (0.5)) * height;
    if(mouse2D[2] > height/2.0)// work around for twho viewports in one window
    {
        mouse2D[2] -= height;
    }

    MAKE_EULER_MAT(transMat, screenH, screenP, screenR);
    mouse3D = transMat.preMult(mouse2D);
    transMat.makeTranslate(screenX, screenY, screenZ);
    mouse3D = transMat.preMult(mouse3D);
    transMat.makeRotate(coVRConfig::instance()->worldAngle(), osg::X_AXIS);
    mouse3D = transMat.preMult(mouse3D);
    //cerr << mx << " , " << my << endl;
    //cerr << " 3D:" << mouse3D[0] << " , " << mouse3D[1] << " , " << mouse3D[2] << " , "<< endl;
    //cerr << " 2D:" << mouse2D[0] << " , " << mouse2D[1] << " , " << mouse2D[2] << " , "<< endl;

    osg::Vec3 YPos(0, 1, 0);
    osg::Vec3 direction;
    osg::Vec3 viewerPos = VRViewer::instance()->getViewerPos();

    // if orthographic projection: project viewerPos onto screenplane-normal passing through mouse3D
    if (coVRConfig::instance()->orthographic())
    {
        osg::Vec3 normal;
        MAKE_EULER_MAT(transMat, screenH, screenP, screenR);
        normal = transMat.preMult(osg::Vec3(0.0, 1.0, 0.0));
        transMat.makeRotate(coVRConfig::instance()->worldAngle(), osg::X_AXIS);
        normal = transMat.preMult(normal);
        normal.normalize();
        osg::Vec3 projectionOntoNormal = normal * (normal * (viewerPos - mouse3D));
        viewerPos = mouse3D + projectionOntoNormal;
    }

    direction = mouse3D - viewerPos;
    direction.normalize();
    matrix.makeRotate(YPos, direction);
    matrix.rotate(osg::inDegrees(90.0), osg::Vec3(0.0, 1.0, 0.0));
    osg::Matrix tmp;
    tmp.makeTranslate(direction[0] * wc, direction[1] * wc, direction[2] * wc);
    matrix.postMult(tmp);

    tmp.makeTranslate(viewerPos[0], viewerPos[1], viewerPos[2]);
    matrix.postMult(tmp);

    //cerr << " VP:" << viewerPos[0] << " , "<< viewerPos[1] << " , "<< viewerPos[2] << " , "<< endl;
    //mouse3D /=10; // umrechnung in cm (scheisse!!!!!!)
    //matrix.makeTrans(mouse3D[0],mouse3D[1],mouse3D[2]);
}

float coMousePointer::x() const
{
    return mouseX;
}

float coMousePointer::y() const
{
    return mouseY;
}

float coMousePointer::winWidth() const
{
    return xres;
}

float coMousePointer::winHeight() const
{
    return yres;
}

float coMousePointer::screenWidth() const
{
    return width;
}

float coMousePointer::screenHeight() const
{
    return height;
}

const osg::Matrix &coMousePointer::getMatrix() const
{
    return matrix;
}

void coMousePointer::setMatrix(const osg::Matrix &mat)
{

    matrix = mat;
}

int coMousePointer::wheel(size_t num) const
{

    if (num >= 2)
        return 0;

    return wheelCounter[num];
}

unsigned int coMousePointer::buttonState() const
{

    return buttons->getButtonState();
}
