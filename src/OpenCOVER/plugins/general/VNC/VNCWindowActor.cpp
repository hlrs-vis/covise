/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                          (C)2009 HLRS  **
 **                                                                        **
 ** Description: VNC Window Actor Class                                    **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 ** Created on: Nov 17, 2008                                               **
 **                                                                        **
 ** License: GPL v2 or later                                               **
 **                                                                        **
\****************************************************************************/

#include "VNCWindowActor.h"

#include "VNCWindow.h"
#include "VNCPlugin.h"
#include <cover/coVRPluginSupport.h>

#include <cstdlib>
#include <cassert>

#include <iostream>

VNCWindowActor::VNCWindowActor(VNCWindow *window)
    : window(window)
    , haveFocus(false)
{
    assert(window != NULL);
}

VNCWindowActor::~VNCWindowActor()
{
    if (haveFocus)
    {
        looseFocus();
    }
}

void VNCWindowActor::gainFocus()
{
    if (!haveFocus)
    {
        std::cerr << "VNCWindowActor::gainFocus() info: called" << std::endl;
        haveFocus = true;
        VNCPlugin::plugin->setNewWindowActor(this);
        cover->grabKeyboard(VNCPlugin::plugin);
    }
}

void VNCWindowActor::looseFocus()
{
    if (haveFocus)
    {
        std::cerr << "VNCWindowActor::looseFocus() info: called" << std::endl;
        cover->releaseKeyboard(VNCPlugin::plugin);
        haveFocus = false;
        VNCPlugin::plugin->setNewWindowActor(0);
    }
}

void VNCWindowActor::texturePointerLeft(vrui::coTextureRectBackground *)
{
    looseFocus();
} // texturePointerClicked

void VNCWindowActor::texturePointerClicked(vrui::coTextureRectBackground *, float x,
                                           float y)
{
    std::cerr << "VNCWindowActor::texturePointerClicked: " << x << ", " << y << std::endl;
    window->mouseButtonPressed(x, y);
} // texturePointerClicked

void VNCWindowActor::texturePointerReleased(vrui::coTextureRectBackground *, float x,
                                            float y)
{
    std::cerr << "VNCWindowActor::texturePointerReleased: " << x << ", " << y << std::endl;
    window->mouseButtonReleased(x, y);
} // texturePointerReleased

void VNCWindowActor::texturePointerDragged(vrui::coTextureRectBackground *, float x,
                                           float y)
{
    std::cerr << "VNCWindowActor::texturePointerDragged: " << x << ", " << y << std::endl;
    window->mouseDragged(x, y);
    gainFocus();
} // texturePointerDragged

void VNCWindowActor::texturePointerMoved(vrui::coTextureRectBackground *, float x,
                                         float y)
{
    std::cerr << "VNCWindowActor::texturePointerMoved: " << x << ", " << y << std::endl;
    window->mouseMoved(x, y);
    gainFocus();
} // texturePointerMoved

void VNCWindowActor::keyDown(int keysym, int mod)
{
    std::cerr << "VNCWindowActor::keyDown: " << keysym << ", " << mod << std::endl;
    window->keyPressed(keysym, mod);
}

void VNCWindowActor::keyUp(int keysym, int mod)
{
    std::cerr << "VNCWindowActor::keyUp: " << keysym << ", " << mod << std::endl;
    window->keyReleased(keysym, mod);
}
