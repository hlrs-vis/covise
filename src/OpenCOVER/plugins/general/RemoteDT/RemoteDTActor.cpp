/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                          (C)2007 HLRS  **
 **                                                                        **
 ** Description: Remote Desktop Event Handler class                        **
 **                                                                        **
 **                                                                        **
 ** Author: Lukas Pinkowski                                                **
 **                                                                        **
 ** History:                                                               **
 ** Oct-10 2007  v1                                                        **
 **                                                                        **
 ** License: GPL v2 or later                                               **
 **                                                                        **
\****************************************************************************/

#include "RemoteDTActor.h"
#include "RemoteDT.h"
#include <cover/coVRPluginSupport.h>
#include "Interface.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

RemoteDTActor::RemoteDTActor(IRemoteDesktop *desktop)
{
    assert(desktop != NULL);
    this->desktop = desktop;
    haveFocus = false;
}

void RemoteDTActor::texturePointerLeft(coTextureRectBackground *)
{
    cover->releaseKeyboard(RemoteDT::plugin);
    haveFocus = false;
} // texturePointerClicked

void RemoteDTActor::texturePointerClicked(coTextureRectBackground *, float x, float y)
{
    if (desktop->isConnected())
    {
        desktop->mouseButtonPressed(x, y);
    }
} // texturePointerClicked

void RemoteDTActor::texturePointerReleased(coTextureRectBackground *, float x, float y)
{
    if (desktop->isConnected())
    {
        desktop->mouseButtonReleased(x, y);
    }
} // texturePointerReleased

void RemoteDTActor::texturePointerDragged(coTextureRectBackground *, float x, float y)
{
    if (desktop->isConnected())
    {
        desktop->mouseDragged(x, y);
    }
    if (!haveFocus)
    {
        cover->grabKeyboard(RemoteDT::plugin);
        haveFocus = true;
    }
} // texturePointerDragged

void RemoteDTActor::texturePointerMoved(coTextureRectBackground *, float x, float y)
{
    if (desktop->isConnected())
    {
        desktop->mouseMoved(x, y);
    }
    if (!haveFocus)
    {
        cover->grabKeyboard(RemoteDT::plugin);
        haveFocus = true;
    }
} // texturePointerMoved

void RemoteDTActor::keyDown(int keysym, int mod)
{
    if (desktop->isConnected())
    {
        desktop->keyPressed(keysym, mod);
    }
}

void RemoteDTActor::keyUp(int keysym, int mod)
{
    if (desktop->isConnected())
    {
        desktop->keyReleased(keysym, mod);
    }
}
