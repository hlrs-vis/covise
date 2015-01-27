/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _REMOTEDTACTOR_H
#define _REMOTEDTACTOR_H

#include <OpenVRUI/coTextureRectBackground.h>

using vrui::coTextureRectBackgroundActor;
using vrui::coTextureRectBackground;

class IRemoteDesktop;

class RemoteDTActor : public coTextureRectBackgroundActor
{
public:
    /**
       * the actor needs to be attached to a remote desktop class, which sends the events
       * to the remote server
       */
    RemoteDTActor(IRemoteDesktop *);
    virtual ~RemoteDTActor()
    {
    }
    virtual void texturePointerClicked(coTextureRectBackground *, float, float);
    virtual void texturePointerReleased(coTextureRectBackground *, float, float);
    virtual void texturePointerDragged(coTextureRectBackground *, float, float);
    virtual void texturePointerMoved(coTextureRectBackground *, float, float);
    virtual void texturePointerLeft(coTextureRectBackground *);

    /**
       * keysym needs to be a X-keysym
       */
    virtual void keyDown(int keysym, int mod);
    virtual void keyUp(int keysym, int mod);

private:
    IRemoteDesktop *desktop;
    bool haveFocus;
}; // class RemoteDTActor

#endif // _REMOTEDTACTOR_H
