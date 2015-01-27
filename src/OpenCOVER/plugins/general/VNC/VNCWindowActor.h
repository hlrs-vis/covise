/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VNCWINDOWACTOR_H_
#define VNCWINDOWACTOR_H_

#include <OpenVRUI/coTextureRectBackground.h>

class VNCWindow;

class VNCWindowActor : public vrui::coTextureRectBackgroundActor
{
public:
    /**
    * the actor needs to be attached to a remote desktop class, which sends the events
    * to the remote server
    */
    VNCWindowActor(VNCWindow *);
    virtual ~VNCWindowActor();
    virtual void texturePointerClicked(vrui::coTextureRectBackground *, float, float);
    virtual void texturePointerReleased(vrui::coTextureRectBackground *, float, float);
    virtual void texturePointerDragged(vrui::coTextureRectBackground *, float, float);
    virtual void texturePointerMoved(vrui::coTextureRectBackground *, float, float);
    virtual void texturePointerLeft(vrui::coTextureRectBackground *);

    /**
    * keysym needs to be a X-keysym
    */
    virtual void keyDown(int keysym, int mod);
    virtual void keyUp(int keysym, int mod);

    void gainFocus();
    void looseFocus();

    VNCWindow *getWindow()
    {
        return window;
    }

private:
    VNCWindow *window;
    bool haveFocus;
}; // class VNCWindowActor

#endif /* VNCWINDOWACTOR_H_ */
