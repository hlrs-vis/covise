/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Plugin.h

#pragma once

#include "TouchInteraction.h"

#include <cover/coVRPlugin.h>

//
// Touch interaction plugin
//
//
// The plugin handles touch messages, filtering out the touch events interacting
// with the OpenCOVER menu. Other events are forwarded to TouchHandler's which can
// take appropriate actions.
//
// Multiple handlers are managed by the plugin. E.g. there might be one handler which
// should be active if only one touch point is recognized, another handling two touch points,
// and yet another which will be active in all other cases.
//
// See Plugin.cpp on how to add new handlers.
//
//
// The config format [COVER.Plugin.TouchInteraction]
//
// handler0:    The handler whcih will be active if no other handler is active (Optional)
// handler1:    The handler whcih will be active if only a single touch point is recognized
// handler2:    The handler whcih will be active if two touch points are recognized
// handlerN:    The handler whcih will be active if N touch points are recognized
//
// Here N is a number specified by the value of
//
// maxHandlers: The maximum number of touch handlers.
//              This is optional (Currently the default value is 10)
//
// The plugin reads its config information from
//
// A typical tag might look like this
//
// <COVER>
//   <Plugin>
//     <TouchInteraction value="on" handler1="Utouch3D" handler2="ScreenSpace" />
//   </Plugin>
// </COVER>
//

class Plugin
    : public opencover::coVRPlugin,
      public TouchInteraction
{
public:
    Plugin();

    ~Plugin() override;

    //
    // from coVRPlugin
    //

    // this function is called when COVER is up and running and the plugin is initialized
    virtual bool init() override;

    // reimplement to do early cleanup work and return false to prevent unloading
    virtual bool destroy() override;

    // this function is called from the main thread before rendering a frame
    virtual void preFrame() override;

    // this function is called if a message arrives
    virtual void message(int toWhom, int type, int length, void const *data) override;

protected:
    //
    // from TouchInteraction
    //

    void onTouchesBeingPressed(Touches const &touches) override;

    //
    // Called when a new touch is recognized
    // Returns whether the message has been processed
    //
    bool onTouchPressed(Touches const &touches, Touch const &reason) override;

    //
    // Called when a touch has moved
    // Returns whether the message has been processed
    //
    bool onTouchesMoved(Touches const &touches) override;

    void onTouchesBeingReleased(Touches const &touches) override;

    //
    // Called when a touch is released
    // Returns whether the message has been processed
    //
    // NOTE: id is actually invalid and touches might be empty!
    //
    bool onTouchReleased(Touches const &touches, Touch const &reason) override;

    //
    // Called when the touch interaction has been canceled
    // Returns whether the message has been processed
    //
    bool onTouchesCanceled() override;

private:
    void sendPressEvents(Touches const &touches);

    void sendReleaseEvents(Touches const &touches);
};
