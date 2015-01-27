/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CFDGUI_PLUGIN_H
#define _CFDGUI_PLUGIN_H

#include <cover/coVRPlugin.h>

#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coMenuItem.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coButtonMenuItem;
class coIconButtonToolboxItem;
}

class CfdGuiPlugin : public opencover::coVRPlugin, public vrui::coMenuListener
{
public:
    // constructor
    CfdGuiPlugin();

    // destructor: deletes all items in the list
    virtual ~CfdGuiPlugin();

    virtual bool init();
    virtual void guiToRenderMsg(const char *msg);
    virtual void preFrame();
    virtual void key(int type, int keySym, int mod);

    vrui::coMenuItem *getMenuButton(const std::string &buttonName);

    void sendPresentationForwardMsgToGui();
    void sendPresentationReloadMsgToGui();
    void sendPresentationBackwardMsgToGui();
    void sendPresentationPlayMsgToGui();
    void sendPresentationStopMsgToGui();
    void sendPresentationToEndMsgToGui();
    void sendPresentationToStartMsgToGui();

    void handleTransformCaseMsg(const char *objectName, float *row0, float *row1, float *row2, float *row3);

protected:
    vrui::coSubMenuItem *coverMenuButton_;
    vrui::coRowMenu *presentationMenu_;
    vrui::coButtonMenuItem *forwardButton_;
    vrui::coButtonMenuItem *reloadButton_;
    vrui::coButtonMenuItem *toEndButton_;
    vrui::coButtonMenuItem *backwardButton_;
    vrui::coButtonMenuItem *toStartButton_;
    vrui::coButtonMenuItem *playButton_;
    vrui::coButtonMenuItem *stopButton_;

    bool hideMenu;

private:
    void menuEvent(vrui::coMenuItem *);
};

#endif
