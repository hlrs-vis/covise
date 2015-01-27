/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef VNCPLUGIN_H_
#define VNCPLUGIN_H_

#ifdef WIN32

#if (_MSC_VER > 1310)
/* && !(defined(MIDL_PASS) || defined(RC_INVOKED)) */
#define POINTER_64 __ptr64
#endif

#include <winsock2.h>
#endif

#include <list>

#include <cover/coVRPluginSupport.h>
#include <OpenVRUI/coMenuItem.h>
#include <OpenVRUI/coMenu.h>
#include <OpenVRUI/coTexturedBackground.h>
#include <cover/coVRTui.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coPopupHandle;
}

namespace opencover
{
class coVRLabel;
class coTUITab;
class coTUIEditIntField;
class coTUILabel;
class coTUIEditField;
class coTUIToggleButton;
class coTUIButton;
class coTUIElement;
class coTUIListBox;
}

class VNCWindow;
class VNCWindowActor;
class VNCClient;
class VNCConnection;

using namespace vrui;
using namespace opencover;

class VNCPlugin : public coVRPlugin, public coMenuListener, public coTUIListener
{
public:
    static VNCPlugin *plugin;

    VNCPlugin();
    virtual ~VNCPlugin();
    bool init();

    void initUI();
    void tabletEvent(coTUIElement *);
    void tabletPressEvent(coTUIElement *);
    void tabletReleaseEvent(coTUIElement *);

    /// server may be a hostname or an IP-address
    void connectToServer(const char *server, unsigned port, const char *password);

    /// disconnect from selected server (tui)
    void terminateSelectedConnection();

    /// disconnect from specific server
    void terminateConnection(const char *hostname, unsigned int port);

    /// Show specific session
    void show(const char *hostname, unsigned int port);

    /// Hide specific session
    void hide(const char *hostname, unsigned int port);

    /// Set visibility of specific session
    void setVisible(bool on, const char *hostname, unsigned int port);

    void preFrame();
    virtual void menuEvent(coMenuItem *menuItem);
    void drawInit();
    void message(int type, int len, const void *buf);

    /**
    * this is here until there are other means to get events
    * it is called by coVRKey()
    *
    * type is either 6 (key down) or 7 (key up)
    * keysym seems to be a X keysym
    */
    void key(int type, int keysym, int mod);

    void setNewWindowActor(VNCWindowActor *actor);

private:
    std::list<VNCConnection *> connections;

    coSubMenuItem *pinboardEntry;
    coRowMenu *VNCPluginMenu;
    coCheckboxMenuItem *desktopEntry;

    coTUITab *tuiVNCPluginTab;

    coTUIFrame *tuiNewConnectionFrame;
    coTUIEditIntField *tuiPortEdit;
    coTUILabel *tuiIPaddrLabel;
    coTUIEditField *tuiIPaddr;
    coTUILabel *tuiPasswdLabel;
    coTUIEditField *tuiPasswd;
    coTUIButton *tuiConnectButton;

    coTUIFrame *tuiOpenConnectionsFrame;
    coTUIListBox *tuiConnectionList;
    coTUIButton *tuiDisconnectButton;

    coTUIFrame *tuiConnectionOperationFrame;
    coTUILabel *tuiCommandlineLabel;
    coTUIEditField *tuiCommandline;
    coTUIButton *tuiCommandlineSend;
    coTUIButton *tuiCommandlineSendAndReturn;
    coTUIButton *tuiCommandlineReturn;
    coTUIButton *tuiCommandlineCTRL_C;
    coTUIButton *tuiCommandlineCTRL_D;
    coTUIButton *tuiCommandlineCTRL_Z;
    coTUIButton *tuiCommandlineESCAPE;

    VNCWindowActor *focusedActor;
};

#endif /* VNCPLUGIN_H_ */
