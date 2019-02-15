/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _REMOTEDT_H
#define _REMOTEDT_H
#ifdef WIN32

#if (_MSC_VER > 1310)
/* && !(defined(MIDL_PASS) || defined(RC_INVOKED)) */
#define POINTER_64 __ptr64
#endif

#include <winsock2.h>
#endif

#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>

#include <OpenVRUI/coMenuItem.h>
#include <cover/coTabletUI.h>

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coPotiMenuItem;
class coVRLabel;
class coPopupHandle;
class pfHighlight;
class coTrackerButtonInteraction;
}
namespace opencover
{
class coTUITab;
class coTUIEditIntField;
class coTUILabel;
class coTUIEditField;
class coTUIToggleButton;
class coTUIButton;
class coTUIElement;
}

class IRemoteDesktop;
class RemoteDTActor;

using namespace vrui;
using namespace opencover;

class RemoteDT : public coVRPlugin, public coMenuListener, public coTUIListener
{
public:
    static RemoteDT *plugin;

    RemoteDT();
    virtual ~RemoteDT();
    bool init();

    void initUI();
    void tabletEvent(coTUIElement *);
    void tabletPressEvent(coTUIElement *);
    void tabletReleaseEvent(coTUIElement *);

    /// server may be a hostname or an IP-address
    void connectToServer(const char *server, unsigned port, const char *password);

    /// disconnect from current server (if connected to any)
    void terminateConnection();

    /// Show the desktop
    void show();

    /// Hide the desktop
    void hide();

    /// Sets the desktop visibility
    void setVisible(bool on);

    void preFrame();
    virtual void menuEvent(coMenuItem *menuItem);
    void drawInit();
    void message(int toWhom, int type, int len, const void *buf);

    /**
       * this is here until there are other means to get events
       * it is called by coVRKey()
       *
       * type is either 6 (key down) or 7 (key up)
       * keysym seems to be a X keysym
       */
    void keyEvent(int type, int keysym, int mod);

private:
    IRemoteDesktop *desktop;
    RemoteDTActor *eventHandler;

    coSubMenuItem *pinboardEntry;
    coRowMenu *remoteDTMenu;
    coCheckboxMenuItem *desktopEntry;

    coTUITab *tuiRemoteDTTab;
    coTUIEditIntField *tuiPortEdit;
    coTUILabel *tuiIPaddrLabel;
    coTUIEditField *tuiIPaddr;
    coTUILabel *tuiPasswdLabel;
    coTUIEditField *tuiPasswd;

    coTUIToggleButton *tuiToggleVNC;
    coTUIToggleButton *tuiToggleRDP;
    coTUIButton *tuiConnectButton;
    coTUIButton *tuiDisconnectButton;

    coTUILabel *tuiCommandlineLabel;
    coTUIEditField *tuiCommandline;
    coTUIButton *tuiCommandlineSend;
    coTUIButton *tuiCommandlineSendAndReturn;
    coTUIButton *tuiCommandlineReturn;
    coTUIButton *tuiCommandlineCTRL_C;
    coTUIButton *tuiCommandlineCTRL_D;
    coTUIButton *tuiCommandlineCTRL_Z;
    coTUIButton *tuiCommandlineESCAPE;

    enum Protocol
    {
        PROTO_VNC,
        PROTO_RDP
    };
    Protocol protocol;
};
#endif
