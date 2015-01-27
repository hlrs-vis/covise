/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VR_UI_TOOLBAR_H
#define _VR_UI_TOOLBAR_H

/****************************************************************************\
 **                                                     (C)2003 VirCinity  **
 **                                                                        **
 ** Description: Plugin for Toolbar-based User-Interface                   **
 **                                                                        **
 **                                                                        **
 ** Author: Andreas Werner                                                 **
 **                                                                        **
 ** History:  			   	                                   **
 **       11.08.2003 we  Initial Version                                   **
 **                                                                        **
\****************************************************************************/

#include <OpenVRUI/coMenuItem.h>
#include <util/common.h>
#ifndef _WIN32
#include <sys/time.h>
#endif

#include <osg/Depth>
#include <osg/Stencil>
#include <cover/coVRPlugin.h>

namespace vrui
{
class coToolboxMenu;
class coIconButtonToolboxItem;
class coIconSubMenuToolboxItem;
class coLabelSubMenuToolboxItem;
class coSliderToolboxItem;
class coMenuItem;
class coIconToggleButtonToolboxItem;

class coLabelSubMenuToolboxItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coRowMenu;
class coLabelButtonToolboxItem;
class coSubMenuItem;
class coPotiMenuItem;
class coPotiToolboxItem;
}

namespace opencover
{
class coVRPlugin;
}

class AkSubMenu;

using namespace vrui;
using namespace opencover;

class AKToolbar : public coVRPlugin, public coMenuFocusListener, public coMenuListener
{
public:
    static AKToolbar *plugin;

    AKToolbar();
    virtual ~AKToolbar();
    bool init();
    coToolboxMenu *akToolbar_;

    // Status button
    coIconButtonToolboxItem *stateButton_;
    bool stateBusy_;

    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // ++ Virual Functions from coMenuFocusListener
    // +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    /// react on menu focus events
    virtual void focusEvent(bool focus, coMenu *menu);

    /// react on menu events
    virtual void menuEvent(coMenuItem *menuItem);

    /// Shortcuts
    struct Shortcut
    {
        coMenuItem *button;
        coMenuItem *observed;
        string plugin;
        string command;
        Shortcut()
            : button(NULL)
            , observed(NULL)
        {
        }
    };
    typedef std::map<coMenuItem *, Shortcut> ShortcutMap;
    ShortcutMap shortcutMap_;
    typedef std::vector<Shortcut *> ShortcutList;
    ShortcutList shortcutList_;

    /// add a certain shortcut
    void addShortcut(const char *name, string command, bool toggleButton = false, const char *plugin = NULL);

    /// check plugin's DebugLevel
    static bool debugLevel(int level)
    {
        return level <= debugLevel_;
    }

    /// receive messages from modules
    void message(int type, int len, const void *buf);

    void preFrame();

    void updatePlugins();
    void updateAnimationSlider();

private:
    /// last time we received a menu event - prevent doubles
    double lastEventTime_;

    /// minimal time between 2 klicks
    double minTimeBetweenClicks_;

    /// current debugging level
    static int debugLevel_;

    osg::ref_ptr<osg::Stencil> defaultStencil;
    osg::ref_ptr<osg::Stencil> pointerStencil;
    osg::ref_ptr<osg::Stencil> menuStencil;

    osg::ref_ptr<osg::Depth> depth;

    bool oldStencil;
    bool coverMenu_;

    coSliderToolboxItem *animSlider_;

    AkSubMenu *coverMenuClone_;
};
#endif
