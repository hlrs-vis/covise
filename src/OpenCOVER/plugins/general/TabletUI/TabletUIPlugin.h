/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TABLET_UI_PLUGIN_H
#define TABLET_UI_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>
#include <QMetaObject>

class TUIMainWindow;

namespace opencover
{
class coTabletUI;
class coVRTui;
} // namespace opencover

class TabletUIPlugin : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    TabletUIPlugin();
    ~TabletUIPlugin();
    bool init() override;
    bool destroy() override;
    bool update() override;

    TUIMainWindow *m_window = nullptr;
    QMetaObject::Connection m_connection;
    bool m_ownsQApp = false;
    int m_tuiFds[2]{-1, -1};
    int m_tuiSgFds[2]{-1, -1};
};
#endif
