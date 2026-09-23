/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TABLET_UI_PLUGIN_H
#define TABLET_UI_PLUGIN_H

#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>
#include <QMetaObject>
#include <QWidget>
#include <osg/Texture2D>
#include <OpenVRUI/coAction.h>

#include <tui/TUIMain.h>

namespace opencover
{
class coTabletUI;
class coVRTui;
} // namespace opencover

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
class coCheckboxMenuItem;
class coButtonMenuItem;
class coPotiMenuItem;
class coTrackerButtonInteraction;
class coTexturedBackground;
class coPopupHandle;
class coLabel;
class coCombinedButtonInteraction;
class coColoredBackground;
class vruiHit;
class OSGVruiTransformNode;
}

class TUIMainWidget : public QWidget, public TUIMain
{
    Q_OBJECT

public:
    TUIMainWidget(const QSize &size);
    QObject *getEventTarget();
private slots:
    void processMessages();
};

class VirtualTabletUIPlugin : public opencover::coVRPlugin, public vrui::coAction
{
public:
    VirtualTabletUIPlugin();
    ~VirtualTabletUIPlugin();
    bool init() override;
    bool destroy() override;
    bool update() override;

    void createGeometry();
    void renderWidget();

    int hit(vrui::vruiHit *hit) override;
    void miss() override;
    void key(int type, int keySym, int mod) override;

    bool unregister;

    void moveFocus(QWidget *widget);

    QWidget *m_mainWidget;
    QWidget *m_focusWidget = nullptr;
    QWidget *m_previousFocusWidget = nullptr;
    TUIMainWidget *main;
    int m_width = 1024;
    int m_height = 1024;
    QImage m_image;
    //
    // osg::ref_ptr<osg::Texture2D> m_texture;
    // osg::ref_ptr<osg::Image> m_osgImage;

    QMetaObject::Connection m_connection;
    bool m_ownsQApp = false;

    QFlags<Qt::MouseButton> m_currentMouseButton = Qt::NoButton;

    vrui::coCombinedButtonInteraction *m_interactionA;
    vrui::coCombinedButtonInteraction *m_interactionB;
    vrui::coCombinedButtonInteraction *m_interactionC;
    vrui::coCombinedButtonInteraction *m_interactionWheel;

    vrui::coPopupHandle *m_popupHandle = nullptr;
    vrui::coTexturedBackground *m_videoTexture = nullptr;
};
#endif
