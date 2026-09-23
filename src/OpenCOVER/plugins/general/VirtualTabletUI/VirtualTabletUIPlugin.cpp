/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "VirtualTabletUIPlugin.h"
#include "tui/TUIStyleSheet.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>
#include <cover/coCommandLine.h>
#include <cover/VRSceneGraph.h>
#include <cover/coVRMSController.h>
#include <cover/OpenCOVER.h>
#include <cover/VRWindow.h>
#include <cover/ui/Manager.h>
#include <cover/ui/Action.h>
#include <cover/ui/Button.h>
#include <cover/ui/Menu.h>
#include <cover/ui/TabletView.h>
#include <cover/coTabletUI.h>
#include <cover/coVRTui.h>
#include <config/CoviseConfig.h>
#include <config/coConfig.h>

#include <QtGlobal>
#include <QMenuBar>
#include <QToolBar>
#include <QApplication>
#include <QLayout>
#include <QMessageBox>
#include <QDialog>
#include <QWindow>
#include <QFile>
#include <QPushButton>
#include <QPainter>
#include <QEvent>
#include <QFocusEvent>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>

#include <cover/coIntersection.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coAction.h>
#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coPopupHandle.h>
#include <OpenVRUI/coTexturedBackground.h>
#include <OpenVRUI/coTrackerButtonInteraction.h>
#include <OpenVRUI/sginterface/vruiTransformNode.h>
#include <OpenVRUI/coCombinedButtonInteraction.h>
#include <OpenVRUI/osg/OSGVruiHit.h>
#include <OpenVRUI/osg/OSGVruiMatrix.h>

#include <cassert>

#ifdef USE_X11
#ifdef HAVE_QTX11EXTRAS
#include <QX11Info>
#endif

#include <X11/Xlib.h>
#include <X11/Xatom.h>
#endif

#include <QStyleFactory>

#include <tui/TUIMainWindow.h>

using namespace opencover;

// Undefine these from X11 headers so as to not confuse the compiler what QEvent::* means
#ifdef FocusOut
#undef FocusOut
#endif
#ifdef FocusIn
#undef FocusIn
#endif
#ifdef KeyPress
#undef KeyPress
#endif
#ifdef KeyRelease
#undef KeyRelease
#endif

// Produced with the help of
// https://github.com/hluk/qxtglobalshortcut/blob/16446200b699e0610b8a5fb20b74938225d81d87/src/xcbkeyboard.h#L249,
// Copyright (c) 2006 - 2011, the LibQxt project.
// clang-format off
static const unsigned int KeyTbl[] = {
    // misc keys
    osgGA::GUIEventAdapter::KEY_Escape, Qt::Key_Escape,
    osgGA::GUIEventAdapter::KEY_Tab, Qt::Key_Tab,
    osgGA::GUIEventAdapter::KEY_BackSpace, Qt::Key_Backspace,
    osgGA::GUIEventAdapter::KEY_Return, Qt::Key_Return,
    osgGA::GUIEventAdapter::KEY_Insert, Qt::Key_Insert,
    osgGA::GUIEventAdapter::KEY_Delete, Qt::Key_Delete,
    osgGA::GUIEventAdapter::KEY_Clear, Qt::Key_Delete,
    osgGA::GUIEventAdapter::KEY_Pause, Qt::Key_Pause,
    osgGA::GUIEventAdapter::KEY_Print, Qt::Key_Print,

    // cursor movement

    osgGA::GUIEventAdapter::KEY_Home, Qt::Key_Home,
    osgGA::GUIEventAdapter::KEY_End, Qt::Key_End,
    osgGA::GUIEventAdapter::KEY_Left, Qt::Key_Left,
    osgGA::GUIEventAdapter::KEY_Up, Qt::Key_Up,
    osgGA::GUIEventAdapter::KEY_Right, Qt::Key_Right,
    osgGA::GUIEventAdapter::KEY_Down, Qt::Key_Down,
    osgGA::GUIEventAdapter::KEY_Prior, Qt::Key_PageUp,
    osgGA::GUIEventAdapter::KEY_Next, Qt::Key_PageDown,

    // modifiers

    osgGA::GUIEventAdapter::KEY_Shift_L, Qt::Key_Shift,
    osgGA::GUIEventAdapter::KEY_Shift_R, Qt::Key_Shift,
    osgGA::GUIEventAdapter::KEY_Shift_Lock, Qt::Key_Shift,
    osgGA::GUIEventAdapter::KEY_Control_L, Qt::Key_Control,
    osgGA::GUIEventAdapter::KEY_Control_R, Qt::Key_Control,
    osgGA::GUIEventAdapter::KEY_Meta_L, Qt::Key_Meta,
    osgGA::GUIEventAdapter::KEY_Meta_R, Qt::Key_Meta,
    osgGA::GUIEventAdapter::KEY_Alt_L, Qt::Key_Alt,
    osgGA::GUIEventAdapter::KEY_Alt_R, Qt::Key_Alt,
    osgGA::GUIEventAdapter::KEY_Caps_Lock, Qt::Key_CapsLock,
    osgGA::GUIEventAdapter::KEY_Num_Lock, Qt::Key_NumLock,
    osgGA::GUIEventAdapter::KEY_Scroll_Lock, Qt::Key_ScrollLock,
    osgGA::GUIEventAdapter::KEY_Super_L, Qt::Key_Super_L,
    osgGA::GUIEventAdapter::KEY_Super_R, Qt::Key_Super_R,
    osgGA::GUIEventAdapter::KEY_Menu, Qt::Key_Menu,
    osgGA::GUIEventAdapter::KEY_Hyper_L, Qt::Key_Hyper_L,
    osgGA::GUIEventAdapter::KEY_Hyper_R, Qt::Key_Hyper_R,
    osgGA::GUIEventAdapter::KEY_Help, Qt::Key_Help,

    // numeric and function keypad keys

    osgGA::GUIEventAdapter::KEY_KP_Space, Qt::Key_Space,
    osgGA::GUIEventAdapter::KEY_KP_Tab, Qt::Key_Tab,
    osgGA::GUIEventAdapter::KEY_KP_Enter, Qt::Key_Enter,
    osgGA::GUIEventAdapter::KEY_KP_F1, Qt::Key_F1,
    osgGA::GUIEventAdapter::KEY_KP_F2, Qt::Key_F2,
    osgGA::GUIEventAdapter::KEY_KP_F3, Qt::Key_F3,
    osgGA::GUIEventAdapter::KEY_KP_F4, Qt::Key_F4,
    osgGA::GUIEventAdapter::KEY_KP_Home, Qt::Key_Home,
    osgGA::GUIEventAdapter::KEY_KP_Left, Qt::Key_Left,
    osgGA::GUIEventAdapter::KEY_KP_Up, Qt::Key_Up,
    osgGA::GUIEventAdapter::KEY_KP_Right, Qt::Key_Right,
    osgGA::GUIEventAdapter::KEY_KP_Down, Qt::Key_Down,
    osgGA::GUIEventAdapter::KEY_KP_Prior, Qt::Key_PageUp,
    osgGA::GUIEventAdapter::KEY_KP_Next, Qt::Key_PageDown,
    osgGA::GUIEventAdapter::KEY_KP_End, Qt::Key_End,
    osgGA::GUIEventAdapter::KEY_KP_Begin, Qt::Key_Clear,
    osgGA::GUIEventAdapter::KEY_KP_Insert, Qt::Key_Insert,
    osgGA::GUIEventAdapter::KEY_KP_Delete, Qt::Key_Delete,
    osgGA::GUIEventAdapter::KEY_KP_Equal, Qt::Key_Equal,
    osgGA::GUIEventAdapter::KEY_KP_Multiply, Qt::Key_Asterisk,
    osgGA::GUIEventAdapter::KEY_KP_Add, Qt::Key_Plus,
    osgGA::GUIEventAdapter::KEY_KP_Separator, Qt::Key_Comma,
    osgGA::GUIEventAdapter::KEY_KP_Subtract, Qt::Key_Minus,
    osgGA::GUIEventAdapter::KEY_KP_Decimal, Qt::Key_Period,
    osgGA::GUIEventAdapter::KEY_KP_Divide, Qt::Key_Slash,

    // Misc Functions
    osgGA::GUIEventAdapter::KEY_Mode_switch, Qt::Key_Mode_switch,

    // end of table,
    0, 0,
};
// clang-format on
//
Qt::Key mapKey(int keySym)
{
    int i = 0, k;
    while (true)
    {
        if (KeyTbl[i] == 0)
            return (Qt::Key)0;
        if (KeyTbl[i] == keySym)
            return (Qt::Key)KeyTbl[i + 1];
        i += 2;
    }
}

TUIMainWidget::TUIMainWidget(const QSize &size)
    : TUIMain(this)
{
    port = covise::coCoviseConfig::getInt("port", "COVER.TabletUI", port);
    mainFrame = new QFrame(this);
    mainFrame->setFixedSize(size);
    mainFrame->setContentsMargins(1, 1, 1, 1);
    mainFrame->setFrameStyle(QFrame::NoFrame | QFrame::Plain);
    mainGrid = new QGridLayout(mainFrame);
}

QObject *TUIMainWidget::getEventTarget()
{
    return this;
}

void TUIMainWidget::processMessages()
{
    TUIMain::processMessages();
}

VirtualTabletUIPlugin::VirtualTabletUIPlugin()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , m_interactionA(new vrui::coCombinedButtonInteraction(vrui::coInteraction::ButtonA, "CEFBrowser", vrui::coInteraction::Menu))
    , m_interactionB(new vrui::coCombinedButtonInteraction(vrui::coInteraction::ButtonB, "CEFBrowser", vrui::coInteraction::Menu))
    , m_interactionC(new vrui::coCombinedButtonInteraction(vrui::coInteraction::ButtonC, "CEFBrowser", vrui::coInteraction::Menu))
    , m_interactionWheel(new vrui::coCombinedButtonInteraction(vrui::coInteraction::WheelVertical, "CEFBrowser", vrui::coInteraction::Menu))
    , m_popupHandle(new vrui::coPopupHandle("VirtualTabletUI"))
    , m_videoTexture(new vrui::coTexturedBackground(NULL, NULL, NULL, 4, 32, 32, 0))

{
    double px = 50;
    double py = -300;
    double pz = 25;

    // default is Mathematic.MenuSize then COVER.Menu.Size then 1.0
    float s = 1.0;

    // vrui::OSGVruiMatrix matrix, transMatrix, rotateMatrix, scaleMatrix;
    // transMatrix.makeTranslate(px, py, pz);
    // rotateMatrix.makeEuler(0, 90, 0);
    // scaleMatrix.makeScale(s, s, s);
    //
    // matrix.makeIdentity();
    // matrix.mult(&scaleMatrix);
    // matrix.mult(&rotateMatrix);
    // matrix.mult(&transMatrix);
    //
    // m_popupHandle->setTransformMatrix(&matrix);
    // m_popupHandle->setScale(cover->getSceneSize() / 2500);

    m_popupHandle->setScale(10 * cover->getSceneSize() / 2500);
    m_popupHandle->setPos(-m_width * cover->getSceneSize() / 2500, 0, -m_height * cover->getSceneSize() / 2500);
    m_popupHandle->addElement(m_videoTexture);
    m_popupHandle->setVisible(true);
}

// this is called if the plugin is removed at runtime
VirtualTabletUIPlugin::~VirtualTabletUIPlugin()
{
    // TODO: clean up qt?
    delete main;
}

void VirtualTabletUIPlugin::renderWidget()
{
    m_image.fill(Qt::transparent);

    QPainter painter(&m_image);
    main->render(&painter);
    painter.end();

    // OSG/OpenGL normally expects the origin at the lower-left.
    m_image.flip(Qt::Vertical);

    m_videoTexture->setUpdated(true);
    m_videoTexture->setTexSize(1024, 1024);
    m_videoTexture->setImage((uint *)m_image.constBits(), NULL, NULL, 4, m_width, m_height, 0, vrui::coTexturedBackground::TextureSet::PF_RGBA);
}

bool VirtualTabletUIPlugin::init()
{
    if (!qApp)
    {
        QApplication::setAttribute(Qt::AA_UseDesktopOpenGL);
        QApplication::setAttribute(Qt::AA_ShareOpenGLContexts);
        new QApplication(coCommandLine::argc(), coCommandLine::argv());
        qApp->setWindowIcon(QIcon(":/icons/cover.ico"));
        // qApp->setAttribute(Qt::AA_PluginApplication);
        qApp->setAttribute(Qt::AA_MacDontSwapCtrlAndMeta);
        qApp->setAttribute(Qt::AA_DontCheckOpenGLContextThreadAffinity);
        qApp->setStyleSheet(QString::fromStdString(TUI_STYLESHEET));
#ifdef __APPLE__
        qApp->setAttribute(Qt::AA_DontShowIconsInMenus);
#endif
        m_ownsQApp = true;
    }

    // fprintf(stderr, "VirtualTabletUIPlugin::VirtualTabletUIPlugin\n");
    main = new TUIMainWidget(QSize(m_width, m_height));
    main->setUpdatesEnabled(true);
    main->openServer();

    coIntersection::getIntersectorForAction("coAction")->add(m_videoTexture->getDCS(), this);
    m_videoTexture->setSize(m_width, m_height, 0);
    m_videoTexture->setTexSize(m_width, -m_height);
    m_videoTexture->setMinWidth(m_width);
    m_videoTexture->setMinHeight(m_height);

    // m_texture = new osg::Texture2D;
    // m_osgImage = new osg::Image;

    const qreal dpr = main->devicePixelRatioF();
    m_image.setDevicePixelRatio(dpr);
    m_image = QImage(
        int(m_width * dpr),
        int(m_height * dpr),
        QImage::Format_RGBA8888);

    renderWidget();

    return true;
}

bool VirtualTabletUIPlugin::destroy()
{
    if (m_ownsQApp)
    {
        qApp->quit();
        qApp->sendPostedEvents();
        qApp->processEvents();
        delete qApp;
    }

    return true;
}

bool VirtualTabletUIPlugin::update()
{
    if (unregister)
    {
        if (m_interactionA->isRegistered() && (m_interactionA->getState() != vrui::coInteraction::Active))
            vrui::coInteractionManager::the()->unregisterInteraction(m_interactionA);

        if (m_interactionB->isRegistered() && (m_interactionB->getState() != vrui::coInteraction::Active))
            vrui::coInteractionManager::the()->unregisterInteraction(m_interactionB);

        if (m_interactionC->isRegistered() && (m_interactionC->getState() != vrui::coInteraction::Active))
            vrui::coInteractionManager::the()->unregisterInteraction(m_interactionC);

        if (m_interactionWheel->isRegistered() && (m_interactionWheel->getState() != vrui::coInteraction::Active))
            vrui::coInteractionManager::the()->unregisterInteraction(m_interactionWheel);

        if (!m_interactionA->isRegistered() && !m_interactionB->isRegistered() && !m_interactionC->isRegistered() && !m_interactionWheel->isRegistered())
            unregister = false;
    }

    if (qApp)
    {
        qApp->sendPostedEvents();
        qApp->processEvents();
    }
    // main->update();
    // main->repaint();
    renderWidget();
    return false;
}
void VirtualTabletUIPlugin::moveFocus(QWidget *widget)
{
    if (m_focusWidget == widget)
        return;

    if (m_focusWidget)
    {
        QApplication::postEvent(m_focusWidget, new QFocusEvent(QEvent::FocusOut));
    }

    if (widget)
    {
        cover->grabKeyboard(this);
        main->setFocus(Qt::FocusReason::ActiveWindowFocusReason);
        QApplication::postEvent(widget, new QFocusEvent(QEvent::FocusIn));
    }
    else
    {
        cover->releaseKeyboard(this);
        main->setFocus(Qt::FocusReason::NoFocusReason);
    }
    m_focusWidget = widget;
    m_previousFocusWidget = widget;
}

int VirtualTabletUIPlugin::hit(vrui::vruiHit *hit)
{
    // if (coVRCollaboration::instance()->getCouplingMode() == coVRCollaboration::MasterSlaveCoupling && !coVRCollaboration::instance()->isMaster())
    //     return ACTION_DONE;

    if (!m_interactionA->isRegistered())
    {
        vrui::coInteractionManager::the()->registerInteraction(m_interactionA);
        m_interactionA->setHitByMouse(hit->isMouseHit());
    }
    if (!m_interactionB->isRegistered())
    {
        vrui::coInteractionManager::the()->registerInteraction(m_interactionB);
        m_interactionB->setHitByMouse(hit->isMouseHit());
    }
    if (!m_interactionC->isRegistered())
    {
        vrui::coInteractionManager::the()->registerInteraction(m_interactionC);
        m_interactionC->setHitByMouse(hit->isMouseHit());
    }
    if (!m_interactionWheel->isRegistered())
    {
        vrui::coInteractionManager::the()->registerInteraction(m_interactionWheel);
        m_interactionWheel->setHitByMouse(hit->isMouseHit());
    }

    osgUtil::LineSegmentIntersector::Intersection osgHit = dynamic_cast<vrui::OSGVruiHit *>(hit)->getHit();

    if (!osgHit.drawable.valid())
        return ACTION_CALL_ON_MISS;

    osg::Vec3 point = osgHit.getLocalIntersectPoint();

    QPointF pos(
        std::clamp<int>(point[0], 0, m_width),
        m_height - std::clamp<int>(point[1], 0, m_height));

    QWidget *receiver = main;
    QPointF localPos = pos;
    while (auto child = receiver->childAt(localPos))
    {
        localPos = child->mapFrom(receiver, pos);
        receiver = child;
    }

    // QObject *receiver = main->getEventTarget();
    if (!receiver)
    {
        // focus the widget that was focused before we left the frame, ie. before the "window" lost its focus
        moveFocus(m_previousFocusWidget);
        return ACTION_CALL_ON_MISS;
    }

    moveFocus(receiver);

    if (m_interactionA->wasStarted())
    {
        m_currentMouseButton |= Qt::LeftButton;
        QApplication::postEvent(receiver, new QMouseEvent(QEvent::MouseButtonPress, localPos, pos, Qt::LeftButton, m_currentMouseButton, Qt::NoModifier));
    }
    else if (m_interactionA->wasStopped())
    {
        m_currentMouseButton &= ~Qt::LeftButton;
        QApplication::postEvent(receiver, new QMouseEvent(QEvent::MouseButtonRelease, localPos, pos, Qt::LeftButton, m_currentMouseButton, Qt::NoModifier));
    }

    if (m_interactionB->wasStarted())
    {
        m_currentMouseButton |= Qt::MiddleButton;
        QApplication::postEvent(receiver, new QFocusEvent(QEvent::FocusIn));
        QApplication::postEvent(receiver, new QMouseEvent(QEvent::MouseButtonPress, localPos, pos, Qt::MiddleButton, m_currentMouseButton, Qt::NoModifier));
    }
    else if (m_interactionB->wasStopped())
    {
        m_currentMouseButton &= ~Qt::MiddleButton;
        QApplication::postEvent(receiver, new QMouseEvent(QEvent::MouseButtonRelease, localPos, pos, Qt::MiddleButton, m_currentMouseButton, Qt::NoModifier));
    }

    if (m_interactionC->wasStarted())
    {
        m_currentMouseButton |= Qt::RightButton;
        QApplication::postEvent(receiver, new QFocusEvent(QEvent::FocusIn));
        QApplication::postEvent(receiver, new QMouseEvent(QEvent::MouseButtonPress, localPos, pos, Qt::RightButton, m_currentMouseButton, Qt::NoModifier));
    }
    else if (m_interactionC->wasStopped())
    {
        m_currentMouseButton &= ~Qt::RightButton;
        QApplication::postEvent(receiver, new QMouseEvent(QEvent::MouseButtonRelease, localPos, pos, Qt::RightButton, m_currentMouseButton, Qt::NoModifier));
    }

    if (m_interactionWheel->wasStarted() || m_interactionWheel->isRunning())
    {
        int wheel = m_interactionWheel->getWheelCount();
        QPoint pixel(0, wheel);
        QPoint angle(0, wheel * 125);
        QApplication::postEvent(receiver, new QWheelEvent(localPos, pos, pixel, angle, m_currentMouseButton, Qt::NoModifier, Qt::ScrollBegin, false, Qt::MouseEventSynthesizedByApplication));
    }

    QApplication::postEvent(receiver, new QMouseEvent(QEvent::MouseMove, localPos, pos, Qt::NoButton, m_currentMouseButton, Qt::NoModifier));

    return ACTION_CALL_ON_MISS;
}

void VirtualTabletUIPlugin::miss()
{
    // remove focus from UI
    unregister = true;

    m_currentMouseButton = Qt::NoButton;

    if (m_focusWidget)
    {
        moveFocus(nullptr);
    }
}

void VirtualTabletUIPlugin::key(int type, int keySym, int mod)
{
    if (!m_focusWidget)
        return;
    if (!coVRMSController::instance()->isMaster())
        return;
    bool down = type == osgGA::GUIEventAdapter::KEYDOWN;
    bool up = type == osgGA::GUIEventAdapter::KEYUP;

    if (!down && !up)
        return;

    Qt::KeyboardModifiers modifiers = Qt::NoModifier;

    if (mod & osgGA::GUIEventAdapter::MODKEY_SHIFT)
        modifiers |= Qt::ShiftModifier;
    if (mod & osgGA::GUIEventAdapter::MODKEY_CTRL)
        modifiers |= Qt::ControlModifier;
    if (mod & osgGA::GUIEventAdapter::MODKEY_ALT)
        modifiers |= Qt::AltModifier;
    if (mod & osgGA::GUIEventAdapter::MODKEY_META)
        modifiers |= Qt::MetaModifier;
    if (mod & osgGA::GUIEventAdapter::MODKEY_SUPER)
        modifiers |= Qt::MetaModifier; // ??

    // this is totally wrong and hacky, but works for basic ascii for now, since keySyms and ascii overlap :)
    QString text = QString::fromUtf16((char16_t *)&keySym, 1);
    if (!text.isValidUtf16())
        text = "";

    // auto text = QString::fromLocal8Bit(std::string(1, c).c_str());
    auto event = new QKeyEvent(down ? (QEvent::KeyPress) : (QEvent::KeyRelease), mapKey(keySym), modifiers, text);
    QApplication::postEvent(m_focusWidget, event);
}

COVERPLUGIN(VirtualTabletUIPlugin)
