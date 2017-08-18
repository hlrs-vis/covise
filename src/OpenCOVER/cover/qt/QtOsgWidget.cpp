#include "QtOsgWidget.h"

#include <osg/Camera>
#include <osgGA/EventQueue>
#include <osgViewer/View>

#include <QKeyEvent>
#include <QWheelEvent>
#include <QInputEvent>
#include <QWindow>
#include <QOpenGLContext>

#include <cover/VRViewer.h>

class QtKeyboardMap
{

public:
    QtKeyboardMap()
    {
        mKeyMap[Qt::Key_Escape     ] = osgGA::GUIEventAdapter::KEY_Escape;
        mKeyMap[Qt::Key_Delete   ] = osgGA::GUIEventAdapter::KEY_Delete;
        mKeyMap[Qt::Key_Home       ] = osgGA::GUIEventAdapter::KEY_Home;
        mKeyMap[Qt::Key_Enter      ] = osgGA::GUIEventAdapter::KEY_KP_Enter;
        mKeyMap[Qt::Key_End        ] = osgGA::GUIEventAdapter::KEY_End;
        mKeyMap[Qt::Key_Return     ] = osgGA::GUIEventAdapter::KEY_Return;
        mKeyMap[Qt::Key_PageUp     ] = osgGA::GUIEventAdapter::KEY_Page_Up;
        mKeyMap[Qt::Key_PageDown   ] = osgGA::GUIEventAdapter::KEY_Page_Down;
        mKeyMap[Qt::Key_Left       ] = osgGA::GUIEventAdapter::KEY_Left;
        mKeyMap[Qt::Key_Right      ] = osgGA::GUIEventAdapter::KEY_Right;
        mKeyMap[Qt::Key_Up         ] = osgGA::GUIEventAdapter::KEY_Up;
        mKeyMap[Qt::Key_Down       ] = osgGA::GUIEventAdapter::KEY_Down;
        mKeyMap[Qt::Key_Backspace  ] = osgGA::GUIEventAdapter::KEY_BackSpace;
        mKeyMap[Qt::Key_Tab        ] = osgGA::GUIEventAdapter::KEY_Tab;
        mKeyMap[Qt::Key_Space      ] = osgGA::GUIEventAdapter::KEY_Space;
        mKeyMap[Qt::Key_Delete     ] = osgGA::GUIEventAdapter::KEY_Delete;
        mKeyMap[Qt::Key_Alt      ] = osgGA::GUIEventAdapter::KEY_Alt_L;
        mKeyMap[Qt::Key_Shift    ] = osgGA::GUIEventAdapter::KEY_Shift_L;
        mKeyMap[Qt::Key_Control  ] = osgGA::GUIEventAdapter::KEY_Control_L;
        mKeyMap[Qt::Key_Meta     ] = osgGA::GUIEventAdapter::KEY_Meta_L;

        mKeyMap[Qt::Key_F1             ] = osgGA::GUIEventAdapter::KEY_F1;
        mKeyMap[Qt::Key_F2             ] = osgGA::GUIEventAdapter::KEY_F2;
        mKeyMap[Qt::Key_F3             ] = osgGA::GUIEventAdapter::KEY_F3;
        mKeyMap[Qt::Key_F4             ] = osgGA::GUIEventAdapter::KEY_F4;
        mKeyMap[Qt::Key_F5             ] = osgGA::GUIEventAdapter::KEY_F5;
        mKeyMap[Qt::Key_F6             ] = osgGA::GUIEventAdapter::KEY_F6;
        mKeyMap[Qt::Key_F7             ] = osgGA::GUIEventAdapter::KEY_F7;
        mKeyMap[Qt::Key_F8             ] = osgGA::GUIEventAdapter::KEY_F8;
        mKeyMap[Qt::Key_F9             ] = osgGA::GUIEventAdapter::KEY_F9;
        mKeyMap[Qt::Key_F10            ] = osgGA::GUIEventAdapter::KEY_F10;
        mKeyMap[Qt::Key_F11            ] = osgGA::GUIEventAdapter::KEY_F11;
        mKeyMap[Qt::Key_F12            ] = osgGA::GUIEventAdapter::KEY_F12;
        mKeyMap[Qt::Key_F13            ] = osgGA::GUIEventAdapter::KEY_F13;
        mKeyMap[Qt::Key_F14            ] = osgGA::GUIEventAdapter::KEY_F14;
        mKeyMap[Qt::Key_F15            ] = osgGA::GUIEventAdapter::KEY_F15;
        mKeyMap[Qt::Key_F16            ] = osgGA::GUIEventAdapter::KEY_F16;
        mKeyMap[Qt::Key_F17            ] = osgGA::GUIEventAdapter::KEY_F17;
        mKeyMap[Qt::Key_F18            ] = osgGA::GUIEventAdapter::KEY_F18;
        mKeyMap[Qt::Key_F19            ] = osgGA::GUIEventAdapter::KEY_F19;
        mKeyMap[Qt::Key_F20            ] = osgGA::GUIEventAdapter::KEY_F20;

        mKeyMap[Qt::Key_hyphen         ] = '-';
        mKeyMap[Qt::Key_Equal         ] = '=';

        mKeyMap[Qt::Key_division      ] = osgGA::GUIEventAdapter::KEY_KP_Divide;
        mKeyMap[Qt::Key_multiply      ] = osgGA::GUIEventAdapter::KEY_KP_Multiply;
        mKeyMap[Qt::Key_Minus         ] = '-';
        mKeyMap[Qt::Key_Plus          ] = '+';
        //mKeyMap[Qt::Key_H              ] = osgGA::GUIEventAdapter::KEY_KP_Home;
        //mKeyMap[Qt::Key_                    ] = osgGA::GUIEventAdapter::KEY_KP_Up;
        //mKeyMap[92                    ] = osgGA::GUIEventAdapter::KEY_KP_Page_Up;
        //mKeyMap[86                    ] = osgGA::GUIEventAdapter::KEY_KP_Left;
        //mKeyMap[87                    ] = osgGA::GUIEventAdapter::KEY_KP_Begin;
        //mKeyMap[88                    ] = osgGA::GUIEventAdapter::KEY_KP_Right;
        //mKeyMap[83                    ] = osgGA::GUIEventAdapter::KEY_KP_End;
        //mKeyMap[84                    ] = osgGA::GUIEventAdapter::KEY_KP_Down;
        //mKeyMap[85                    ] = osgGA::GUIEventAdapter::KEY_KP_Page_Down;
        mKeyMap[Qt::Key_Insert        ] = osgGA::GUIEventAdapter::KEY_KP_Insert;
        //mKeyMap[Qt::Key_Delete        ] = osgGA::GUIEventAdapter::KEY_KP_Delete;
    }

    ~QtKeyboardMap()
    {
    }

    int remapKey(QKeyEvent* event)
    {
        KeyMap::iterator itr = mKeyMap.find(event->key());
        if (itr == mKeyMap.end())
        {
            return int(*(event->text().toLatin1().data()));
        }
        else
            return itr->second;
    }

private:
    typedef std::map<unsigned int, int> KeyMap;
    KeyMap mKeyMap;
};

static QtKeyboardMap s_QtKeyboardMap;

QtGraphicsWindow::QtGraphicsWindow(QOpenGLWidget *glWidget, int x, int y, int width, int height)
    : osgViewer::GraphicsWindowEmbedded(x, y, width, height)
    , m_glWidget(glWidget)
{
}

QOpenGLWidget *QtGraphicsWindow::widget() const
{
    return m_glWidget;
}

bool QtGraphicsWindow::makeCurrentImplementation()
{
    if (m_glWidget)
        m_glWidget->makeCurrent();
    return true;
}

bool QtGraphicsWindow::releaseContextImplementation()
{
    if (m_glWidget)
        m_glWidget->doneCurrent();
    return true;
}

bool QtGraphicsWindow::setWindowRectangleImplementation(int x, int y, int width, int height) {
    if (m_glWidget)
        m_glWidget->setGeometry(x, y, width, height);
    return true;
}

void QtGraphicsWindow::swapBuffersImplementation()
{
    if (m_glWidget)
        m_glWidget->context()->swapBuffers(m_glWidget->context()->surface());
}

void QtGraphicsWindow::useCursor(bool cursorOn)
{
    if (m_glWidget)
    {
        _traits->useCursor = cursorOn;
        if (!cursorOn)
            m_glWidget->setCursor(Qt::BlankCursor);
        else
            m_glWidget->setCursor(m_currentCursor);
    }
}

void QtGraphicsWindow::setCursor(MouseCursor cursor)
{
    if (cursor==InheritCursor && m_glWidget)
    {
        m_glWidget->unsetCursor();
    }

    switch (cursor)
    {
    case NoCursor: m_currentCursor = Qt::BlankCursor; break;
    case RightArrowCursor: case LeftArrowCursor: m_currentCursor = Qt::ArrowCursor; break;
    case InfoCursor: m_currentCursor = Qt::SizeAllCursor; break;
    case DestroyCursor: m_currentCursor = Qt::ForbiddenCursor; break;
    case HelpCursor: m_currentCursor = Qt::WhatsThisCursor; break;
    case CycleCursor: m_currentCursor = Qt::ForbiddenCursor; break;
    case SprayCursor: m_currentCursor = Qt::SizeAllCursor; break;
    case WaitCursor: m_currentCursor = Qt::WaitCursor; break;
    case TextCursor: m_currentCursor = Qt::IBeamCursor; break;
    case CrosshairCursor: m_currentCursor = Qt::CrossCursor; break;
    case HandCursor: m_currentCursor = Qt::OpenHandCursor; break;
    case UpDownCursor: m_currentCursor = Qt::SizeVerCursor; break;
    case LeftRightCursor: m_currentCursor = Qt::SizeHorCursor; break;
    case TopSideCursor: case BottomSideCursor: m_currentCursor = Qt::UpArrowCursor; break;
    case LeftSideCursor: case RightSideCursor: m_currentCursor = Qt::SizeHorCursor; break;
    case TopLeftCorner: m_currentCursor = Qt::SizeBDiagCursor; break;
    case TopRightCorner: m_currentCursor = Qt::SizeFDiagCursor; break;
    case BottomRightCorner: m_currentCursor = Qt::SizeBDiagCursor; break;
    case BottomLeftCorner: m_currentCursor = Qt::SizeFDiagCursor; break;
    default: break;
    };

    if (m_glWidget)
        m_glWidget->setCursor( m_currentCursor );
}

bool QtGraphicsWindow::setWindowDecorationImplementation(bool flag)
{
    auto win = dynamic_cast<QWindow *>(m_glWidget->parent());
    if (win)
    {
        auto f = win->flags();
        if (flag)
            f |= Qt::FramelessWindowHint;
        else
            f &= ~Qt::FramelessWindowHint;
        win->setFlags(f);
        return true;
    }
    return false;
}

QtOsgWidget::QtOsgWidget(QWidget *parent, Qt::WindowFlags f)
    : QOpenGLWidget(parent, f)
    , m_graphicsWindow(new QtGraphicsWindow(this, x(), y(), width(), height()))
{
    setFocusPolicy(Qt::StrongFocus);
    setMinimumSize(128, 128);
    setMouseTracking(true);
}

QtOsgWidget::~QtOsgWidget()
{
}

osgViewer::GraphicsWindowEmbedded *QtOsgWidget::graphicsWindow() const
{
    return m_graphicsWindow.get();
}

void QtOsgWidget::paintEvent(QPaintEvent *paintEvent)
{
    //opencover::VRViewer::instance()->requestRedraw();
}

void QtOsgWidget::paintGL()
{
}

void QtOsgWidget::resizeGL(int width, int height)
{
    auto pr = devicePixelRatio();
    getEventQueue()->windowResize(x()*pr, y()*pr, width*pr, height*pr);
    m_graphicsWindow->resized(x()*pr, y()*pr, width*pr, height*pr);
}

void QtOsgWidget::setKeyboardModifiers(QInputEvent *event)
{
    int modkey = event->modifiers() & (Qt::ShiftModifier | Qt::ControlModifier | Qt::AltModifier);
    unsigned int mask = 0;
    if (modkey & Qt::ShiftModifier) mask |= osgGA::GUIEventAdapter::MODKEY_SHIFT;
    if (modkey & Qt::ControlModifier) mask |= osgGA::GUIEventAdapter::MODKEY_CTRL;
    if (modkey & Qt::AltModifier) mask |= osgGA::GUIEventAdapter::MODKEY_ALT;
    getEventQueue()->getCurrentEventState()->setModKeyMask(mask);
}

void QtOsgWidget::keyPressEvent(QKeyEvent *event)
{
    setKeyboardModifiers(event);
    int value = s_QtKeyboardMap.remapKey(event);
    getEventQueue()->keyPress(value);
}

void QtOsgWidget::keyReleaseEvent(QKeyEvent *event)
{
    if( event->isAutoRepeat() )
    {
        event->ignore();
    }
    else
    {
        setKeyboardModifiers(event);
        int value = s_QtKeyboardMap.remapKey(event);
        getEventQueue()->keyRelease(value);
    }
}

void QtOsgWidget::mouseMoveEvent(QMouseEvent *event )
{
    auto pr = devicePixelRatio();
    getEventQueue()->mouseMotion(event->x()*pr, event->y()*pr);
}

void QtOsgWidget::mousePressEvent(QMouseEvent *event)
{
    unsigned int button = 0;
    switch(event->button())
    {
    case Qt::LeftButton:
        button = 1;
        break;

    case Qt::MiddleButton:
        button = 2;
        break;

    case Qt::RightButton:
        button = 3;
        break;

    default:
        break;
    }

    auto pr = devicePixelRatio();
    getEventQueue()->mouseButtonPress(event->x()*pr, event->y()*pr, button);
}

void QtOsgWidget::mouseReleaseEvent(QMouseEvent* event)
{
    unsigned int button = 0;
    switch( event->button() )
    {
    case Qt::LeftButton:
        button = 1;
        break;

    case Qt::MiddleButton:
        button = 2;
        break;

    case Qt::RightButton:
        button = 3;
        break;

    default:
        break;
    }

    auto pr = devicePixelRatio();
    getEventQueue()->mouseButtonRelease(event->x(), pr, event->y()*pr, button);
}

void QtOsgWidget::wheelEvent(QWheelEvent *event)
{
    event->accept();
    int delta = event->delta();

    osgGA::GUIEventAdapter::ScrollingMotion motion =
            delta>0 ? osgGA::GUIEventAdapter::SCROLL_UP : osgGA::GUIEventAdapter::SCROLL_DOWN;
    getEventQueue()->mouseScroll(motion);
}

osgGA::EventQueue* QtOsgWidget::getEventQueue() const
{
    return m_graphicsWindow->getEventQueue();
}
