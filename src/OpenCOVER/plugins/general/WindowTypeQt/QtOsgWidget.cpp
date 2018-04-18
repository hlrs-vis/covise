#include "QtOsgWidget.h"

#include <osg/Camera>
#include <osgGA/EventQueue>
#include <osgViewer/View>

#include <QKeyEvent>
#include <QWheelEvent>
#include <QInputEvent>
#include <QWindow>
#include <QOpenGLContext>
#include <QApplication>

#include <cover/VRViewer.h>

// there is no header where this could be found
void qt_set_sequence_auto_mnemonic(bool b);

class QtKeyboardMap
{

public:
    QtKeyboardMap()
    {
        using namespace osgGA;

#ifdef Q_OS_MACOS
        mKeyMap[Qt::Key_Control]         = GUIEventAdapter::KEY_Meta_L;
        mKeyMap[Qt::Key_Meta]            = GUIEventAdapter::KEY_Control_L;
#else
        mKeyMap[Qt::Key_Control]         = GUIEventAdapter::KEY_Control_L;
        mKeyMap[Qt::Key_Meta]            = GUIEventAdapter::KEY_Meta_L;
#endif
        mKeyMap[Qt::Key_Alt]             = GUIEventAdapter::KEY_Alt_L;
        mKeyMap[Qt::Key_Shift]           = GUIEventAdapter::KEY_Shift_L;

        mKeyMap[Qt::Key_Escape]          = GUIEventAdapter::KEY_Escape;
        mKeyMap[Qt::Key_Delete]          = GUIEventAdapter::KEY_Delete;
        mKeyMap[Qt::Key_Home]            = GUIEventAdapter::KEY_Home;
        mKeyMap[Qt::Key_Enter]           = GUIEventAdapter::KEY_KP_Enter;
        mKeyMap[Qt::Key_End]             = GUIEventAdapter::KEY_End;
        mKeyMap[Qt::Key_Return]          = GUIEventAdapter::KEY_Return;
        mKeyMap[Qt::Key_PageUp]          = GUIEventAdapter::KEY_Page_Up;
        mKeyMap[Qt::Key_PageDown]        = GUIEventAdapter::KEY_Page_Down;
        mKeyMap[Qt::Key_Left]            = GUIEventAdapter::KEY_Left;
        mKeyMap[Qt::Key_Right]           = GUIEventAdapter::KEY_Right;
        mKeyMap[Qt::Key_Up]              = GUIEventAdapter::KEY_Up;
        mKeyMap[Qt::Key_Down]            = GUIEventAdapter::KEY_Down;
        mKeyMap[Qt::Key_Backspace]       = GUIEventAdapter::KEY_BackSpace;
        mKeyMap[Qt::Key_Tab]             = GUIEventAdapter::KEY_Tab;
        mKeyMap[Qt::Key_Space]           = GUIEventAdapter::KEY_Space;
        mKeyMap[Qt::Key_Delete]          = GUIEventAdapter::KEY_Delete;

        mKeyMap[Qt::Key_F1]              = GUIEventAdapter::KEY_F1;
        mKeyMap[Qt::Key_F2]              = GUIEventAdapter::KEY_F2;
        mKeyMap[Qt::Key_F3]              = GUIEventAdapter::KEY_F3;
        mKeyMap[Qt::Key_F4]              = GUIEventAdapter::KEY_F4;
        mKeyMap[Qt::Key_F5]              = GUIEventAdapter::KEY_F5;
        mKeyMap[Qt::Key_F6]              = GUIEventAdapter::KEY_F6;
        mKeyMap[Qt::Key_F7]              = GUIEventAdapter::KEY_F7;
        mKeyMap[Qt::Key_F8]              = GUIEventAdapter::KEY_F8;
        mKeyMap[Qt::Key_F9]              = GUIEventAdapter::KEY_F9;
        mKeyMap[Qt::Key_F10]             = GUIEventAdapter::KEY_F10;
        mKeyMap[Qt::Key_F11]             = GUIEventAdapter::KEY_F11;
        mKeyMap[Qt::Key_F12]             = GUIEventAdapter::KEY_F12;
        mKeyMap[Qt::Key_F13]             = GUIEventAdapter::KEY_F13;
        mKeyMap[Qt::Key_F14]             = GUIEventAdapter::KEY_F14;
        mKeyMap[Qt::Key_F15]             = GUIEventAdapter::KEY_F15;
        mKeyMap[Qt::Key_F16]             = GUIEventAdapter::KEY_F16;
        mKeyMap[Qt::Key_F17]             = GUIEventAdapter::KEY_F17;
        mKeyMap[Qt::Key_F18]             = GUIEventAdapter::KEY_F18;
        mKeyMap[Qt::Key_F19]             = GUIEventAdapter::KEY_F19;
        mKeyMap[Qt::Key_F20]             = GUIEventAdapter::KEY_F20;

        mKeyMap[Qt::Key_hyphen] = GUIEventAdapter::KEY_Minus;
        mKeyMap[Qt::Key_Comma] = GUIEventAdapter::KEY_Comma;
        mKeyMap[Qt::Key_Period] = GUIEventAdapter::KEY_Period;
        mKeyMap[Qt::Key_Slash] = GUIEventAdapter::KEY_Slash;
        mKeyMap[Qt::Key_Colon] = GUIEventAdapter::KEY_Colon;
        mKeyMap[Qt::Key_Semicolon] = GUIEventAdapter::KEY_Semicolon;
        mKeyMap[Qt::Key_At] = GUIEventAdapter::KEY_At;
        mKeyMap[Qt::Key_Less] = GUIEventAdapter::KEY_Less;
        mKeyMap[Qt::Key_Equal] = GUIEventAdapter::KEY_Equals;
        mKeyMap[Qt::Key_Greater] = GUIEventAdapter::KEY_Greater;

        mKeyMap[Qt::Key_division] = GUIEventAdapter::KEY_KP_Divide;
        mKeyMap[Qt::Key_multiply] = GUIEventAdapter::KEY_KP_Multiply;
        mKeyMap[Qt::Key_Minus] = GUIEventAdapter::KEY_Minus;
        mKeyMap[Qt::Key_Plus] = GUIEventAdapter::KEY_Plus;
        mKeyMap[Qt::Key_Asterisk] = GUIEventAdapter::KEY_Asterisk;
        mKeyMap[Qt::Key_Apostrophe] = GUIEventAdapter::KEY_Quote;
        mKeyMap[Qt::Key_QuoteDbl] = GUIEventAdapter::KEY_Quotedbl;
        mKeyMap[Qt::Key_Question] = GUIEventAdapter::KEY_Question;
        //mKeyMap[Qt::Key_H              ] = osgGA::GUIEventAdapter::KEY_KP_Home;
        //mKeyMap[Qt::Key_                    ] = osgGA::GUIEventAdapter::KEY_KP_Up;
        //mKeyMap[92                    ] = osgGA::GUIEventAdapter::KEY_KP_Page_Up;
        //mKeyMap[86                    ] = osgGA::GUIEventAdapter::KEY_KP_Left;
        //mKeyMap[87                    ] = osgGA::GUIEventAdapter::KEY_KP_Begin;
        //mKeyMap[88                    ] = osgGA::GUIEventAdapter::KEY_KP_Right;
        //mKeyMap[83                    ] = osgGA::GUIEventAdapter::KEY_KP_End;
        //mKeyMap[84                    ] = osgGA::GUIEventAdapter::KEY_KP_Down;
        //mKeyMap[85                    ] = osgGA::GUIEventAdapter::KEY_KP_Page_Down;
        mKeyMap[Qt::Key_Insert] = GUIEventAdapter::KEY_KP_Insert;
        //mKeyMap[Qt::Key_Delete        ] = osgGA::GUIEventAdapter::KEY_KP_Delete;

        mKeyMap[Qt::Key_A] = GUIEventAdapter::KEY_A;
        mKeyMap[Qt::Key_B] = GUIEventAdapter::KEY_B;
        mKeyMap[Qt::Key_C] = GUIEventAdapter::KEY_C;
        mKeyMap[Qt::Key_D] = GUIEventAdapter::KEY_D;
        mKeyMap[Qt::Key_E] = GUIEventAdapter::KEY_E;
        mKeyMap[Qt::Key_F] = GUIEventAdapter::KEY_F;
        mKeyMap[Qt::Key_G] = GUIEventAdapter::KEY_G;
        mKeyMap[Qt::Key_H] = GUIEventAdapter::KEY_H;
        mKeyMap[Qt::Key_I] = GUIEventAdapter::KEY_I;
        mKeyMap[Qt::Key_J] = GUIEventAdapter::KEY_J;
        mKeyMap[Qt::Key_K] = GUIEventAdapter::KEY_K;
        mKeyMap[Qt::Key_L] = GUIEventAdapter::KEY_L;
        mKeyMap[Qt::Key_M] = GUIEventAdapter::KEY_M;
        mKeyMap[Qt::Key_N] = GUIEventAdapter::KEY_N;
        mKeyMap[Qt::Key_O] = GUIEventAdapter::KEY_O;
        mKeyMap[Qt::Key_P] = GUIEventAdapter::KEY_P;
        mKeyMap[Qt::Key_Q] = GUIEventAdapter::KEY_Q;
        mKeyMap[Qt::Key_R] = GUIEventAdapter::KEY_R;
        mKeyMap[Qt::Key_S] = GUIEventAdapter::KEY_S;
        mKeyMap[Qt::Key_T] = GUIEventAdapter::KEY_T;
        mKeyMap[Qt::Key_U] = GUIEventAdapter::KEY_U;
        mKeyMap[Qt::Key_V] = GUIEventAdapter::KEY_V;
        mKeyMap[Qt::Key_W] = GUIEventAdapter::KEY_W;
        mKeyMap[Qt::Key_X] = GUIEventAdapter::KEY_X;
        mKeyMap[Qt::Key_Y] = GUIEventAdapter::KEY_Y;
        mKeyMap[Qt::Key_Z] = GUIEventAdapter::KEY_Z;

        mKeyMap[Qt::Key_0] = GUIEventAdapter::KEY_0;
        mKeyMap[Qt::Key_1] = GUIEventAdapter::KEY_1;
        mKeyMap[Qt::Key_2] = GUIEventAdapter::KEY_2;
        mKeyMap[Qt::Key_3] = GUIEventAdapter::KEY_3;
        mKeyMap[Qt::Key_4] = GUIEventAdapter::KEY_4;
        mKeyMap[Qt::Key_5] = GUIEventAdapter::KEY_5;
        mKeyMap[Qt::Key_6] = GUIEventAdapter::KEY_6;
        mKeyMap[Qt::Key_7] = GUIEventAdapter::KEY_7;
        mKeyMap[Qt::Key_8] = GUIEventAdapter::KEY_8;
        mKeyMap[Qt::Key_9] = GUIEventAdapter::KEY_9;
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
    , m_currentCursor(Qt::CrossCursor)
{
    useCursor(true);
}

QOpenGLWidget *QtGraphicsWindow::widget() const
{
    return m_glWidget;
}

bool QtGraphicsWindow::realizeImplementation()
{
    if (m_glWidget)
    {
        m_glWidget->makeCurrent();
        if (qApp)
        {
            qApp->sendPostedEvents();
            qApp->processEvents();
        }
    }
    else
    {
        std::cerr << "QtGraphicsWindow: realizing without GL widget" << std::endl;
    }
    return true;
}

bool QtGraphicsWindow::makeCurrentImplementation()
{
    if (m_glWidget)
        m_glWidget->makeCurrent();
    else
        std::cerr << "QtGraphicsWindow: making context current without GL widget" << std::endl;
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
    if (m_glWidget && m_glWidget->context())
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

    qt_set_sequence_auto_mnemonic(false);
}

QtOsgWidget::~QtOsgWidget()
{
}

osgViewer::GraphicsWindowEmbedded *QtOsgWidget::graphicsWindow() const
{
    return m_graphicsWindow.get();
}

void QtOsgWidget::focusWasLost()
{
    //std::cerr << "QtOsgWidget: focus lost: releasing all keys" << std::endl;
    for (auto &key: m_pressedKeys)
    {
        if (key.second)
            getEventQueue()->keyRelease(key.first);
        key.second = false;
    }
    m_pressedKeys.clear();

    m_modifierMask = 0;
    getEventQueue()->getCurrentEventState()->setModKeyMask(0);
}

void QtOsgWidget::focusOutEvent(QFocusEvent *event)
{
    focusWasLost();
    event->accept();
}

void QtOsgWidget::paintEvent(QPaintEvent *paintEvent)
{
    //opencover::VRViewer::instance()->requestRedraw();
}

void QtOsgWidget::initializeGL()
{
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
    unsigned int modkey = event->modifiers() & (Qt::ShiftModifier | Qt::ControlModifier | Qt::AltModifier | Qt::MetaModifier);
    int mask = 0;
#ifdef Q_OS_MACOS
    if (modkey & Qt::ControlModifier) mask |= osgGA::GUIEventAdapter::MODKEY_META;
    if (modkey & Qt::MetaModifier) mask |= osgGA::GUIEventAdapter::MODKEY_CTRL;
#else
    if (modkey & Qt::ControlModifier) mask |= osgGA::GUIEventAdapter::MODKEY_CTRL;
    if (modkey & Qt::MetaModifier) mask |= osgGA::GUIEventAdapter::MODKEY_META;
#endif
    if (modkey & Qt::ShiftModifier) mask |= osgGA::GUIEventAdapter::MODKEY_SHIFT;
    if (modkey & Qt::AltModifier) mask |= osgGA::GUIEventAdapter::MODKEY_ALT;
    m_modifierMask = mask;
    getEventQueue()->getCurrentEventState()->setModKeyMask(mask);
}

void QtOsgWidget::keyPressEvent(QKeyEvent *event)
{
    setKeyboardModifiers(event);
    int value = s_QtKeyboardMap.remapKey(event);
    getEventQueue()->keyPress(value);
    m_pressedKeys[value] = true;
}

void QtOsgWidget::keyReleaseEvent(QKeyEvent *event)
{
    setKeyboardModifiers(event);
#if 0
    if( event->isAutoRepeat() )
    {
        event->ignore();
    }
    else
#endif
    {
        int value = s_QtKeyboardMap.remapKey(event);
        getEventQueue()->keyRelease(value);
        m_pressedKeys[value] = false;
    }
}

void QtOsgWidget::mouseMoveEvent(QMouseEvent *event )
{
    setKeyboardModifiers(event);
    auto pr = devicePixelRatio();
    getEventQueue()->mouseMotion(event->x()*pr, event->y()*pr);
}

void QtOsgWidget::mousePressEvent(QMouseEvent *event)
{
    setKeyboardModifiers(event);
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
    setKeyboardModifiers(event);
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
    getEventQueue()->mouseButtonRelease(event->x()*pr, event->y()*pr, button);
}

void QtOsgWidget::wheelEvent(QWheelEvent *event)
{
    setKeyboardModifiers(event);
    event->accept();
    int delta = event->delta();

    osgGA::GUIEventAdapter::ScrollingMotion motion =
            delta>0 ? osgGA::GUIEventAdapter::SCROLL_UP : osgGA::GUIEventAdapter::SCROLL_DOWN;
    if (event->orientation() == Qt::Horizontal)
            motion = delta>0 ? osgGA::GUIEventAdapter::SCROLL_LEFT : osgGA::GUIEventAdapter::SCROLL_RIGHT;
    getEventQueue()->mouseScroll(motion);
}

osgGA::EventQueue* QtOsgWidget::getEventQueue() const
{
    return m_graphicsWindow->getEventQueue();
}

#include "moc_QtOsgWidget.cpp"
