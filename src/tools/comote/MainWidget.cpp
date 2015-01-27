/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// MainWidget.cpp

#include <GL/glew.h>

#include "Debug.h"
#include "RR/RRServer.h"
#include "RR/RRFrame.h"
#include "Shader.h"

#include "MyTuioServer.h"
#include "MainWidget.h"

#include <QMouseEvent>
#include <QWheelEvent>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QString>

#ifndef GL_RGBA8
#define GL_RGBA8 0x8058
#endif
#ifndef GL_BGRA
#define GL_BGRA 0x80E1
#endif
#ifndef GL_UNSIGNED_INT_8_8_8_8_REV
#define GL_UNSIGNED_INT_8_8_8_8_REV 0x8367
#endif

// Texture format
static const int kInternalFormat = GL_RGBA8;
static const int kFormat = GL_BGRA;
static const int kType = GL_UNSIGNED_INT_8_8_8_8_REV;

static const unsigned int kRed = 0xC00040;
static const unsigned int kGreen = 0x80C040;
static const unsigned int kBlue = 0x4080C0;

static unsigned char Y(unsigned int rgb)
{
    const int r = (rgb >> 16) & 0xff;
    const int g = (rgb >> 8) & 0xff;
    const int b = rgb & 0xff;
    return (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

static unsigned char U(unsigned int rgb)
{
    const int r = (rgb >> 16) & 0xff;
    const int g = (rgb >> 8) & 0xff;
    const int b = rgb & 0xff;
    return (unsigned char)(128 - 0.169f * r + 0.331f * g + 0.5f * b);
}

static unsigned char V(unsigned int rgb)
{
    const int r = (rgb >> 16) & 0xff;
    const int g = (rgb >> 8) & 0xff;
    const int b = rgb & 0xff;
    return (unsigned char)(128 + 0.5f * r - 0.419f * g - 0.081f * b);
}

MainWidget::MainWidget(const QGLFormat &format, QWidget *parent)
    : QGLWidget(format, parent)
#if USE_TUIO
    , tuioServer(0)
#endif
    , rrServer(0)
    , shader(0)
    , texture(0)
    , textureW(0)
    , hSub(1)
    , textureH(0)
    , vSub(1)
    , handlingTouchEvents(false)
    , gpuColorConversion(false)
    , decompressor(NULL)
{
    yuvTextures[0] = yuvTextures[1] = yuvTextures[2] = 0;

    setAttribute(Qt::WA_AcceptTouchEvents);
    setAttribute(Qt::WA_StaticContents);
    setAttribute(Qt::WA_NoSystemBackground);

    setFocus();

    // Set a minimum window size to prevent a crash when width and/or height <= 0.
    setMinimumWidth(300);
    setMinimumHeight(200);

    setAutoBufferSwap(false); // Manually swap the buffers.
    setAutoFillBackground(false); // Don't repaint the background.
    setMouseTracking(true); // We need mouseMoveEvent's even if no button is pressed!

    connect(&fpsUpdateTimer, SIGNAL(timeout()), this, SLOT(updateFPS()));

    fpsUpdateTimer.setInterval(1000 /*ms*/);
    fpsUpdateTimer.start();
}

MainWidget::~MainWidget()
{
    fpsUpdateTimer.stop();

    disconnectFromHost();

    deleteBuffers();

    delete decompressor;
}

bool MainWidget::connectToHost(const QString &hostname, unsigned short tcpPort, unsigned short udpPort)
{
    ASSERT(rrServer == 0);
#if USE_TUIO
    ASSERT(tuioServer == 0);
#endif

    clearTexture(kBlue);

    emit signalMessage(Message_ServerConnecting);

#if USE_TUIO

    tuioServer = new MyTuioServer;
    if (!tuioServer->connectToHost(hostname.toStdString().c_str(), udpPort, 65536))
    {
        Log() << "WARNING!!! MainWidget::connectToHost: could not connect to TUIO client";

        delete tuioServer;
        tuioServer = 0;
    }

#else

    UNUSED(udpPort);
#endif

    rrServer = new RRServer(hostname, tcpPort, gpuColorConversion);

    connect(rrServer,
            SIGNAL(signalFrameComplete()),
            this,
            SLOT(frameComplete()),
            Qt::QueuedConnection);

    connect(rrServer,
            SIGNAL(signalFrameReceived(RRTileDecompressor::TileVector *)),
            this,
            SLOT(frameReceived(RRTileDecompressor::TileVector *)),
            Qt::QueuedConnection);

    connect(rrServer,
            SIGNAL(signalMessage(int /*RRServer::Message*/, QString, quint16)),
            this,
            SLOT(message(int /*RRServer::Message*/, QString, quint16)),
            Qt::QueuedConnection);

    rrServer->sendEvent(rrxevent(RREV_RESIZE, width(), height(), 0, 0));
    rrServer->start();

    return true;
}

bool MainWidget::disconnectFromHost()
{
    if (rrServer)
    {
        rrServer->stop();
        rrServer->wait();
    }

    delete rrServer;
    rrServer = 0;

#if USE_TUIO

    if (tuioServer)
    {
        tuioServer->disconnectFromHost();
    }

    delete tuioServer;
    tuioServer = 0;
#endif

    return true;
}

void MainWidget::initializeGL()
{
    glewInit();

    glClearColor(0.0f, 0.5f, 1.0f, 0.0f);

    glEnable(GL_TEXTURE_2D);

    glDisable(GL_DEPTH_TEST);

    gpuColorConversion = Shader::haveGLSL();

    decompressor = new RRTileDecompressor(gpuColorConversion);

    if (gpuColorConversion)
    {
        shader = new Shader;
        shader->loadFragmentSource(":/yuv2rgb.fsh");
        shader->link();
    }

    clearTexture(kGreen);

    //  Log() << "OpenGL version: " << (const char*)glGetString(GL_VERSION);
}

void MainWidget::resizeGL(int w, int h)
{
    glViewport(0, 0, w, h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 1.0, 0.0, -1.0, 1.0);
    //gluOrtho2D(0, w, h, 0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void MainWidget::paintGL()
{
    ASSERT(texture != 0);

    ///*double dt = */fpsCounter.registerFrame();

    //  glClear(GL_COLOR_BUFFER_BIT);

    if (rrServer && rrServer->isConnected())
    {
        RRFrame *frame = rrServer->getFrame();
        if (frame)
        {
            frame->lock();
            {
                // Resize the texture if required
                if (frame->getWidth() != textureW || frame->getHeight() != textureH)
                {
                    deleteBuffers();
                    createBuffers(frame->getWidth(), frame->getHeight(),
                                  frame->getHorizSubSampling(), frame->getVertSubSampling());
                }

                // Update texture data
                if (gpuColorConversion)
                {
                    for (int i = 0; i < 3; ++i)
                    {
                        glActiveTexture(GL_TEXTURE0 + i);
                        glBindTexture(GL_TEXTURE_2D, yuvTextures[i]);
                        glPixelStorei(GL_UNPACK_ROW_LENGTH, frame->getRowBytes() / (i == 0 ? 1 : frame->getHorizSubSampling()));
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                                        i == 0 ? textureW : (textureW + frame->getHorizSubSampling() - 1) / frame->getHorizSubSampling(),
                                        i == 0 ? textureH : (textureH + frame->getVertSubSampling() - 1) / frame->getVertSubSampling(),
                                        GL_LUMINANCE, GL_UNSIGNED_BYTE,
                                        i == 0 ? frame->yData(0, 0) : i == 1 ? frame->uData(0, 0) : frame->vData(0, 0));
                    }
                }
                else
                {
                    glBindTexture(GL_TEXTURE_2D, texture);
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, textureW, textureH, kFormat, kType, frame->getData());
                }
            }
            frame->unlock();
        }
    }

    if (gpuColorConversion)
    {
        shader->enable();
        for (int i = 0; i < 3; ++i)
        {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, yuvTextures[i]);
        }
        shader->setUniform1i("y_tex", 0);
        shader->setUniform1i("u_tex", 1);
        shader->setUniform1i("v_tex", 2);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, texture);
    }

    glBegin(GL_QUADS);
    if (gpuColorConversion)
    {
        float corrX = ((textureW + hSub - 1) / hSub * hSub - textureW) / (float)textureW;
        float corrY = ((textureH + vSub - 1) / vSub * vSub - textureH) / (float)textureH;
#define V(x, y)                                                       \
    glMultiTexCoord2f(GL_TEXTURE0, (x), (y));                         \
    glMultiTexCoord2f(GL_TEXTURE1, (x) - (x)*corrX, (y) - (y)*corrY); \
    glMultiTexCoord2f(GL_TEXTURE2, (x) - (x)*corrX, (y) - (y)*corrY); \
    glVertex2f((x), (y));
        V(0.f, 0.f);
        V(1.f, 0.f);
        V(1.f, 1.f);
        V(0.f, 1.f);
#undef V
    }
    else
    {
#define V(x, y)             \
    glTexCoord2f((x), (y)); \
    glVertex2f((x), (y));
        V(0.f, 0.f);
        V(1.f, 0.f);
        V(1.f, 1.f);
        V(0.f, 1.f);
#undef V
    }
    glEnd();

    if (gpuColorConversion)
        shader->disable();

    if (gpuColorConversion)
    {
        for (int i = 0; i < 3; ++i)
        {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glActiveTexture(GL_TEXTURE0);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

namespace
{

int mapQtModifiers(const QKeyEvent *ke)
{
    int ret = 0;
    const int m = ke->modifiers();
    if (m & Qt::ShiftModifier)
        ret |= RRMODKEY_SHIFT;
    if (m & Qt::ControlModifier)
        ret |= RRMODKEY_CTRL;
    if (m & Qt::AltModifier)
        ret |= RRMODKEY_ALT;
    if (m & Qt::MetaModifier)
        ret |= RRMODKEY_META;
    return ret;
}

int mapQtKeys(const QKeyEvent *ke)
{
    const int k = ke->key();

    if (k >= Qt::Key_Space && k <= Qt::Key_AsciiTilde)
    {
        if (k >= Qt::Key_A && k <= Qt::Key_Z && !(ke->modifiers() & Qt::ShiftModifier))
            return k + 0x20;
        else
            return k;
    }
    else if (k >= Qt::Key_F1 && k <= Qt::Key_F35)
    {
        return k - Qt::Key_F1 + 0xffbe;
    }

    return 0;
}

} // namespace

void MainWidget::keyPressEvent(QKeyEvent *ke)
{
    if (rrServer)
        rrServer->sendEvent(rrxevent(RREV_KEYPRESS, -1.f, -1.f,
                                     mapQtModifiers(ke),
                                     mapQtKeys(ke)));
}

void MainWidget::keyReleaseEvent(QKeyEvent *ke)
{
    if (rrServer)
        rrServer->sendEvent(rrxevent(RREV_KEYRELEASE, -1.f, -1.f,
                                     mapQtModifiers(ke),
                                     mapQtKeys(ke)));
}

namespace
{

int mapQtButton(int button)
{
    switch (button)
    {
    case Qt::LeftButton:
        return 1;
    case Qt::MidButton:
        return 2;
    case Qt::RightButton:
        return 3;
    }

    return 0;
}

int mapQtButtons(int buttons)
{
    int result = 0;

    if (buttons & Qt::LeftButton)
        result |= 1 << (1 - 1);
    if (buttons & Qt::MidButton)
        result |= 1 << (2 - 1);
    if (buttons & Qt::RightButton)
        result |= 1 << (3 - 1);

    return result;
}

} // namespace

QPointF MainWidget::mapMousePosition(const QPointF &pos)
{
    float x = pos.x() / (float)width();
    float y = 1.0f - pos.y() / (float)height();

    return QPointF(x, y);
}

QPointF MainWidget::mapTouchPosition(const QPointF &pos)
{
    float x = pos.x() / (float)width();
    float y = 1.0f - pos.y() / (float)height();

    return QPointF(x, y);
}

void MainWidget::mousePressEvent(QMouseEvent *mouseEvent)
{
    if (!handlingTouchEvents && rrServer /* && rrServer->isConnected() */)
    {
        QPointF pos = mapMousePosition(mouseEvent->pos());

        rrServer->sendEvent(rrxevent(RREV_BTNPRESS, pos.x(), pos.y(), mapQtButton(mouseEvent->button()), mapQtButtons(mouseEvent->buttons())));
    }
}

void MainWidget::mouseMoveEvent(QMouseEvent *mouseEvent)
{
    if (!handlingTouchEvents && rrServer /* && rrServer->isConnected() */)
    {
        QPointF pos = mapMousePosition(mouseEvent->pos());

        rrServer->sendEvent(rrxevent(RREV_MOTION, pos.x(), pos.y(), 0, mapQtButtons(mouseEvent->buttons())));
    }
}

void MainWidget::mouseReleaseEvent(QMouseEvent *mouseEvent)
{
    if (!handlingTouchEvents && rrServer /* && rrServer->isConnected() */)
    {
        QPointF pos = mapMousePosition(mouseEvent->pos());

        rrServer->sendEvent(rrxevent(RREV_BTNRELEASE, pos.x(), pos.y(), mapQtButton(mouseEvent->button()), mapQtButtons(mouseEvent->buttons())));
    }
}

void MainWidget::wheelEvent(QWheelEvent *wheelEvent)
{
    if (!handlingTouchEvents && rrServer /* && rrServer->isConnected() */)
    {
        QPointF pos = mapMousePosition(QPointF(wheelEvent->pos()));

        rrServer->sendEvent(rrxevent(RREV_WHEEL, pos.x(), pos.y(), wheelEvent->delta() / 120, 0));
    }
}

void MainWidget::resizeEvent(QResizeEvent *resizeEvent)
{
    QGLWidget::resizeEvent(resizeEvent);

    if (rrServer /* && rrServer->isConnected() */)
    {
        rrServer->sendEvent(rrxevent(RREV_RESIZE, resizeEvent->size().width(), resizeEvent->size().height(), 0, 0));
    }

    update();
}

void MainWidget::paintEvent(QPaintEvent *paintEvent)
{
    QGLWidget::paintEvent(paintEvent);

#if SHOW_TOUCH_POINTS

    const float penWidth = 3.0f;

    QPainter painter(this);

    for (StrokeMap::iterator it = strokeMap.begin(); it != strokeMap.end(); ++it)
    {
        painter.setPen(QPen(QBrush(it->second.color), penWidth));
        painter.drawPath(it->second.path);
    }
#endif

    swapBuffers();
}

void MainWidget::handleTouchPressed(const QTouchEvent::TouchPoint &p)
{
//  Log() << "Touch pressed";

#if SHOW_TOUCH_POINTS

    static const QColor kColors[] = {
        QColor(0, 255, 255), // primary == id 0 ???
        QColor(255, 0, 0),
        QColor(255, 255, 0),
        QColor(0, 0, 255),
        QColor(128, 0, 0),
        QColor(0, 128, 0),
        QColor(0, 0, 128),
        QColor(192, 0, 0),
        QColor(0, 192, 0),
        QColor(0, 0, 192),
    };

    ASSERT(strokeMap.find(p.id()) == strokeMap.end());
    ASSERT(p.id() < sizeof(kColors) / sizeof(kColors[0]));

    Stroke stroke;

    stroke.path.addEllipse(p.rect());
    stroke.path.moveTo(p.pos());
    stroke.color = p.isPrimary() ? QColor(255, 255, 255) : kColors[p.id()];
    stroke.id = p.id();

    strokeMap[p.id()] = stroke;
#endif

#if USE_TUIO
    if (tuioServer)
    {
        float x = (float)p.pos().x() / (float)width();
        float y = (float)p.pos().y() / (float)height();

        tuioServer->updateCursorList(p.id(), x, y, MyTuioServer::TPS_DOWN);
    }
#else
    if (rrServer && rrServer->isConnected())
    {
        QPointF pos = mapTouchPosition(p.pos());

        rrServer->sendEvent(rrxevent(RREV_TOUCHPRESS, pos.x(), pos.y(), p.id(), 0));
    }
#endif

    update();
}

void MainWidget::handleTouchMoved(const QTouchEvent::TouchPoint &p)
{
//  Log() << "Touch moved";

#if SHOW_TOUCH_POINTS

    ASSERT(strokeMap.find(p.id()) != strokeMap.end());

    strokeMap[p.id()].path.lineTo(p.pos());
#endif

#if USE_TUIO
    if (tuioServer)
    {
        float x = (float)p.pos().x() / (float)width();
        float y = (float)p.pos().y() / (float)height();

        tuioServer->updateCursorList(p.id(), x, y, MyTuioServer::TPS_MOVE);
    }
#else
    if (rrServer && rrServer->isConnected())
    {
        QPointF pos = mapTouchPosition(p.pos());

        rrServer->sendEvent(rrxevent(RREV_TOUCHMOVE, pos.x(), pos.y(), p.id(), 0));
    }
#endif

    update();
}

void MainWidget::handleTouchReleased(const QTouchEvent::TouchPoint &p)
{
//  Log() << "Touch released";

#if SHOW_TOUCH_POINTS

    StrokeMap::iterator pos = strokeMap.find(p.id());

    ASSERT(pos != strokeMap.end());

    strokeMap.erase(pos);
#endif

#if USE_TUIO
    if (tuioServer)
    {
        float x = (float)p.pos().x() / (float)width();
        float y = (float)p.pos().y() / (float)height();

        tuioServer->updateCursorList(p.id(), x, y, MyTuioServer::TPS_UP);
    }
#else
    if (rrServer && rrServer->isConnected())
    {
        QPointF pos = mapTouchPosition(p.pos());

        rrServer->sendEvent(rrxevent(RREV_TOUCHRELEASE, pos.x(), pos.y(), p.id(), 0));
    }
#endif

    update();
}

void MainWidget::touchEventBegin(QTouchEvent *touchEvent)
{
#if 1
    touchEventUpdate(touchEvent);
#else
    //
    // TouchBegin event is sent only once when the
    // first touch point is recognized
    //

    const QList<QTouchEvent::TouchPoint> &points = touchEvent->touchPoints();

    ASSERT(points.length() == 1);
    ASSERT(points.first().state() == Qt::TouchPointPressed);

    handleTouchPressed(points.first());
#endif

    handlingTouchEvents = true;
}

void MainWidget::touchEventUpdate(QTouchEvent *touchEvent)
{
    const QList<QTouchEvent::TouchPoint> &points = touchEvent->touchPoints();

    foreach (const QTouchEvent::TouchPoint &p, points)
    {
        switch (p.state())
        {
        case Qt::TouchPointPressed:
            handleTouchPressed(p);
            break;

        case Qt::TouchPointMoved:
            handleTouchMoved(p);
            break;

        case Qt::TouchPointStationary: // Ignored
            break;

        case Qt::TouchPointReleased:
            handleTouchReleased(p);
            break;

        default:
            break;
        }
    }
}

void MainWidget::touchEventEnd(QTouchEvent *touchEvent)
{
#if 1
    touchEventUpdate(touchEvent);
#else
    //
    // TouchEnd event is sent only once when the
    // last touch point has left
    //

    const QList<QTouchEvent::TouchPoint> &points = touchEvent->touchPoints();

    ASSERT(points.length() == 1);
    ASSERT(points.first().state() == Qt::TouchPointReleased);

    handleTouchReleased(points.first());
#endif

    handlingTouchEvents = false;
}

bool MainWidget::event(QEvent *event)
{
    QTouchEvent *touchEvent = dynamic_cast<QTouchEvent *>(event);

    if (touchEvent)
    {
#if USE_TUIO
        if (tuioServer)
            tuioServer->beginFrame();
#endif

        switch (touchEvent->type())
        {
        case QEvent::TouchBegin:
            touchEventBegin(touchEvent);
            break;

        case QEvent::TouchUpdate:
            touchEventUpdate(touchEvent);
            break;

        case QEvent::TouchEnd:
            touchEventEnd(touchEvent);
            break;

        default:
            break;
        }

#if USE_TUIO
        if (tuioServer)
            tuioServer->endFrame();
#endif

#if SHOW_TOUCH_POINTS
        update();
#endif

        touchEvent->accept();
        return true;
    }

    return QGLWidget::event(event);
}

void MainWidget::createBuffers(int w, int h, int hs, int vs, const void *data)
{
    ASSERT(texture == 0);

    textureW = w;
    textureH = h;
    hSub = hs;
    vSub = vs;

    glGenTextures(1, &texture);
    glGenTextures(3, yuvTextures);
    if (gpuColorConversion)
    {
        unsigned char yuv[3] = { 0, 0, 0 };
        if (data)
        {
            yuv[0] = Y(*(unsigned int *)data);
            yuv[1] = U(*(unsigned int *)data);
            yuv[2] = V(*(unsigned int *)data);
        }
        for (int i = 0; i < 3; ++i)
        {
            glActiveTexture(GL_TEXTURE0 + i);
            glBindTexture(GL_TEXTURE_2D, yuvTextures[i]);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, i > 0 ? GL_LINEAR : GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, i > 0 ? GL_LINEAR : GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE,
                         i == 0 ? textureW : (textureW + hs - 1) / hs,
                         i == 0 ? textureH : (textureH + vs - 1) / vs,
                         0, GL_LUMINANCE, GL_UNSIGNED_BYTE,
                         data ? &yuv[i] : NULL);
            glBindTexture(GL_TEXTURE_2D, 0);
        }
        glActiveTexture(GL_TEXTURE0);
    }
    else
    {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, kInternalFormat, textureW, textureH, 0, kFormat, kType, data);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void MainWidget::deleteBuffers()
{
    ASSERT(texture != 0);

    glDeleteTextures(1, &texture);
    glDeleteTextures(3, yuvTextures);

    texture = 0;
    yuvTextures[0] = yuvTextures[1] = yuvTextures[2] = 0;
    textureW = 0;
    textureH = 0;
}

void MainWidget::frameReceived(RRTileDecompressor::TileVector *tv)
{
    RRFrame *frame = rrServer->getFrame();
    if (!frame || !decompressor)
    {
        delete tv;
        return;
    }

    //fprintf(stderr, "frame: %d tiles\n", (int)(tv->size()));
    decompressor->run(tv, frame, true);
    delete tv;

    frameComplete();
}

void MainWidget::frameComplete()
{
    fpsCounter.registerFrame();

    update();
}

void MainWidget::updateFPS()
{
    fpsCounter.update();

    emit signalMessage(Message_FPS, textureW, textureH, (int)(fpsCounter.getFPS() + 0.5));
}

void MainWidget::message(int /*RRServer::Message*/ type, QString address, quint16 port)
{
    //  ASSERT( dynamic_cast<RRServer*>(sender()) ); // Handles messages from RRServer only!

    switch (type)
    {
    case RRServer::Message_Failed:
    case RRServer::Message_Disconnected:
    {
        if (type == RRServer::Message_Failed)
        {
            Log() << "RRServer: Failed to connect.";
        }
        else
        {
            Log() << "RRServer: Disconnected.";
        }

        disconnectFromHost();

        clearTexture(kGreen);

        if (type == RRServer::Message_Failed)
        {
            emit signalMessage(Message_ServerFailed);
        }
        else
        {
            emit signalMessage(Message_ServerDisconnected);
        }
    }
    break;

    case RRServer::Message_Connecting:
    {
        Log() << "RRServer: Connecting to " << address.toStdString() << ":" << port;

        emit signalMessage(Message_ServerConnecting);
    }
    break;

    case RRServer::Message_Connected:
    {
        Log() << "RRServer: Connected.";

        emit signalMessage(Message_ServerConnected);
    }
    break;
    }
}

void MainWidget::clearTexture(unsigned int color)
{
    if (texture)
        deleteBuffers();

    createBuffers(1, 1, 1, 1, &color);

    update();
}
