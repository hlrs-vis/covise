/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// MainWidget.h

#ifndef MAIN_WIDGET_H
#define MAIN_WIDGET_H

#define SHOW_TOUCH_POINTS 0

#include <QGLWidget>
#include <QTimer>
#include <QTouchEvent>
#include <QString>
#include <QPoint>
#include <QMenu>

#if SHOW_TOUCH_POINTS
#include <map>
#endif

#include "FPSCounter.h"
#include "RR/RRDecompressor.h"

#if USE_TUIO
class MyTuioServer;
#endif
class RRServer;
class Shader;

class MainWidget : public QGLWidget
{
    Q_OBJECT

public:
    enum Message
    {
        Message_Quit,
        Message_ToggleFullscreen,
        Message_FPS,
        Message_ServerDisconnected,
        Message_ServerConnecting,
        Message_ServerConnected,
        Message_ServerFailed,
    };

    MainWidget(const QGLFormat &format, QWidget *parent = 0);

    virtual ~MainWidget();

    // Returns true on success, false on failure.
    // A message() slot is called to notify about the connection status.
    // udpPort is only used if the TUIO protocol is used to transmit touch events.
    bool connectToHost(const QString &hostname, unsigned short tcpPort, unsigned short udpPort);

    // Manually disconnect from TUIO client and RRServer
    bool disconnectFromHost();

signals:
    void signalMessage(int /*Message*/ type, int x = 0, int y = 0, int z = 0, int w = 0);

protected:
    //
    // QGLWidget overrides
    //
    virtual void initializeGL();
    virtual void resizeGL(int w, int h);
    virtual void paintGL();

    virtual void keyPressEvent(QKeyEvent *event);
    virtual void keyReleaseEvent(QKeyEvent *event);

    virtual void mousePressEvent(QMouseEvent *mouseEvent);
    virtual void mouseMoveEvent(QMouseEvent *mouseEvent);
    virtual void mouseReleaseEvent(QMouseEvent *mouseEvent);

    virtual void wheelEvent(QWheelEvent *wheelEvent);

    virtual void resizeEvent(QResizeEvent *resizeEvent);

    virtual void paintEvent(QPaintEvent *paintEvent);

    virtual bool event(QEvent *event);

private:
    void touchEventBegin(QTouchEvent *touchEvent);
    void touchEventUpdate(QTouchEvent *touchEvent);
    void touchEventEnd(QTouchEvent *touchEvent);

    void handleTouchPressed(const QTouchEvent::TouchPoint &p);
    void handleTouchMoved(const QTouchEvent::TouchPoint &p);
    void handleTouchReleased(const QTouchEvent::TouchPoint &p);

    void createBuffers(int w, int h, int hs, int vs, const void *data = 0);
    void deleteBuffers();

    void clearTexture(unsigned int color);

    // Maps widget coordinates to [0,1]^2
    QPointF mapMousePosition(const QPointF &pos);
    QPointF mapTouchPosition(const QPointF &pos);

private slots:
    // Signaled by RRServer::frameComplete();
    // New frame ready for display
    void frameComplete();

    // Signaled by RRServer::frameReceived();
    // New frame ready to be decompressed
    void frameReceived(RRTileDecompressor::TileVector *tv);

    // Signaled by fpsUpdateTimer;
    // Time to update the titlebar of the main window
    void updateFPS();

    void message(int /*RRServer::Message*/ type, QString address, quint16 port);

private:
    FPSCounter fpsCounter;

    QTimer fpsUpdateTimer;

#if USE_TUIO
    MyTuioServer *tuioServer;
#endif

    RRServer *rrServer;
    Shader *shader;

    GLuint texture; // Framebuffer
    GLuint yuvTextures[3];
    int textureW;
    int textureH;
    int hSub, vSub;

#if SHOW_TOUCH_POINTS
    struct Stroke
    {
        QPainterPath path;
        QColor color;
        unsigned int id;
    };

    typedef std::map<int, Stroke> StrokeMap;

    StrokeMap strokeMap;
#endif

    bool handlingTouchEvents;
    bool gpuColorConversion;
    RRTileDecompressor *decompressor;
};
#endif
