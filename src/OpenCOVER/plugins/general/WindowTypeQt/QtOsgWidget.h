#ifndef QT_OSG_WIDGET_H
#define QT_OSG_WIDGET_H

#include <QOpenGLWidget>
#include <QInputEvent>

#include <osg/ref_ptr>
#include <osgViewer/GraphicsWindow>

class QtOsgWidget: public QOpenGLWidget
{
    Q_OBJECT

public:
    QtOsgWidget(QWidget* parent=nullptr, Qt::WindowFlags f=0);
    virtual ~QtOsgWidget();

    osgViewer::GraphicsWindowEmbedded *graphicsWindow() const;

protected:
    virtual void paintEvent(QPaintEvent *paintEvent);
    virtual void paintGL();
    virtual void resizeGL(int width, int height);

    virtual void keyPressEvent(QKeyEvent *event);
    virtual void keyReleaseEvent(QKeyEvent *event);

    virtual void mouseMoveEvent(QMouseEvent *event);
    virtual void mousePressEvent(QMouseEvent *event);
    virtual void mouseReleaseEvent(QMouseEvent *event);
    virtual void wheelEvent(QWheelEvent *event);

private:
    void setKeyboardModifiers(QInputEvent *event);
    osgGA::EventQueue *getEventQueue() const;

    osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> m_graphicsWindow;
};

class QtGraphicsWindow: public osgViewer::GraphicsWindowEmbedded
{
public:
    QtGraphicsWindow(QOpenGLWidget *glWidget, int x, int y, int width, int height);

    QOpenGLWidget *widget() const;

    virtual bool makeCurrentImplementation() override;
    virtual bool releaseContextImplementation() override;
    virtual bool setWindowRectangleImplementation(int x, int y, int width, int height) override;
    virtual void swapBuffersImplementation() override;
    virtual void useCursor(bool cursorOn) override;
    virtual void setCursor(MouseCursor cursor) override;
    virtual bool setWindowDecorationImplementation(bool flag) override;

protected:
    QOpenGLWidget *m_glWidget;
    Qt::CursorShape m_currentCursor;
};


#endif
