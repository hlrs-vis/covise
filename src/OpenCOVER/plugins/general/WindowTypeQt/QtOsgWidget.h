#ifndef QT_OSG_WIDGET_H
#define QT_OSG_WIDGET_H

#include <QOpenGLWidget>
#include <QInputEvent>

#include <osg/ref_ptr>
#include <osgViewer/GraphicsWindow>

#include <map>

class QtOsgWidget: public QOpenGLWidget
{
    Q_OBJECT

public:
    QtOsgWidget(QWidget* parent=nullptr, Qt::WindowFlags f=0);
    virtual ~QtOsgWidget();

    osgViewer::GraphicsWindowEmbedded *graphicsWindow() const;

public slots:
    void focusWasLost();

protected:
    virtual void focusOutEvent(QFocusEvent *event) override;
    virtual void paintEvent(QPaintEvent *paintEvent) override;
    virtual void initializeGL() override;
    virtual void paintGL() override;
    virtual void resizeGL(int width, int height) override;

    virtual void keyPressEvent(QKeyEvent *event) override;
    virtual void keyReleaseEvent(QKeyEvent *event) override;

    virtual void mouseMoveEvent(QMouseEvent *event) override;
    virtual void mousePressEvent(QMouseEvent *event) override;
    virtual void mouseReleaseEvent(QMouseEvent *event) override;
    virtual void wheelEvent(QWheelEvent *event) override;

private:
    void setKeyboardModifiers(QInputEvent *event);
    osgGA::EventQueue *getEventQueue() const;

    osg::ref_ptr<osgViewer::GraphicsWindowEmbedded> m_graphicsWindow;
    std::map<int, bool> m_pressedKeys;
    int m_modifierMask = 0;
};

class QtGraphicsWindow: public osgViewer::GraphicsWindowEmbedded
{
public:
    QtGraphicsWindow(QOpenGLWidget *glWidget, int x, int y, int width, int height);

    QOpenGLWidget *widget() const;

    virtual bool realizeImplementation() override;
    virtual bool makeCurrentImplementation() override;
    virtual bool releaseContextImplementation() override;
    virtual bool setWindowRectangleImplementation(int x, int y, int width, int height) override;
    virtual void swapBuffersImplementation() override;
    virtual void useCursor(bool cursorOn) override;
    virtual void setCursor(MouseCursor cursor) override;
    virtual bool setWindowDecorationImplementation(bool flag) override;

protected:
    QOpenGLWidget *m_glWidget = nullptr;
    Qt::CursorShape m_currentCursor;
};


#endif
