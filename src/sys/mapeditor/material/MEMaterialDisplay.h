/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ME_GLWIDGET_H
#define ME_GLWIDGET_H

#include <QGLWidget>

class MEMaterialDisplay : public QGLWidget
{
    Q_OBJECT

public:
    MEMaterialDisplay(QWidget *parent = 0);
    ~MEMaterialDisplay();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;
    void setValues(const QVector<float> &data);
    void setSelected(bool);
    const QVector<float> getValues();

signals:
    void clicked();

private:
    bool selected;

    QVector<float> data;
    GLuint makeObject();
    GLuint object;

protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void mouseReleaseEvent(QMouseEvent *e);
};

#endif
