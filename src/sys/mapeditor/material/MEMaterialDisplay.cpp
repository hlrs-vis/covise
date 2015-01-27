/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include <QtGui>
#include <QtOpenGL>
#include <QSizePolicy>
#ifdef __APPLE__
#include <OpenGL/glu.h>
#else
#include <GL/glu.h>
#endif

#include "MEMaterialDisplay.h"

/*!
   \class MEMaterialDisplay
   \brief This class shows a material by an OpenGl widget
*/

MEMaterialDisplay::MEMaterialDisplay(QWidget *parent)
    : QGLWidget(parent)
    , selected(false)
{
    object = 0;
    /*QSizePolicy sizeP;
    sizeP.setHeightForWidth(true);
    setSizePolicy(sizeP);*/
}

//!
//! delete the OpenGL widget
//!
MEMaterialDisplay::~MEMaterialDisplay()
{
    makeCurrent();
    glDeleteLists(object, 1);
}

//!
//! set the minimum size of this widget
//!
QSize MEMaterialDisplay::minimumSizeHint() const
{
    return QSize(30, 30);
}

//!
//! set the preferred size of this widget
//!
QSize MEMaterialDisplay::sizeHint() const
{
    return QSize(30, 30);
}

//!
//! set the values defining a material
//!
void MEMaterialDisplay::setValues(const QVector<float> &values)
{
    data.clear();
    data = values;
    updateGL();
}

//!
//! generate a sphere object
//!
void MEMaterialDisplay::initializeGL()
{
    object = makeObject();
}

//!
//! draw the sphere with the current material values
//!
void MEMaterialDisplay::paintGL()
{
    // define material
    if (!data.isEmpty())
    {
        float trans = data[13];
        GLfloat mat_ambient[] = { data[0], data[1], data[2], trans };
        GLfloat mat_diffuse[] = { data[3], data[4], data[5], trans };
        GLfloat mat_specular[] = { data[6], data[7], data[8], trans };
        GLfloat mat_emission[] = { data[9], data[10], data[11], 1.0f };
        GLfloat mat_shininess[] = { data[12] * 128.0f };
        glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
        glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
        glMaterialfv(GL_FRONT, GL_EMISSION, mat_emission);
        glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
    }

    // set shading & lightning
    glShadeModel(GL_SMOOTH);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST);

    GLfloat pos[] = { 1.f, 2.f, 5.f, 0.f };
    glLightfv(GL_LIGHT0, GL_POSITION, pos);
    QColor c = palette().color(backgroundRole());
    glClearColor(c.red() / 256.0, c.green() / 256.0, c.blue() / 256.0, c.alpha() / 256.0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();
    glCallList(object);
}

//!
//! set viewport & projection
//!
void MEMaterialDisplay::resizeGL(int w, int h)
{
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (w > h)
    {
        glOrtho(-((float)w / (float)h), ((float)w / (float)h), -1, 1, -4, 15);
    }
    else
    {
        glOrtho(-1, 1, -((float)h / (float)w), (float)h / (float)w, -4, 15);
    }
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

//!
//! generate the sphere object
//!
GLuint MEMaterialDisplay::makeObject()
{
    GLuint list = glGenLists(1);
    glNewList(list, GL_COMPILE);

    GLUquadric *quadric = gluNewQuadric();
    gluSphere(quadric, 1.0f, 30, 30);

    glEndList();
    return list;
}

//!
//! get the current material values
//!
const QVector<float> MEMaterialDisplay::getValues()
{
    return data;
}

//!
//! draw the sphere with the current material values
//!
void MEMaterialDisplay::setSelected(bool state)
{
    selected = state;
}

//!
//! draw the sphere with the current material values
//!
void MEMaterialDisplay::mouseReleaseEvent(QMouseEvent *)
{
    selected = true;
    emit clicked();
}
