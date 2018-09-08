// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include <QMouseEvent>

#include "vvlightinteractor.h"

#include <virvo/vvopengl.h>
#include <virvo/gl/util.h>
#include <virvo/private/project.h>

#include <iostream>

namespace gl = virvo::gl;
using virvo::mat4;
using virvo::recti;
using virvo::vec3;


vvLightInteractor::vvLightInteractor()
  : _lightingEnabled(true)
  , _mouseButton(Qt::NoButton)
{

}

void vvLightInteractor::render() const
{


// Qt 4 unfortunately overrides MAC_OS_X_VERSION_MIN_REQUIRED:
// http://comments.gmane.org/gmane.comp.lib.qt.user/10220
#if  defined(__APPLE__) //&& MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wdeprecated"

#endif


  // store GL state
  glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_LIGHTING_BIT | GL_CURRENT_BIT);

  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);

  float r = 3.0f;

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glTranslatef(pos_[0], pos_[1], pos_[2]);
  GLUquadricObj* quad = gluNewQuadric();
  if (_lightingEnabled)
  {
    glColor3f(1.0f, 1.0f, 0.0f);
  }
  else
  {
    glColor3f(0.2f, 0.2f, 0.0f);
  }
  gluSphere(quad, r, 10.0f, 10.0f);

  // axes
  if (hasFocus())
  {
    glBegin(GL_LINES);
      glColor3f(1.0f, 0.0f, 0.0f);
      glVertex3f(r, 0.0f, 0.0f);
      glVertex3f(5 * r, 0.0f, 0.0f);

      glColor3f(0.0f, 1.0f, 0.0f);
      glVertex3f(0.0f, r, 0.0f);
      glVertex3f(0.0f, 5 * r, 0.0f);

      glColor3f(0.0f, 0.0f, 1.0f);
      glVertex3f(0.0f, 0.0f, r);
      glVertex3f(0.0f, 0.0f, 5 * r);
    glEnd();

    GLUquadricObj* quadx = gluNewQuadric();
    glRotatef(90.0f, 0.0f, 1.0f, 0.0f);
    glTranslatef(0.0f, 0.0f, 5 * r);
    glColor3f(1.0f, 0.0f, 0.0f);
    gluCylinder(quadx, r / 2.0f, 0.0f, r * 2.0f, 10.0f, 10.0f);
    glTranslatef(0.0f, 0.0f, -(5 * r));
    glRotatef(-90.0f, 0.0f, 1.0f, 0.0f);

    GLUquadricObj* quady = gluNewQuadric();
    glRotatef(270.0f, 1.0f, 0.0f, 0.0f);
    glTranslatef(0.0f, 0.0f, 5 * r);
    glColor3f(0.0f, 1.0f, 0.0f);
    gluCylinder(quady, r / 2.0f, 0.0f, r * 2.0f, 10.0f, 10.0f);
    glTranslatef(0.0f, 0.0f, -(5 * r));
    glRotatef(-270.0f, 1.0f, 0.0f, 0.0f);

    GLUquadricObj* quadz = gluNewQuadric();
    glTranslatef(0.0f, 0.0f, 5 * r);
    glColor3f(0.0f, 0.0f, 1.0f);
    gluCylinder(quadz, r / 2.0f, 0.0f, r * 2.0f, 10.0f, 10.0f);
    glTranslatef(0.0f, 0.0f, -(5 * r));
  }

  glPopMatrix();

  glPopAttrib();

// Qt 4 unfortunately overrides MAC_OS_X_VERSION_MIN_REQUIRED:
// http://comments.gmane.org/gmane.comp.lib.qt.user/10220
#if defined(__APPLE__) //&& MAC_OS_X_VERSION_MIN_REQUIRED >= MAC_OS_X_VERSION_10_9

  #pragma GCC diagnostic pop

#endif

}

void vvLightInteractor::mouseMoveEvent(QMouseEvent* event)
{
  if (_mouseButton == Qt::LeftButton)
  {
    mat4  mv = gl::getModelviewMatrix();
    mat4  pr = gl::getProjectionMatrix();
    recti vp = gl::getViewport();

    vec3 obj;
    virvo::project(&obj, pos_, mv, pr, vp);
    vec3 win(event->x(), vp[3] - event->y(), obj[2]);
    virvo::unproject(&pos_, win, mv, pr, vp);
    emit lightPos(pos_);
  }
}

void vvLightInteractor::mousePressEvent(QMouseEvent* event)
{
  _mouseButton = event->button();
}

void vvLightInteractor::mouseReleaseEvent(QMouseEvent*)
{
  if (_mouseButton == Qt::LeftButton)
  {
    emit lightPos(pos_);
  }

  _mouseButton = Qt::NoButton;
}

void vvLightInteractor::setLightingEnabled(bool enabled)
{
  _lightingEnabled = enabled;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
