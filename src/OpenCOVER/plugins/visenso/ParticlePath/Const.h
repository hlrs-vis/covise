/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CONST_H
#define _CONST_H

#include <osg/Vec3>
#include <osg/Vec3d>
#include <osg/Vec4>

static const double GUI_SCALING_MASS = 1.660538921e-27;
static const double GUI_SCALING_CHARGE = 1.602176565e-19;
static const double GUI_SCALING_VELOCITY = 1000.0;
static const double GUI_SCALING_VOLTAGE = 1000.0;
static const double GUI_SCALING_ANGLE = 3.14159265359 / 180.0;
static const double GUI_SCALING_ELECTRIC_FIELD = 1000.0;
static const double GUI_SCALING_MAGNETIC_FIELD = 0.001;

static const float EPSILON = 0.001f;
static const osg::Vec3 EPSILON_CENTER = osg::Vec3(EPSILON, EPSILON, EPSILON);

static const float ANIMATION_PAUSE = 50.0f;
static const float ANIMATION_SPEED = 100.0f;

static const osg::Vec4 VELOCITY_ARROW_COLOR = osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f);
static const osg::Vec4 COMBINED_FORCE_ARROW_COLOR = osg::Vec4(1.0f, 0.5f, 0.0f, 1.0f);
static const osg::Vec4 ELECTRIC_FORCE_ARROW_COLOR = osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f);
static const osg::Vec4 MAGNETIC_FORCE_ARROW_COLOR = osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f);
static const osg::Vec4 PATH_COLOR = osg::Vec4(0.2f, 1.0f, 0.2f, 1.0f);
static const osg::Vec4 PREVIOUS_PATH_COLOR = osg::Vec4(0.4f, 0.5f, 0.4f, 1.0f);
static const osg::Vec4 TARGET_COLOR = osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f);
static const osg::Vec4 PARTICLE_COLOR = osg::Vec4(0.9f, 0.9f, 0.9f, 1.0f);
static const osg::Vec4 PARTICLE_COLOR_POSITIVE = osg::Vec4(1.0f, 0.7f, 0.7f, 1.0f);
static const osg::Vec4 PARTICLE_COLOR_NEGATIVE = osg::Vec4(0.7f, 0.7f, 1.0f, 1.0f);

static const float ARROW_SIZE = 0.01f;
static const float PARTICLE_SIZE = 0.02f;
static const float TARGET_SIZE = 0.075f;
static const float PATH_WIDTH = 2.0f;

static const double VELOCITY_ARROW_SCALING = 1.1e-6;
static const double FORCE_ARROW_SCALING = 0.7e14;
static const double ELECTRIC_FIELD_ARROW_SCALING = 1.1e-5;
static const double MAGNETIC_FIELD_ARROW_SCALING = 1.1;

static const osg::Vec3 TRACE_CENTER = osg::Vec3(0.0f, -0.25, 0.0f);
static const double TRACE_DELTA = 1e-8;
static const int TRACE_MAX_STEPS = 25000;
static const double TRACE_MAX_REGION = 1.0;

static const osg::Vec3d BASE_VECTOR_PARTICLE = osg::Vec3d(1.0, 0.0, 0.0);
static const osg::Vec3d BASE_VECTOR_PARTICLE_AXIS = osg::Vec3d(0.0, 0.0, -1.0);
static const osg::Vec3d BASE_VECTOR_ELECTRIC = osg::Vec3d(0.0, 0.0, 1.0);
static const osg::Vec3d BASE_VECTOR_MAGNETIC = osg::Vec3d(0.0, -1.0, 0.0);

/*
 * X (right)  : direction of particle (angle rotates towards Y)
 * Y (forward): - magnetic field
 * Z (up)     : electric field
 */

#endif
