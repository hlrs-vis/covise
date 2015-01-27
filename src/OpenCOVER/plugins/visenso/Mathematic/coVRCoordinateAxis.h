/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRCoordinateAxis                                        **
 **              Draws the three coordinates axis                          **
**                 only positiv or both directions                         **
 **                with colors red (x), green (y), blue (z)                **
 **                and ticks per unit length                               **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _COVRCOORDAXIS_H
#define _COVRCOORDAXIS_H

#include <osg/Vec3>
#include <osg/MatrixTransform>
#include <osg/StateSet>

#include <PluginUtil/coArrow.h>

class coVRCoordinateAxis
{
public:
    // constructor destructor
    coVRCoordinateAxis(double axisRadius = 0.07, double axisLength = 10.0, bool negAndPos = false, bool showTicks = true);
    ~coVRCoordinateAxis();

    // methods
    /// sets the drawables (in)visible
    void setVisible(bool visible);

private:
    // variables
    osg::ref_ptr<osg::MatrixTransform> axisNode_;
    osg::ref_ptr<osg::MatrixTransform> xTransform_;
    osg::ref_ptr<osg::MatrixTransform> yTransform_;
    osg::ref_ptr<osg::MatrixTransform> zTransform_;
    osg::ref_ptr<opencover::coArrow> xAxis_;
    osg::ref_ptr<opencover::coArrow> yAxis_;
    osg::ref_ptr<opencover::coArrow> zAxis_;
    double axisRadius_;
    double axisLength_;
    double tickRadius_;
    double tickLength_;
    double negAndPos_;

    // methods
    /// x y z axis
    void makeAxis();
    /// cylindrical ticks per unit (not at origin)
    void makeTicks();
    osg::StateSet *makeRedStateSet();
    osg::StateSet *makeGreenStateSet();
    osg::StateSet *makeBlueStateSet();
};

#endif
