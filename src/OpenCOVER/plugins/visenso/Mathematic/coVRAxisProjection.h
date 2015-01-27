/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2009 Visenso  **
 **                                                                        **
 ** Description: coVRAxisProjection                                        **
 **              Draws the way to the point                                **
 **                following the x, y and z axis                           **
 **                                                                        **
 ** Author: M. Theilacker                                                  **
 **                                                                        **
 ** History:                                                               **
 **     4.2009 initial version                                             **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#ifndef _coVRAXISPPRO_H
#define _coVRAXISPPRO_H

#include <osg/Vec3>
#include <osg/MatrixTransform>

// #include <osg/ref_ptr>

class coVRAxisProjection
{
public:
    // constructor destructor
    coVRAxisProjection(osg::Vec3 point, double dashRadius = 0.09, double dashLength = 0.8);
    ~coVRAxisProjection();

    // methods
    void setPoint(osg::Vec3 point);
    /// sets the drawables (in)visible
    void setVisible(bool visible);
    /// updates x, y, z projection
    void update(osg::Vec3 point);

private:
    // variables
    osg::ref_ptr<osg::MatrixTransform> node_;
    osg::Vec3 point_;
    double dashRadius_;
    double dashLength_;

    // methods
    void updateXProjection();
    void updateYProjection();
    void updateZProjection();
};

#endif
