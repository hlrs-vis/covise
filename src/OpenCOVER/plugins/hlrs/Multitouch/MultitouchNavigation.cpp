/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2012 HLRS  **
**                                                                          **
** Description: Multitouch Navigation										**
**                                                                          **
**                                                                          **
** Author:																	**
**         Jens Dehlke														**
**                                                                          **
** History:  								                                **
** Feb-13  v1.1																** 
** Sep-12  v1.0	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

// Note:
// The code to transform screen position to Xform position, is based on:
// http://forum.openscenegraph.org/viewtopic.php?t=6168
// What you get is actually a line that goes from the near plane
// to the far plane, since you're transforming 2D coordinates into 3D it
// means that a point in 2D becomes a line in 3D.
// To transform a 3D position into 2D window coordinates you would just do
// the opposite of that, using the same MVPW.
// Just make sure the posIn3D is in world space and not in some object's
// local coordinate system.

#include "MultitouchNavigation.h"

MultitouchNavigation::MultitouchNavigation()
{
    cout << "MultitouchNavigation::MultitouchNavigation" << endl;
    _counter = 0;
}

MultitouchNavigation::~MultitouchNavigation()
{
}

void MultitouchNavigation::rotateXY(TouchContact c)
{
    // Concept based on http://viewport3d.com/trackball.htm

    osg::Vec2d currentPosition2D(c.x, c.y);

    // Translate 0,0 to the center
    double x = currentPosition2D.x() - cover->frontWindowHorizontalSize / 2;
    double y = currentPosition2D.y() - cover->frontWindowVerticalSize / 2;

    // set bounds of unit sphere
    /*
	// set radius depending on finger
	double currRadius = (currentPosition2D - osg::Vec2d(cover->frontWindowHorizontalSize/2,cover->frontWindowVerticalSize/2)).length();
	if(_counter == 0)
	{
		// set to position of initial contact
		_radius = currRadius;
		// create a minimum sphere size, preventing too fast rotation
		double minRadius = osg::Vec2d(cover->frontWindowHorizontalSize*1./8., cover->frontWindowVerticalSize*1/8.).length();
		if(_radius < minRadius)
		{
			_radius = minRadius;
		}
	}
	else if(currRadius > _radius && currRadius < _previousValue)
	{
		_radius = _previousValue;
	}
	_previousValue = currRadius;

	// normalize	
	x /= _radius;
	y /= _radius;
	*/

    /*
	// set radius to screen diagonal
	if(_counter == 0)
	{
		_radius = sqrt(float((cover->frontWindowVerticalSize*cover->frontWindowVerticalSize)+(cover->frontWindowHorizontalSize*cover->frontWindowHorizontalSize))) / 2;
	}

	// normalize	
	x /= _radius;
	y /= _radius;
	*/

    x /= cover->frontWindowHorizontalSize / 2;
    y /= cover->frontWindowVerticalSize / 2;

    //get z position on unit sphere
    double z2 = 1 - x * x - y * y;
    double z = z2 > 0 ? sqrt(z2) : 0;
    osg::Vec3d currentPosition3D(x, y, z);

    if (_counter > 0)
    {
        // calculate angle between current and previous contact
        double angle = angleBetween3DVectors(_previousVector3D, currentPosition3D);
        // calculate rotation axis and rotate according to viewer
        osg::Vec3d axis = _previousVector3D ^ currentPosition3D;
        osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
        osg::Matrixd M;
        M.makeRotate(osg::Matrixd::inverse(cam->getViewMatrix()).getRotate());
        axis = axis * M;
        //apply transformation
        osg::Quat delta = osg::Quat(angle, axis);
        osg::Matrixd temp;
        temp.makeRotate(delta);
        cover->setXformMat(cover->getXformMat() * temp);
    }

    _previousVector3D = currentPosition3D;
    _counter++;
}

double MultitouchNavigation::angleBetween3DVectors(osg::Vec3 v1, osg::Vec3 v2)
{
    // http://codered.sat.qc.ca/redmine/projects/spinframework/repository/revisions/b6245189c19a7c6ba4fdb126940321c41c44e228/raw/src/spin/osgUtil.cpp

    // normalize vectors (note: this must be done alone, not within any vector arithmetic. why?!)
    v1.normalize();
    v2.normalize();

    // Get the dot product of the vectors
    double dotProduct = v1 * v2;

    // for acos, the value has to be between -1.0 and 1.0, but due to numerical imprecisions it sometimes comes outside this range
    if (dotProduct > 1.0)
        dotProduct = 1.0;
    if (dotProduct < -1.0)
        dotProduct = -1.0;

    // Get the angle in radians between the 2 vectors (should this be -acos ? ie, negative?)
    double angle = acos(dotProduct);

    // Here we make sure that the angle is not a -1.#IND0000000 number, which means indefinite
    if (isnan(angle)) //__isnand(x)
        return 0;

    // Return the angle in radians
    return (angle);
}

void MultitouchNavigation::moveXY(TouchContact c)
{
    // calculate ModelView - Projection - Window Transformation
    osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
    osg::Matrix MVPW(cam->getViewMatrix() * cam->getProjectionMatrix() * cam->getViewport()->computeWindowMatrix());
    osg::Matrixd inverseMVPW = osg::Matrixd::inverse(MVPW);

    // determine z plane of Xform in screen coordinates
    osg::Vec3d XformTranslation2D = cover->getXformMat().getTrans() * MVPW;
    // determine center of both fingers
    osg::Vec3d currentVector3D(c.x, c.y, XformTranslation2D.z());

    if (_counter > 0)
    {
        osg::Matrixd temp;
        temp.makeTranslate(currentVector3D * inverseMVPW - _previousVector3D * inverseMVPW);
        cover->setXformMat(cover->getXformMat() * temp);
    }

    _previousVector3D = currentVector3D;
    _counter++;
}

void MultitouchNavigation::continuousMoveXY(TouchContact c)
{
    // calculate ModelView - Projection - Window Transformation
    osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
    osg::Matrix MVPW(cam->getViewMatrix() * cam->getProjectionMatrix() * cam->getViewport()->computeWindowMatrix());
    osg::Matrixd inverseMVPW = osg::Matrixd::inverse(MVPW);

    // determine z plane of Xform in screen coordinates
    osg::Vec3d XformTranslation2D = cover->getXformMat().getTrans() * MVPW;
    // determine center of both fingers
    osg::Vec3d currentVector3D(c.x, c.y, XformTranslation2D.z());

    if (_counter > 0)
    {
        double xDistance, yDistance, xValue, yValue;
        int xSign, ySign;
        xSign = ySign = 1;
        xDistance = currentVector3D.x() - _previousVector3D.x();
        xValue = sqrt(xDistance * xDistance);
        yDistance = currentVector3D.y() - _previousVector3D.y();
        yValue = sqrt(yDistance * yDistance);
        if (xDistance != 0.)
        {
            xSign = xDistance / xValue;
        }
        if (yDistance != 0.)
        {
            ySign = yDistance / yValue;
        }
        osg::Vec3d trans(xSign * exp(xValue / 50.), ySign * exp(yValue / 50.), 0.);
        osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
        osg::Matrixd M;
        M.makeRotate(osg::Matrixd::inverse(cam->getViewMatrix()).getRotate());
        trans = trans * M;
        osg::Matrixd temp;
        temp.makeTranslate(trans);

        cover->setXformMat(cover->getXformMat() * temp);
    }

    if (_counter == 0)
        _previousVector3D = currentVector3D;
    _counter++;
}

void MultitouchNavigation::walkXY(TouchContact c)
{
    osg::Vec3 velDir;
    osg::Matrix dcs_mat;
    dcs_mat = cover->getXformMat();
    osg::Matrix tmp;
    osg::Matrix tmp2;
    float driveSpeed = coVRNavigationManager::instance()->getDriveSpeed();
    float currentVelocity;

    if (_counter == 0)
    {
        _initial = osg::Vec3d(c.x, c.y, 0.);
    }
    float angle = (c.x - _initial.x()) / 300;
    tmp = osg::Matrix::rotate(angle * M_PI / 180, 0.0, 0.0, 1.0);
    osg::Vec3 viewerPos = cover->getViewerMat().getTrans();
    tmp2.makeTranslate(viewerPos);
    tmp.postMult(tmp2);
    tmp2.makeTranslate(-viewerPos);
    tmp.preMult(tmp2);

    dcs_mat.postMult(tmp);
    currentVelocity = (c.y - _initial.y()) * driveSpeed * -0.5;
    velDir = osg::Vec3(0.0 * currentVelocity, 1.0 * currentVelocity, 0.0 * currentVelocity);
    osg::Matrix M;
    M.makeRotate(osg::PI_2, osg::X_AXIS);
    velDir = velDir * M;
    osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
    osg::Matrixd M2;
    M2.makeRotate(osg::Matrixd::inverse(cam->getViewMatrix()).getRotate());
    velDir = velDir * M2;
    tmp.makeTranslate(velDir);
    dcs_mat.postMult(tmp);
    cover->setXformMat(dcs_mat);

    _counter++;
}

void MultitouchNavigation::scaleXYZ(std::list<TouchContact> &contacts)
{
    // convert contact coordinates to current vectors
    osg::Vec2d currentPosition2DFinger1(contacts.front().x, cover->frontWindowVerticalSize - contacts.front().y);
    osg::Vec2d currentPosition2DFinger2(contacts.back().x, cover->frontWindowVerticalSize - contacts.back().y);

    // calculate center of line between finger1 and finger2
    osg::Vec2d scaleCenter = (currentPosition2DFinger1 + currentPosition2DFinger2) / 2.;
    // calculate ModelView - Projection - Window Transformation
    osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
    osg::Matrix MVPW(cam->getViewMatrix() * cam->getProjectionMatrix() * cam->getViewport()->computeWindowMatrix());
    osg::Matrixd inverseMVPW = osg::Matrixd::inverse(MVPW);
    // determine z-plane of Xform in screen coordinates
    osg::Vec3d XformTranslation2D = cover->getXformMat().getTrans() * MVPW;
    // scaleCenter in Xform coordinates
    osg::Vec3d currentVector3D(scaleCenter.x(), scaleCenter.y(), XformTranslation2D.z());
    currentVector3D = currentVector3D * inverseMVPW;

    // get distance between fingers
    double currentDistance = osg::Vec2d(currentPosition2DFinger1 - currentPosition2DFinger2).length();

    // divide by previous difference if != 0
    if (_counter > 0 && _previousValue != 0.)
    {
        // create copy of XformMat for calculation
        osg::Matrixd Xform = cover->getXformMat();
        // translate coordinate system to center of line
        Xform.postMultTranslate(-currentVector3D);
        // scale
        double scaleFactor = currentDistance / _previousValue;
        Xform.postMultScale(osg::Vec3d(scaleFactor, scaleFactor, scaleFactor));
        // translate back to origin
        Xform.postMultTranslate(currentVector3D);
        // set XformMat to copy
        cover->setXformMat(Xform);
    }
    _previousValue = currentDistance;
    _counter++;
}

void MultitouchNavigation::continuousScaleXYZ(std::list<TouchContact> &contacts)
{
    // convert contact coordinates to current vectors
    osg::Vec2d currentPosition2DFinger1(contacts.front().x, cover->frontWindowVerticalSize - contacts.front().y);
    osg::Vec2d currentPosition2DFinger2(contacts.back().x, cover->frontWindowVerticalSize - contacts.back().y);

    // calculate center of line between finger1 and finger2
    osg::Vec2d scaleCenter = (currentPosition2DFinger1 + currentPosition2DFinger2) / 2.;
    // calculate ModelView - Projection - Window Transformation
    osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
    osg::Matrix MVPW(cam->getViewMatrix() * cam->getProjectionMatrix() * cam->getViewport()->computeWindowMatrix());
    osg::Matrixd inverseMVPW = osg::Matrixd::inverse(MVPW);
    // determine z-plane of Xform in screen coordinates
    osg::Vec3d XformTranslation2D = cover->getXformMat().getTrans() * MVPW;
    // scaleCenter in Xform coordinates
    osg::Vec3d currentVector3D(scaleCenter.x(), scaleCenter.y(), XformTranslation2D.z());
    currentVector3D = currentVector3D * inverseMVPW;

    // get length of difference
    double currentDistance = osg::Vec2d(currentPosition2DFinger1 - currentPosition2DFinger2).length();
    // set _initialValue
    if (_counter == 0)
        _initialValue = currentDistance;

    // divide by _initialValue if != 0
    if (_counter > 0 && _initialValue != 0.)
    {
        // create copy of XformMat for calculation
        osg::Matrixd Xform = cover->getXformMat();
        // translate coordinate system to center of line
        Xform.postMultTranslate(-currentVector3D);
        // scale
        double scaleFactor = currentDistance / _initialValue;
        if (scaleFactor >= 1.)
        {
            scaleFactor--;
            scaleFactor *= 0.15;
            scaleFactor++;
        }
        else
        {
            scaleFactor = 1 - scaleFactor;
            scaleFactor *= 0.15;
            scaleFactor = 1 - scaleFactor;
        }
        Xform.postMultScale(osg::Vec3d(scaleFactor, scaleFactor, scaleFactor));
        // translate back to origin
        Xform.postMultTranslate(currentVector3D);
        // set XformMat to copy
        cover->setXformMat(Xform);
        //cover->setScale(cover->getScale() * scaleFactor);
    }
    _counter++;
}

void MultitouchNavigation::rotateZ(std::list<TouchContact> &contacts)
{
    // get current screen position of finger
    osg::Vec3d curr3DVec1(contacts.front().x, cover->frontWindowVerticalSize - contacts.front().y, 0.);
    osg::Vec3d curr3DVec2(contacts.back().x, cover->frontWindowVerticalSize - contacts.back().y, 0.);

    if (_counter > 0)
    {
        // figure out rotation
        osg::Vec3d lineCurrCurr = osg::Vec3d(curr3DVec1.x(), curr3DVec1.y(), 1.0) ^ osg::Vec3d(curr3DVec2.x(), curr3DVec2.y(), 1.0);
        osg::Vec3d linePrevPrev = osg::Vec3d(_prev3DVec1.x(), _prev3DVec1.y(), 1.0) ^ osg::Vec3d(_prev3DVec2.x(), _prev3DVec2.y(), 1.0);
        osg::Vec3d interception = lineCurrCurr ^ linePrevPrev;
        if (interception.z() != 0.)
        {
            double x = interception.x() / interception.z();
            double y = interception.y() / interception.z();

            // calculate ModelView - Projection - Window Transformation
            osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
            osg::Matrix MVPW(cam->getViewMatrix() * cam->getProjectionMatrix() * cam->getViewport()->computeWindowMatrix());
            osg::Matrixd inverseMVPW = osg::Matrixd::inverse(MVPW);
            // determine z-plane of Xform in screen coordinates
            osg::Vec3d XformTranslation2D = cover->getXformMat().getTrans() * MVPW;
            // rotation center in Xform coordinates
            osg::Vec3d currentVector3D(x, y, XformTranslation2D.z());
            currentVector3D = currentVector3D * inverseMVPW;

            // calculate angle & axis
            double angle = angleBetween3DVectors((_prev3DVec1 - _prev3DVec2), (curr3DVec1 - curr3DVec2));
            osg::Vec3d axis = cover->getViewerMat().getTrans() - currentVector3D;
            osg::Vec3d sign = (_prev3DVec1 - _prev3DVec2) ^ (curr3DVec1 - curr3DVec2);
            sign.normalize();
            axis.x() = axis.x() * sign.z();
            axis.y() = axis.y() * sign.z();
            axis.z() = axis.z() * sign.z();
            osg::Quat delta = osg::Quat(angle, axis);

            // create copy of XformMat for calculation
            osg::Matrixd Xform = cover->getXformMat();
            // translate coordinate system to center of line
            Xform.postMultTranslate(-currentVector3D);
            // rotate
            Xform.postMultRotate(delta);
            // translate back to origin
            Xform.postMultTranslate(currentVector3D);
            // set XformMat to copy
            cover->setXformMat(Xform);
        }
    }

    _prev3DVec1 = curr3DVec1;
    _prev3DVec2 = curr3DVec2;
    _counter++;
}

void MultitouchNavigation::moveZ(TouchContact c)
{
    // calculate ModelView - Projection - Window Matrix to transform screen position to world position:
    osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
    osg::Matrix MVPW(cam->getViewMatrix() * cam->getProjectionMatrix() * cam->getViewport()->computeWindowMatrix());
    osg::Matrixd inverseMVPW = osg::Matrixd::inverse(MVPW);

    // determine z plane of Xform in screen coordinates
    if (_counter == 0)
        _initial = cover->getXformMat().getTrans() * MVPW;
    // transform y-center to 3D world coordinate system (x chosen randomly)
    osg::Vec3d currentVector3D(cover->frontWindowHorizontalSize / 2, c.y, _initial.z());

    if (_counter > 0)
    {
        //rotate y to z and apply transformation
        osg::Vec3d yMov = _previousVector3D * inverseMVPW - currentVector3D * inverseMVPW;

        osg::Matrixd trans, M;
        osg::Vec3d viewerAxis = cover->getViewerMat().getTrans();
        double angle = angleBetween3DVectors(yMov, viewerAxis);
        osg::Vec3d axis = yMov ^ viewerAxis;
        axis.normalize();
        M.makeRotate(angle, axis);

        yMov.x() = sqrt(yMov.x() * yMov.x());
        yMov.y() = sqrt(yMov.y() * yMov.y());
        yMov.z() = sqrt(yMov.z() * yMov.z());
        yMov = yMov * M;
        trans.makeTranslate(yMov);
        cover->setXformMat(cover->getXformMat() * trans);
    }
    _previousVector3D = currentVector3D;
    _counter++;
}

void MultitouchNavigation::continuousMoveZ(TouchContact c)
{

    // calculate ModelView - Projection - Window Matrix to transform screen position to world position:
    osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
    osg::Matrix MVPW(cam->getViewMatrix() * cam->getProjectionMatrix() * cam->getViewport()->computeWindowMatrix());
    osg::Matrixd inverseMVPW = osg::Matrixd::inverse(MVPW);

    // determine z plane of Xform in screen coordinates
    if (_counter == 0)
    {
        _initial = cover->getXformMat().getTrans() * MVPW;
        _initial.y() = c.y;
    }
    // transform y-center to 3D world coordinate system (x chosen randomly)
    osg::Vec3d currentVector3D(cover->frontWindowHorizontalSize / 2, c.y, _initial.z());

    if (_counter > 0)
    {
        //rotate y to z and apply transformation
        osg::Matrixd trans, M;
        osg::Vec3d yMov = osg::Vec3d(cover->frontWindowHorizontalSize / 2, _initial.y(), _initial.z()) * inverseMVPW - currentVector3D * inverseMVPW;

        osg::Vec3d viewerAxis = cover->getViewerMat().getTrans();
        double angle = angleBetween3DVectors(yMov, viewerAxis);
        osg::Vec3d axis = yMov ^ viewerAxis;
        axis.normalize();
        M.makeRotate(angle, axis);

        yMov.x() = sqrt(yMov.x() * yMov.x());
        yMov.y() = sqrt(yMov.y() * yMov.y());
        yMov.z() = sqrt(yMov.z() * yMov.z());
        yMov = yMov * M;
        trans.makeTranslate(yMov);
        cover->setXformMat(cover->getXformMat() * trans);
    }
    _previousVector3D = currentVector3D;
    _counter++;
}

void MultitouchNavigation::walkZ(TouchContact c)
{
    osg::Vec3 velDir;
    osg::Matrix dcs_mat;
    dcs_mat = cover->getXformMat();
    osg::Matrix tmp;
    osg::Matrix tmp2;
    float driveSpeed = coVRNavigationManager::instance()->getDriveSpeed();
    float currentVelocity;

    if (_counter == 0)
    {
        _initial = osg::Vec3d(c.x, c.y, 0.);
    }
    float angle = (c.x - _initial.x()) / 300;
    tmp = osg::Matrix::rotate(angle * M_PI / 180, 0.0, 0.0, 1.0);
    osg::Vec3 viewerPos = cover->getViewerMat().getTrans();
    tmp2.makeTranslate(viewerPos);
    tmp.postMult(tmp2);
    tmp2.makeTranslate(-viewerPos);
    tmp.preMult(tmp2);

    dcs_mat.postMult(tmp);
    currentVelocity = (c.y - _initial.y()) * driveSpeed * -0.5;
    velDir = osg::Vec3(0.0 * currentVelocity, 1.0 * currentVelocity, 0.0 * currentVelocity);
    osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
    osg::Matrixd M2;
    M2.makeRotate(osg::Matrixd::inverse(cam->getViewMatrix()).getRotate());
    velDir = velDir * M2;
    tmp.makeTranslate(velDir);
    dcs_mat.postMult(tmp);
    cover->setXformMat(dcs_mat);

    _counter++;
}

void MultitouchNavigation::fly(TouchContact c)
{
    if (_counter == 0)
    {
        _initial = osg::Vec3d(c.x, c.y, 0.);
    }
    osg::Vec3 velDir;
    osg::Matrix dcs_mat;
    dcs_mat = cover->getXformMat();
    float heading = 0.0;
    float pitch = (c.y - _initial.y()) / 300;
    float roll = (c.x - _initial.x()) / -300;
    osg::Matrix rot;
    MAKE_EULER_MAT(rot, heading, pitch, roll);
    dcs_mat.postMult(rot);
    velDir[0] = 0.0;
    velDir[1] = -1.0;
    velDir[2] = 0.0;
    /*osg::Camera *cam = coVRConfig::instance()->screens[0].camera;
	osg::Matrixd M2;
	M2.makeRotate(osg::Matrixd::inverse(cam->getViewMatrix()).getRotate());
	velDir = velDir * M2;*/
    float driveSpeed = coVRNavigationManager::instance()->getDriveSpeed();
    float currentVelocity = 10. * driveSpeed;
    osg::Matrix tmp;
    tmp.makeTranslate(velDir[0] * currentVelocity, velDir[1] * currentVelocity, velDir[2] * currentVelocity);
    dcs_mat.postMult(tmp);
    cover->setXformMat(dcs_mat);

    _counter++;
}