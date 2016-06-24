/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//----------------------------------------------//
//												//
//												//
// Note:										//
// Obsolete, use MultitouchNavigation instead	//
//												//
//												//
//----------------------------------------------//

#include "SurfaceInteraction.h"
#include <cover/coVRConfig.h>
#include <OpenVRUI/osg/mathUtils.h>
#include <osg/io_utils>

SurfaceInteraction::SurfaceInteraction()
{
    cout << "SurfaceInteraction::SurfaceInteraction" << endl;
    _counter = 0;
    //runningState = StateNotRunning;
}

SurfaceInteraction::~SurfaceInteraction()
{
}

void SurfaceInteraction::rotateXY(std::list<SurfaceContact> &contacts)
{
    if (contacts.size() != 1)
    {
        cerr << "ERROR (SurfaceInteraction): ROTATEXY with " << contacts.size() << " contacts" << endl;
        return;
    }

    // Concept based on http://viewport3d.com/trackball.htm

    //get screen dimensions
    float screenHeight = cover->frontWindowVerticalSize;
    float screenWidth = cover->frontWindowHorizontalSize;
    // get current screen position of finger
    osg::Vec2d currentPosition2D(contacts.front().CenterX, screenHeight - contacts.front().CenterY);

    // Translate 0,0 to the center
    double x = currentPosition2D.x() - screenWidth / 2;
    double y = currentPosition2D.y() - screenHeight / 2;

    // set bounds of unit sphere
    double currRadius = (currentPosition2D - osg::Vec2d(screenWidth / 2, screenHeight / 2)).length();
    if (_counter == 0)
    {
        // set to position of initial contact
        _radius = currRadius;
        // create a minimum sphere size, preventing too fast rotation
        double minRadius = osg::Vec2d(screenWidth * 1. / 8., screenHeight * 1 / 8.).length();
        if (_radius < minRadius)
        {
            _radius = minRadius;
        }
        // alternatively set radius to screen diagonal
        //_radius = sqrt((screenHeight*screenHeight)+(screenWidth*screenWidth)) / 2;
    }
    else if (currRadius > _radius && currRadius < _previousValue)
    {
        _radius = _previousValue;
    }
    _previousValue = currRadius;

    // normalize
    x /= _radius;
    y /= _radius;

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
        osg::Camera *cam = coVRConfig::instance()->channels[0].camera;
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

double SurfaceInteraction::angleBetween3DVectors(osg::Vec3 v1, osg::Vec3 v2)
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
    if (std::isnan(angle)) //__isnand(x)
        return 0;

    // Return the angle in radians
    return (angle);
}

void SurfaceInteraction::moveXY(std::list<SurfaceContact> &contacts)
{
    if (contacts.size() != 2)
    {
        cerr << "ERROR (SurfaceInteraction): MOVEXY with " << contacts.size() << " contacts" << endl;
        return;
    }

    osg::Vec2d currentPosition2DFinger1(contacts.front().CenterX, contacts.front().CenterY);
    osg::Vec2d currentPosition2DFinger2(contacts.back().CenterX, contacts.back().CenterY);

    // calculate ModelView - Projection - Window Transformation
    osg::Camera *cam = coVRConfig::instance()->channels[0].camera;
    osg::Matrix MVPW(cam->getViewMatrix() * cam->getProjectionMatrix() * cam->getViewport()->computeWindowMatrix());
    osg::Matrixd inverseMVPW = osg::Matrixd::inverse(MVPW);

    // determine z plane of Xform in screen coordinates
    osg::Vec3d XformTranslation2D = cover->getXformMat().getTrans() * MVPW;
    // determine center of both fingers
    osg::Vec3d currentVector3D((currentPosition2DFinger1.x() + currentPosition2DFinger2.x()) / 2,
                               cam->getViewport()->height() - (currentPosition2DFinger1.y() + currentPosition2DFinger2.y()) / 2,
                               XformTranslation2D.z());

    if (_counter > 0)
    {
        osg::Matrixd temp;
        if (coVRNavigationManager::instance()->getMode() == 4)
        {
            currentVector3D /= 100.;
            temp.makeTranslate(currentVector3D * inverseMVPW);
        }
        else
            temp.makeTranslate(currentVector3D * inverseMVPW - _previousVector3D * inverseMVPW);
        cover->setXformMat(cover->getXformMat() * temp);
    }

    _previousVector3D = currentVector3D;
    _counter++;
}

void SurfaceInteraction::scaleXYZ(std::list<SurfaceContact> &contacts)
{
    if (contacts.size() != 2)
    {
        cerr << "ERROR (SurfaceInteraction): SCALEXYZ with " << contacts.size() << " contacts" << endl;
        return;
    }

    // convert contact coordinates to current vectors
    osg::Vec2d currentPosition2DFinger1(contacts.front().CenterX, contacts.front().CenterY);
    osg::Vec2d currentPosition2DFinger2(contacts.back().CenterX, contacts.back().CenterY);

    // get length of difference
    double currentDistance = osg::Vec2d(currentPosition2DFinger1 - currentPosition2DFinger2).length();

    // divide by previous difference if != 0
    if (_counter > 0 && _previousValue != 0.)
    {
        cover->setScale(cover->getScale() * (currentDistance / _previousValue));
    }

    _previousValue = currentDistance;
    _counter++;
}

void SurfaceInteraction::rotateZ(std::list<SurfaceContact> &contacts)
{
    if (contacts.size() != 2)
    {
        cerr << "ERROR (SurfaceInteraction): ROTATEZ with " << contacts.size() << " contacts" << endl;
        return;
    }

    // get current screen position of finger
    osg::Vec3d curr3DVec1(contacts.front().CenterX, cover->frontWindowVerticalSize - contacts.front().CenterY, 0.);
    osg::Vec3d curr3DVec2(contacts.back().CenterX, cover->frontWindowVerticalSize - contacts.back().CenterY, 0.);

    if (_counter > 0)
    {
        // calculate distances between current and previous vectors
        double deltaCurrPrev1 = osg::Vec3d(curr3DVec1 - _prev3DVec1).length();
        double deltaCurrPrev2 = osg::Vec3d(curr3DVec2 - _prev3DVec2).length();

        // calculate angle & axis
        double angle = angleBetween3DVectors((_prev3DVec1 - _prev3DVec2), (curr3DVec1 - curr3DVec2));
        osg::Vec3d axis = (_prev3DVec1 - _prev3DVec2) ^ (curr3DVec1 - curr3DVec2);

        // figure out rotation
        osg::Camera *cam = coVRConfig::instance()->channels[0].camera;
        osg::Matrixd M;
        M.makeRotate(osg::Matrixd::inverse(cam->getViewMatrix()).getRotate());
        axis = axis * M;
        osg::Quat delta = osg::Quat(angle, axis);

        // apply to XformMat
        osg::Matrixd temp;
        temp.makeRotate(delta);
        cover->setXformMat(cover->getXformMat() * temp);
    }

    _prev3DVec1 = curr3DVec1;
    _prev3DVec2 = curr3DVec2;
    _counter++;
}

void SurfaceInteraction::moveZ(std::list<SurfaceContact> &contacts)
{
    if (contacts.size() != 3)
    {
        cerr << "ERROR (SurfaceInteraction): MOVEZ with " << contacts.size() << " contacts" << endl;
        return;
    }

    // get screen dimensions
    double height = cover->frontWindowVerticalSize;
    double width = cover->frontWindowHorizontalSize;
    //get current positions of contacts
    osg::Vec2d currentPositions2D[3];
    int i = 0;
    std::list<SurfaceContact>::iterator it;
    for (it = contacts.begin(); it != contacts.end(); it++)
    {
        currentPositions2D[i] = osg::Vec2d((*it).CenterX, height - (*it).CenterY);
        i++;
    }

    // determine y-center of contacts
    double yMin = currentPositions2D[0].y();
    double yMax = currentPositions2D[0].y();
    for (i = 0; i < 3; i++)
    {
        if (currentPositions2D[i].y() > yMax)
            yMax = currentPositions2D[i].y();
        else if (currentPositions2D[i].y() < yMin)
            yMin = currentPositions2D[i].y();
    }
    double currentYCenter = (yMin + yMax) / 2;

    // calculate ModelView - Projection - Window Matrix to transform screen position to world position:
    osg::Camera *cam = coVRConfig::instance()->channels[0].camera;
    osg::Matrix MVPW(cam->getViewMatrix() * cam->getProjectionMatrix() * cam->getViewport()->computeWindowMatrix());
    osg::Matrixd inverseMVPW = osg::Matrixd::inverse(MVPW);

    // determine z plane of Xform in screen coordinates
    if (_counter == 0)
        _initial = cover->getXformMat().getTrans() * MVPW;
    // transform y-center to 3D world coordinate system (x chosen randomly)
    osg::Vec3d currentVector3D(width / 2, currentYCenter, _initial.z());

    if (_counter > 0)
    {
        //rotate y to z and apply transformation
        osg::Matrixd trans, M;
        osg::Vec3d yMov = _previousVector3D * inverseMVPW - currentVector3D * inverseMVPW;
        if (coVRNavigationManager::instance()->getMode() == 4)
        {
            yMov = currentVector3D * inverseMVPW;
            yMov /= cover->frontVerticalSize / 100.;
        }
        M.makeRotate(osg::PI_2, osg::X_AXIS);
        yMov = yMov * M;
        trans.makeTranslate(yMov);
        cover->setXformMat(cover->getXformMat() * trans);
    }
    _previousVector3D = currentVector3D;
    _counter++;
}

//void SurfaceInteraction::update()
//{
//	cout << "SurfaceInteraction::update()" << endl;
//	vruiButtons * button = vruiRendererInterface::the()->getButtons();
//
//	runningState = StateNotRunning;
//
//	if(state == Idle)
//	{
//		if(button->wasPressed())
//		{
//			if(type == ButtonA || type == AllButtons)
//			{
//				if(button->getStatus() == vruiButtons::ACTION_BUTTON)
//				{
//					if(activate())
//					{
//						runningState = StateStarted;
//						startInteraction();
//					}
//				}
//			}
//		}
//	}
//	else if (state == Active || state == Paused || state == ActiveNotify)
//	{
//		if(type == ButtonA || type == AllButtons)
//		{
//			if(button->getStatus() == vruiButtons::ACTION_BUTTON)
//			{
//				if(state == Paused)
//				{
//					runningState = StateStopped;
//				}
//				else
//				{
//					runningState = StateRunning;
//					doInteraction();
//				}
//			}
//			else
//			{
//				runningState = StateStopped;
//				stopInteraction();
//				state = Idle;
//			}
//		}
//	}
//}

//void SurfaceInteraction::cancelInteraction()
//{
//	cout << "SurfaceInteraction::cancelInteraction()" << endl;
//	if (state == Active || state == Paused || state == ActiveNotify)
//	{
//		runningState = StateNotRunning;
//		stopInteraction();
//		state = Idle;
//	}
//}

//void SurfaceInteraction::startInteraction()
//{
//	cout << "SurfaceInteraction::startInteraction()" << endl;
//}
//
//void SurfaceInteraction::doInteraction()
//{
//	if (cover->debugLevel(4))
//		cout << "SurfaceInteraction::doInteraction()" << endl;
//}
//
//void SurfaceInteraction::stopInteraction()
//{
//	cout << "SurfaceInteraction::stopInteraction()" << endl;
//}
