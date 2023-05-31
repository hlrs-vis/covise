/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2012 HLRS  **
**                                                                          **
** Description: Multitouch Plugin											**
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

#include "MultitouchPlugin.h"
#include <cover/input/input.h>
#include <cover/input/coMousePointer.h>
#include <util/unixcompat.h>

MultitouchPlugin::MultitouchPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "MultitouchPlugin::MultitouchPlugin\n");
    _navMode = 1;
    _mode = _prevMode = NONE;
    _navigation = new MultitouchNavigation();
    _buttonState = 0;
    _mouseID = -1;
    _skippedFrames = 0;
}

MultitouchPlugin::~MultitouchPlugin()
{
    fprintf(stderr, "MultitouchPlugin::~MultitouchPlugin\n");
    _navigation->~MultitouchNavigation();
}

void MultitouchPlugin::addContact(TouchContact &c)
{
    _contacts.push_back(c);

    // determine interaction mode:
    // multi touch for Xform, Scale and Drive
    // virtual mouse for other navigation modes
    determineInteractionMode();
}
void MultitouchPlugin::addMouseContact()
{
    // mouse emulation
    if (_contacts.size() == 1)
    {
        TouchContact c = _contacts.front();
        Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::MOVE, static_cast<int>(c.x), static_cast<int>(cover->frontWindowVerticalSize - c.y));
        _buttonState = 1;
        Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::PUSH, _buttonState, 0);
        _mouseID = c.id; // identify this contact as mouse input
    }
}

void MultitouchPlugin::updateContact(TouchContact &c)
{
    //update mouse, but only if this is the one we used as a mouse input
    if (_mode == MOUSE && c.id == _mouseID)
    {
        Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::DRAG, static_cast<int>(c.x), static_cast<int>(cover->frontWindowVerticalSize - c.y));
    }

    // update contact
    std::list<TouchContact>::iterator it;
    for (it = _contacts.begin(); it != _contacts.end(); it++)
    {
        if (it->id == c.id)
        {
            (*it) = c;
            return;
        }
    }

    cerr << "EXCEPTION @MultitouchPlugin::changedContact: \n contact ID = " << c.id << " could not be updated" << endl;
    cerr << "removing and re-adding contact..." << endl;
    removeContact(c);
    addContact(c);
}

void MultitouchPlugin::removeContact(TouchContact &c)
{
    // mouse emulation
    if (_contacts.size() == 1)
    {
        // trigger MouseDown if no finger movement
        if (_mode == TBD && _prevMode == NONE)
        {
            addMouseContact();
        }

        // trigger MouseUp
        if (_mode == MOUSE || (_mode == TBD && _prevMode == NONE))
        {
            if (c.id == _mouseID)
            {
                Input::instance()->mouse()->handleEvent(osgGA::GUIEventAdapter::RELEASE, 0, 0);
                _buttonState = 0;
                _mouseID = -1;
            }
        }
    }

    // remove contact from list
    for (std::list<TouchContact>::iterator it = _contacts.begin(); it != _contacts.end(); it++)
    {
        if (it->id == c.id)
        {
            _contacts.erase(it);
            determineInteractionMode();
            return;
        }
    }

    // failed to remove contact, print error
    cerr << "EXCEPTION @MultitouchPlugin::removedContact: \n contact ID = " << c.id << " could not be removed" << endl;
}

TouchContact MultitouchPlugin::getContactCenter()
{
    if (_contacts.size() > 1)
    {
        const int numContacts = _contacts.size();
        osg::Vec2d *contactList = new osg::Vec2d[numContacts];
        int i = 0;
        std::list<TouchContact>::iterator it;
        for (it = _contacts.begin(); it != _contacts.end(); it++)
        {
            contactList[i] = osg::Vec2d((*it).x, cover->frontWindowVerticalSize - (*it).y);
            i++;
        }

        // determine center of contacts
        float xMin = contactList[0].x();
        float xMax = contactList[0].x();
        float yMin = contactList[0].y();
        float yMax = contactList[0].y();
        for (i = 0; i < numContacts; i++)
        {
            if (contactList[i].x() > xMax)
                xMax = contactList[i].x();
            else if (contactList[i].x() < xMin)
                xMin = contactList[i].x();
            if (contactList[i].y() > yMax)
                yMax = contactList[i].y();
            else if (contactList[i].y() < yMin)
                yMin = contactList[i].y();
        }

        TouchContact c((xMin + xMax) / 2, (yMin + yMax) / 2, -1);
        return c;
    }

    TouchContact c(_contacts.front().x, cover->frontWindowVerticalSize - _contacts.front().y, -1);
    return c;
}

std::list<TouchContact> MultitouchPlugin::getContacts()
{
    return _contacts;
}

void MultitouchPlugin::determineInteractionMode()
{
    _navigation->reset();
    reset();

    // navModes: NavNone = 0, XForm, Scale, Fly, Glide/Drive, Walk

    if (_contacts.size() == 0)
    {
        _mode = NONE;
    }
    else if (_contacts.size() == 1)
    {
        // 1. emulate MOUSE if NavNone
        // 2. wait for movement
        if (_navMode == 0)
        {
            _mode = MOUSE;
            addMouseContact();
        }
        else
            _mode = TBD;
    }
    else if (_contacts.size() == 2)
    {
        // 1. do nothing if Fly, Walk
        // 2. wait for movement to recognize gesture
        if (_navMode == 3 || _navMode == 5)
            _mode = NONE;
        else
            _mode = TBD;
    }
    else if (_contacts.size() == 3)
    {
        // 1. do nothing if Scale, Fly, Walk
        // 2. C_MOVEZ if Drive/Glide
        // 3. MOVEZ
        if (_navMode == 2 || _navMode == 3 || _navMode == 5)
            _mode = NONE;
        else if (_navMode == 4)
            _mode = C_MOVEZ;
        else
            _mode = MOVEZ;
    }
    else
    {
        _mode = NONE;
        cout << "No interaction mode for " << _contacts.size() << " _contacts implemented." << endl;
    }
}

void MultitouchPlugin::recognizeGesture()
{
    // navModes: NavNone = 0, XForm, Scale, Fly, Glide/Drive, Walk

    if (_contacts.size() == 1)
    {
        osg::Vec2f curr2DVec1(getContactCenter().x, getContactCenter().y);

        // wait for previous value
        if (_counter > 0)
        {
            float deltaCurrPrev1 = osg::Vec2f(curr2DVec1 - _prev2DVec1).length();

            if (deltaCurrPrev1 > 2.) // find suited, lower value
            {
                _counter++;
                _skippedFrames = 0;
                if (_counter > 2)
                {
                    if (_navMode == 2)
                    {
                        // navMode Scale
                        _mode = MOVEXY;
                    }
                    else if (_navMode == 3 || _navMode == 5)
                    {
                        // navMode Fly, Walk
                        _mode = NONE;
                    }
                    else
                    {
                        _mode = ROTATEXY;
                    }
                }
            }
            else
            {
                _skippedFrames++;
                if (_skippedFrames > 5)
                    _skippedFrames = _counter = 0;
            }
        }

        _prev2DVec1 = curr2DVec1;
        if (_counter == 0)
            _counter++;
    }
    else if (_contacts.size() == 2)
    {
        osg::Vec2d curr2DVec1(_contacts.front().x, cover->frontWindowVerticalSize - _contacts.front().y);
        osg::Vec2d curr2DVec2(_contacts.back().x, cover->frontWindowVerticalSize - _contacts.back().y);

        // wait for previous value
        if (_counter > 0)
        {
            // calculate distances between current and previous vectors
            double distanceCurrCurr = osg::Vec2d(curr2DVec1 - curr2DVec2).length();
            double distancePrevPrev = osg::Vec2d(_prev2DVec1 - _prev2DVec2).length();
            double deltaScale = distanceCurrCurr - distancePrevPrev;
            deltaScale = sqrt(deltaScale * deltaScale);
            double deltaCurrPrev1 = osg::Vec2d(curr2DVec1 - _prev2DVec1).length();
            double deltaCurrPrev2 = osg::Vec2d(curr2DVec2 - _prev2DVec2).length();
            double deltaRot = angleBetween2DVectors((_prev2DVec1 - _prev2DVec2), (curr2DVec1 - curr2DVec2));
            deltaRot = sqrt(deltaRot * deltaRot);
            int iterations = 2;

            // wait for movement of fingers
            if (deltaCurrPrev1 > 0.001 || deltaCurrPrev2 > 0.001)
            {
                _skippedFrames = 0;
                // check if 2 finger have moved
                if (deltaCurrPrev1 == 0. || deltaCurrPrev2 == 0.)
                {
                    // only 1 finger has moved, therefore can't be MOVE
                    // check if SCALE > ROTATE (percentage)
                    //	ROTATE
                    // else
                    //	SCALE

                    if (deltaScale / _initialDistance >= deltaRot / 2. * osg::PI)
                    {
                        _scale++;
                        if (_scale > iterations)
                        {
                            if (_navMode == 4)
                            {
                                // navMode Drive
                                _mode = C_SCALEXYZ;
                            }
                            else
                            {
                                _mode = SCALEXYZ;
                            }
                        }
                    }
                    else
                    {
                        _rotate++;
                        if (_rotate > iterations)
                        {
                            _mode = ROTATEZ;
                        }
                    }
                }
                else
                {
                    //check if SCALE > ROTATE (percentage)

                    if (deltaScale / _initialDistance >= deltaRot / 2. * osg::PI)
                    {
                        //if deltaScale/2 > distance of mid between current contacts and previous contacts
                        //	SCALE
                        //else
                        //	MOVE

                        osg::Vec2d middleCurrCurr(0.5 * (curr2DVec1.x() + curr2DVec2.x()), 0.5 * (curr2DVec1.y() + curr2DVec2.y()));
                        osg::Vec2d middlePrevPrev(0.5 * (_prev2DVec1.x() + _prev2DVec2.x()), 0.5 * (_prev2DVec1.y() + _prev2DVec2.y()));
                        double deltaMove = (middleCurrCurr - middlePrevPrev).length();

                        if (deltaScale / 2. > deltaMove)
                        {
                            _scale++;
                            if (_scale > iterations)
                            {
                                if (_navMode == 4)
                                {
                                    // navMode Drive
                                    _mode = C_SCALEXYZ;
                                }
                                else
                                {
                                    _mode = SCALEXYZ;
                                }
                            }
                        }
                        else
                        {
                            _move++;
                            if (_move > iterations)
                            {
                                if (_navMode == 2)
                                {
                                    // navMode Scale
                                    _mode = NONE;
                                }
                                else if (_navMode == 4)
                                {
                                    // navMode Drive
                                    _mode = C_MOVEXY;
                                }
                                else
                                {
                                    _mode = MOVEXY;
                                }
                            }
                        }
                    }
                    else
                    {
                        // if pivot point of rotation is between fingers
                        //	ROTATE
                        //else
                        //	MOVE

                        osg::Vec3d lineCurrCurr = osg::Vec3d(curr2DVec1.x(), curr2DVec1.y(), 1.0) ^ osg::Vec3d(curr2DVec2.x(), curr2DVec2.y(), 1.0);
                        osg::Vec3d linePrevPrev = osg::Vec3d(_prev2DVec1.x(), _prev2DVec1.y(), 1.0) ^ osg::Vec3d(_prev2DVec2.x(), _prev2DVec2.y(), 1.0);
                        osg::Vec3d interception = lineCurrCurr ^ linePrevPrev;
                        if (interception.z() != 0.)
                        {
                            int x = interception.x() / interception.z();
                            int y = interception.y() / interception.z();
                            // determine dimensions of smallest enveloping plane
                            float xMin = curr2DVec1.x();
                            float xMax = curr2DVec1.x();
                            float yMin = curr2DVec1.y();
                            float yMax = curr2DVec1.y();
                            if (curr2DVec2.x() < xMin)
                                xMin = curr2DVec2.x();
                            else if (curr2DVec2.x() > xMax)
                                xMax = curr2DVec2.x();
                            if (curr2DVec2.y() > yMax)
                                yMax = curr2DVec2.y();
                            else if (curr2DVec2.y() < yMin)
                                yMin = curr2DVec2.y();

                            if (x > xMin && x < xMax && y > yMin && y < yMax)
                            {
                                _rotate++;
                                if (_rotate > iterations)
                                {
                                    _mode = ROTATEZ;
                                }
                            }
                            else
                            {
                                _move++;
                                if (_move > iterations)
                                {
                                    if (_navMode == 2)
                                    {
                                        // navMode Scale
                                        _mode = NONE;
                                    }
                                    else if (_navMode == 4)
                                    {
                                        // navMode Drive
                                        _mode = C_MOVEXY;
                                    }
                                    else
                                    {
                                        _mode = MOVEXY;
                                    }
                                }
                            }
                        }
                        else
                        {
                            _move++;
                            if (_move > iterations)
                            {
                                if (_navMode == 2)
                                {
                                    // navMode Scale
                                    _mode = NONE;
                                }
                                else if (_navMode == 4)
                                {
                                    // navMode Drive
                                    _mode = C_MOVEXY;
                                }
                                else
                                {
                                    _mode = MOVEXY;
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                _skippedFrames++;
                if (_skippedFrames > 5)
                    _move = _rotate = _scale = _skippedFrames = 0;
            }
        }

        _prev2DVec1 = curr2DVec1;
        _prev2DVec2 = curr2DVec2;
        if (_counter == 0)
        {
            _initialDistance = osg::Vec2d(curr2DVec1 - curr2DVec2).length();
        }
        _counter++;
    }
}

double MultitouchPlugin::angleBetween2DVectors(osg::Vec2 v1, osg::Vec2 v2)
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

void MultitouchPlugin::preFrame()
{
    // navModes: NavNone = 0, XForm, Scale, Fly, Glide/Drive, Walk
    _navMode = coVRNavigationManager::instance()->getMode();

    if (_mode == NONE || _navMode == 0)
        return;
    else
    {
        if (_mode == TBD)
        {
            recognizeGesture();
        }
        else if (_mode == ROTATEXY)
        {
            _navigation->rotateXY(getContactCenter());
        }
        else if (_mode == MOVEXY)
        {
            _navigation->moveXY(getContactCenter());
        }
        else if (_mode == C_MOVEXY)
        {
            _navigation->continuousMoveXY(getContactCenter());
        }
        else if (_mode == WALKXY)
        {
            _navigation->walkXY(getContactCenter());
        }
        else if (_mode == SCALEXYZ)
        {
            _navigation->scaleXYZ(getContacts());
        }
        else if (_mode == C_SCALEXYZ)
        {
            _navigation->continuousScaleXYZ(getContacts());
        }
        else if (_mode == ROTATEZ)
        {
            _navigation->rotateZ(getContacts());
        }
        else if (_mode == MOVEZ)
        {
            _navigation->moveZ(getContactCenter());
        }
        else if (_mode == C_MOVEZ)
        {
            _navigation->continuousMoveZ(getContactCenter());
        }
        else if (_mode == WALKZ)
        {
            _navigation->walkZ(getContactCenter());
        }
        else if (_mode == FLY)
        {
            _navigation->fly(getContactCenter());
        }
    }
}
