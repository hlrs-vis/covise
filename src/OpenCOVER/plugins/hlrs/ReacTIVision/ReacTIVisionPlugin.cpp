/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
**                                                            (C)2009 HLRS  **
**                                                                          **
** Description: ReacTIVision Plugin (does also TouchTable)                  **
**                                                                          **
**                                                                          **
** Author: B. Burbaum                                                       **
**                		                                                    **
**                                                                          **
** History:  								                                **
** Feb-09  v1	    				       		                            **
**                                                                          **
**                                                                          **
\****************************************************************************/

#include "ReacTIVisionPlugin.h"

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/RenderObject.h>
#include <config/CoviseConfig.h>
#include <cover/ARToolKit.h>
#include <cover/coVRConfig.h>
#include <osg/GraphicsContext>

#include "TuioListener.h" // hilft das ??
#include "TuioClient.h"

#include "BBM_Event.h"

ReacTIVisionPlugin::ReacTIVisionPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    fprintf(stderr, "ReacTIVisionPlugin::ReacTIVisionPlugin\n");
    int port = 3333;
    //	ReacTIVisionPlugin dump;                // TuioDump dump;
    //	TuioDump dump;                // TuioDump dump;
    //	TuioClient *client = new TuioClient(port);
    client = new TuioClient(port);
    //	client->addTuioListener(this);
    // client->
    client->start();
    coVRTouchTable::instance()->ttInterface = this;
}

// this is called if the plugin is removed at runtime
ReacTIVisionPlugin::~ReacTIVisionPlugin()
{
    fprintf(stderr, "ReacTIVisionPlugin::~ReacTIVisionPlugin\n");
    //NoTouch  TouchScreenDevice::destroy();
    client->stop();
    //	delete client;   Löscht den client, nicht den Pointer         ??????????????????????
    client = NULL;
}

void ReacTIVisionPlugin::preFrame()
{
    client->mutex.lock();

    //First Object = Marker
    std::list<TuioObject *> *my_TuioObjekt_List = client->getTuioObjects(); // geänderte Fassung

    std::list<TuioObject *>::iterator my_iter;
    for (my_iter = my_TuioObjekt_List->begin(); my_iter != my_TuioObjekt_List->end(); my_iter++)
    {
        TuioObject *tobj = *my_iter;
        std::cout << "set obj " << tobj->getFiducialID() << " (" << tobj->getSessionID() << ") " << tobj->getX() << " " << tobj->getY() << " " << tobj->getAngle()
                  << " " << tobj->getMotionSpeed() << " " << tobj->getRotationSpeed() << " " << tobj->getMotionAccel() << " " << tobj->getRotationAccel() << std::endl;
    }

    //Second Cursors = Finger  = Mouse
    // Object --> Cursor
    std::list<TuioCursor *> *my_TuioCursor_List = client->getTuioCursors(); // geänderte Fassung

    std::list<TuioCursor *>::iterator my_iter2;
    for (my_iter2 = my_TuioCursor_List->begin(); my_iter2 != my_TuioCursor_List->end(); my_iter2++)
    {
        TuioCursor *tcur = *my_iter2;
        std::cout << "set cur " << tcur->getFingerID() << " (" << tcur->getSessionID() << ") " << tcur->getX() << " " << tcur->getY()
                  << " " << tcur->getMotionSpeed() << " " << tcur->getMotionAccel() << " " << std::endl;

        //! Notify that a finger has moved   void TouchTablePlugin::fingerUpdate(TouchData data)
        const osg::GraphicsContext::Traits *traits = coVRConfig::instance()->windows[0].window->getTraits();
        if (!traits)
            return;

        //   printf("BB1 Finger Update X:%f Y:%f Area:%f width:%f  \n", data.X, data.Y, data.area, data.width );
        //   printf("BB1 Finger Update X:%f Y:%f Area:%f width:%f angle=%f  id:%i idtag: %i \n", data.X, data.Y, data.area, data.width,data.angle , data.ID, data.tagID);

        // cover->getMouseButton()->setWheel((int)data.angle);
        cover->getMouseButton()->setWheel((int)tcur->getMotionAccel()); // no  data.angle
        //cover->handleMouseEvent(osgGA::GUIEventAdapter::DRAG,data.X*traits->width,(1.0 -data.Y)*traits->height);
        cover->handleMouseEvent(osgGA::GUIEventAdapter::DRAG, tcur->getX() * traits->width, (1.0 - tcur->getY()) * traits->height);

        // if(data.area < -0.004 && buttonState == 0)

        //  if(data.area < FingerSizeValue && buttonState == 0)
        //if(tcur->getFingerDiameter() < FingerSizeValue && buttonState == 0)
        if (tcur->getMotionAccel() < FingerSizeValue && buttonState == 0)
        {
            buttonState = 1;
            cover->handleMouseEvent(osgGA::GUIEventAdapter::PUSH, buttonState, 0);
        }
    }
    client->mutex.unlock();
}

// for class COVEREXPORT coVRTouchTableInterface
int ReacTIVisionPlugin::getMarker(std::string name)
{
    //	client->mutex.lock();
    {
        std::list<TuioObject *> *my_TuioObjekt_List = client->getTuioObjects();
        std::list<TuioObject *>::iterator my_iter;
        for (my_iter = my_TuioObjekt_List->begin(); my_iter != my_TuioObjekt_List->end(); my_iter++)
        {
            if ((*my_iter)->getFiducialID() == atoi(name.c_str()))
            {
                //				client->mutex.unlock();
                return (*my_iter)->getFiducialID();
            }
        }
    }
    //	client->mutex.unlock();
    //  Never reach this  point
    return 0;
}

bool ReacTIVisionPlugin::isVisible(int objekt_id)
{
    client->mutex.lock();
    {
        std::list<TuioObject *> *my_TuioObjekt_List = client->getTuioObjects();
        std::list<TuioObject *>::iterator my_iter;
        for (my_iter = my_TuioObjekt_List->begin(); my_iter != my_TuioObjekt_List->end(); my_iter++)
        {
            if ((*my_iter)->getFiducialID() == objekt_id)
            {
                client->mutex.unlock();
                return true;
            }
        }
    }
    client->mutex.unlock();
    return false;
}

osg::Vec2 ReacTIVisionPlugin::getPosition(int objekt_id)
{
    client->mutex.lock();
    {
        std::list<TuioObject *> *my_TuioObjekt_List = client->getTuioObjects();
        std::list<TuioObject *>::iterator my_iter;
        for (my_iter = my_TuioObjekt_List->begin(); my_iter != my_TuioObjekt_List->end(); my_iter++)
        {
            if ((*my_iter)->getFiducialID() == objekt_id)
            {
                client->mutex.unlock();
                return osg::Vec2((*my_iter)->getX(), (*my_iter)->getY());
            }
        }
    }
    client->mutex.unlock();
    //  Never reach this  point
    return osg::Vec2(0.0, 0.0);
}

float ReacTIVisionPlugin::getOrientation(int objekt_id)

{
    client->mutex.lock();
    {
        std::list<TuioObject *> *my_TuioObjekt_List = client->getTuioObjects();
        std::list<TuioObject *>::iterator my_iter;
        for (my_iter = my_TuioObjekt_List->begin(); my_iter != my_TuioObjekt_List->end(); my_iter++)
        {
            if ((*my_iter)->getFiducialID() == objekt_id)
            {
                client->mutex.unlock();
                return (*my_iter)->getAngle();
            }
        }
    }
    client->mutex.unlock();
    //  Never reach this  point
    return (0.0);
}

COVERPLUGIN(ReacTIVisionPlugin)
