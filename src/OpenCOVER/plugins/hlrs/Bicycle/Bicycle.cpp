/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//
//

#include "Bicycle.h"
#include <cover/coVRTui.h>
#include <cover/coVRCommunication.h>
#include <cover/coIntersection.h>
#include <osg/LineSegment>
#include <osg/MatrixTransform>
#include "AVRserialcom.h"
#include <util/unixcompat.h>
#include <osgUtil/IntersectionVisitor>

#if !defined(_WIN32) && !defined(__APPLE__)
//#define USE_X11
#define USE_LINUX
#endif

// standard linux sockets
#include <sys/types.h>
#ifndef WIN32
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#endif
#include <pthread.h>
#include <stdio.h>
//#include <unistd. h.>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>

#ifdef USE_X11
#include <X11/Xlib.h>
#include <X11/extensions/XInput.h>
#include <X11/cursorfont.h>
#endif

#include <PluginUtil/PluginMessageTypes.h>

// socket (listen)
#define LOCAL_SERVER_PORT 9930

#define REMOTE_IP "141.58.8.26"
#define REMOTE_PORT 9931

#define BUF 512

using namespace covise;

bool flightgearReset=false;
int s, rc, n, len;
struct sockaddr_in cliAddr, servAddr;
char puffer[BUF];
time_t time1;
char loctime[BUF];
char *ptr;
const int y = 1;
pthread_t thdGetBicycleSpeedIns;
static volatile int rpiSpeed;

void *thdGetBicycleSpeed(void *ptr);

// socket (send)
#define BUFLEN 512

struct sockaddr_in si_other;
int ssend, i, slen = sizeof(si_other);
char buf[BUFLEN];

void setBicycleBreak(int breakvalue);

BicyclePlugin *BicyclePlugin::plugin = NULL;

// --------------------------------------------------------------------
// void setBicycleBreak(int breakvalue)
// breakvalue e {0,10000}
// --------------------------------------------------------------------
void setBicycleBreak(int breakvalue)
{
    if ((breakvalue >= 0) && (breakvalue <= 10000))
    {
        sprintf(buf, "%i\n", breakvalue);

        if (sendto(ssend, buf, BUFLEN, 0, (struct sockaddr *)&si_other, slen) == -1)
        {
            printf("Bicycle: sendto failed");
        }
    }
}

// --------------------------------------------------------------------

void *thdGetBicycleSpeed(void *ptr)
{
    while (1)
    {
        memset(puffer, 0, BUF);
        len = sizeof(cliAddr);

        n = recvfrom(s, puffer, BUF, 0, (struct sockaddr *)&cliAddr, (socklen_t *)&len);

        if (n >= 0)
        {
            sscanf(puffer, "%d", &rpiSpeed);
            fprintf(stderr, "message received %d\n", rpiSpeed);
        }
    }

    // char *message;
    // message = (char *) ptr;
    // printf("%s \n", message);
    return NULL;
}

// --------------------------------------------------------------------

static VrmlNode *creator(VrmlScene *scene)
{
    return new VrmlNodeBicycle(scene);
}

// --------------------------------------------------------------------
// Define the built in VrmlNodeType:: "Bicycle" fields
// --------------------------------------------------------------------

VrmlNodeType *VrmlNodeBicycle::defineType(VrmlNodeType *t)
{
    static VrmlNodeType *st = 0;

    if (!t)
    {
        if (st)
            return st; // Only define the type once.
        t = st = new VrmlNodeType("Bicycle", creator);
    }

    VrmlNodeChild::defineType(t); // Parent class
    t->addEventOut("speed", VrmlField::SFFLOAT);
    t->addEventOut("revs", VrmlField::SFFLOAT);
    t->addEventOut("gear", VrmlField::SFINT32);
    t->addEventOut("button", VrmlField::SFINT32);
    t->addEventIn("thermal", VrmlField::SFBOOL);
    t->addExposedField("bikeRotation", VrmlField::SFROTATION);
    t->addExposedField("bikeTranslation", VrmlField::SFVEC3F);

    return t;
}

// --------------------------------------------------------------------

VrmlNodeType *VrmlNodeBicycle::nodeType() const
{
    return defineType(0);
}

// --------------------------------------------------------------------

VrmlNodeBicycle::VrmlNodeBicycle(VrmlScene *scene)
    : VrmlNodeChild(scene)
    , d_bikeRotation(1, 0, 0, 0)
    , d_bikeTranslation(0, 0, 0)
    , d_button(0)
    , d_thermal(0)
{
    setModified();
    bikeTrans.makeIdentity();
    fgPlaneZeroPos[0] = 0.0;
    fgPlaneZeroPos[1] = 0.0;
    fgPlaneZeroPos[2] = 0.0;
}

// --------------------------------------------------------------------

VrmlNodeBicycle::VrmlNodeBicycle(const VrmlNodeBicycle &n)
    : VrmlNodeChild(n.d_scene)
    , d_bikeRotation(n.d_bikeRotation)
    , d_bikeTranslation(n.d_bikeTranslation)
    , d_button(n.d_button)
    , d_thermal(n.d_thermal)
{
    setModified();
    bikeTrans.makeIdentity();
    fgPlaneZeroPos[0] = 0.0;
    fgPlaneZeroPos[1] = 0.0;
    fgPlaneZeroPos[2] = 0.0;
}

// --------------------------------------------------------------------

VrmlNodeBicycle::~VrmlNodeBicycle()
{
}

// --------------------------------------------------------------------

VrmlNode *VrmlNodeBicycle::cloneMe() const
{
    return new VrmlNodeBicycle(*this);
}

// --------------------------------------------------------------------

VrmlNodeBicycle *VrmlNodeBicycle::toBicycle() const
{
    return (VrmlNodeBicycle *)this;
}

// --------------------------------------------------------------------

ostream &VrmlNodeBicycle::printFields(ostream &os, int indent)
{
    if (!d_bikeRotation.get())
    {
        PRINT_FIELD(bikeRotation);
    }
    if (!d_bikeTranslation.get())
    {
        PRINT_FIELD(bikeTranslation);
    }
    return os;
}

// --------------------------------------------------------------------
// Set the value of one of the node fields.
// --------------------------------------------------------------------

void VrmlNodeBicycle::setField(const char *fieldName, const VrmlField &fieldValue)
{
    if
        TRY_FIELD(bikeRotation, SFRotation)
    else if
        TRY_FIELD(bikeTranslation, SFVec3f)
    else if
        TRY_FIELD(thermal, SFBool)
    else
    {
        VrmlNodeChild::setField(fieldName, fieldValue);
        if (strcmp(fieldName, "bikeRotation") == 0)
        {
            recalcMatrix();
        }
        else if (strcmp(fieldName, "bikeTranslation") == 0)
        {
            recalcMatrix();
        }
    }
}

// --------------------------------------------------------------------

void VrmlNodeBicycle::recalcMatrix()
{
    float *ct = d_bikeTranslation.get();
    osg::Vec3 tr(ct[0], ct[1], ct[2]);
    bikeTrans.makeTranslate(tr);
    osg::Matrix rot;

    ct = d_bikeRotation.get();
    tr.set(ct[0], ct[1], ct[2]);
    rot.makeRotate(ct[3], tr);
    bikeTrans.preMult(rot);
    
       fprintf(stderr,"recalcbikeTrans: %f %f %f\n",bikeTrans(3, 0), bikeTrans(3, 1), bikeTrans(3, 2) );
}

const VrmlField *VrmlNodeBicycle::getField(const char *fieldName)
{
    if (strcmp(fieldName, "enabled") == 0)
        return &d_enabled;
    else if (strcmp(fieldName, "joystickNumber") == 0)
        return &d_joystickNumber;
    else if (strcmp(fieldName, "axes_changed") == 0)
        return &d_axes;
    else if (strcmp(fieldName, "buttons_changed") == 0)
        return &d_buttons;
    else
        cout << "Node does not have this eventOut or exposed field " << nodeType()->getName() << "::" << name() << "." << fieldName << endl;
    return 0;
}

void VrmlNodeBicycle::eventIn(double timeStamp,
                              const char *eventName,
                              const VrmlField *fieldValue)
{
    if (strcmp(eventName, "bikeRotation") == 0)
    {
        fprintf(stderr, "resetRot\n");
        setField(eventName, *fieldValue);
        recalcMatrix();
    }
    else if (strcmp(eventName, "bikeTranslation") == 0)
    {
        fprintf(stderr, "resetTrans\n");
        setField(eventName, *fieldValue);
        recalcMatrix();
    }
    else if (strcmp(eventName, "thermal") == 0)
    {
        setField(eventName, *fieldValue);
        fprintf(stderr, "thermal: %d\n", d_thermal.get());
        
    }
    // Check exposedFields
    else
    {
        VrmlNode::eventIn(timeStamp, eventName, fieldValue);
    }

    setModified();
}

// --------------------------------------------------------------------
/*
void BicyclePlugin::run()
{
   while (running)
   {
      // we get speed from raspi via UDP
      speedCounter = rpiSpeed;
      
      //AVRReadBytes(1,&speedCounter);
   }
}*/

// --------------------------------------------------------------------

char BicyclePlugin::readps2(int fd)
{
    char ch;

    while (read(fd, &ch, 1) && (ch == (char)0xfa || ch == (char)0xaa))
    {
        //fprintf(stderr,"<%02X>",ch&0xff);
    }
    //fprintf(stderr,"[%02X]",ch&0xff);
    return (ch);
}

// --------------------------------------------------------------------

void VrmlNodeBicycle::render(Viewer *)
{
    static bool firstTime = true;
    double dT = cover->frameDuration();
    float wheelBase = 0.98;
    float v = BicyclePlugin::plugin->speed; //*0.06222222;
    if(firstTime)
    {
            recalcMatrix();
	    firstTime=false;
    }
    //fprintf(stderr,"speed: %f", v);
    if (BicyclePlugin::plugin->skateboard)
    {
       osg::Vec3d normal = BicyclePlugin::plugin->skateboard->getNormal();
       fprintf(stderr, "normal(%f %f) %d\n", normal.x(), normal.y(), BicyclePlugin::plugin->skateboard->getButton());
       if (BicyclePlugin::plugin->skateboard->getButton() == 2)
       {
           BicyclePlugin::plugin->speed += 0.01;
       }
       if (BicyclePlugin::plugin->skateboard->getWeight() < 10)
       {
           BicyclePlugin::plugin->speed = 0;
       }
       BicyclePlugin::plugin->speed += -1*0.05*normal.y();
       if(BicyclePlugin::plugin->speed < 0)
       {
           BicyclePlugin::plugin->speed =0;
       }
       if(BicyclePlugin::plugin->speed > 5)
       {
           BicyclePlugin::plugin->speed =5;
       }
       if (BicyclePlugin::plugin->skateboard->getWeight() < 10)
       {
           BicyclePlugin::plugin->speed = 0;
       }
       v = BicyclePlugin::plugin->speed;
       float s = v * dT;
       osg::Vec3 V(0, 0, -s);
       wheelBase = 0.5;
       float rotAngle = 0.0;
        fprintf(stderr,"v: %f \n",v );
       if ((s < 0.0001 && s > -0.0001)) // straight
       {
           //fprintf(stderr,"bikeTrans: %f %f %f\n",bikeTrans(3, 0), bikeTrans(3, 1), bikeTrans(3, 2) );
           //fprintf(stderr,"V: %f %f %f\n",V[0], V[1], V[2] );
       }
       else
       {
           float wheelAngle = normal.x()/-2.0;
           //float r = tan(M_PI_2-vehicleParameters->getWheelAngle()) * wheelBase;
           float r = tan(M_PI_2 - wheelAngle * 0.2 / (((v * 0.2) + 1))) * wheelBase;
           float u = 2.0 * r * M_PI;
           rotAngle = (s / u) * 2.0 * M_PI;
           V[2] = -r * sin(rotAngle);
           V[0] = r - r * cos(rotAngle);
       }

       osg::Matrix relTrans;
       osg::Matrix relRot;
       relRot.makeRotate(rotAngle, 0, 1, 0);
       relTrans.makeTranslate(V);
       
       //fprintf(stderr,"bikeTrans: %f %f %f\n",bikeTrans(3, 0), bikeTrans(3, 1), bikeTrans(3, 2) );
       bikeTrans = relRot * relTrans * bikeTrans;

       moveToStreet();


       osg::Quat q;
       q.set(bikeTrans);
       osg::Quat::value_type orient[4];
       q.getRotate(orient[3], orient[0], orient[1], orient[2]);
       
       
       
        if (coVRMSController::instance()->isMaster())
        {
            coVRMSController::instance()->sendSlaves((char *)bikeTrans.ptr(), sizeof(bikeTrans));
            coVRMSController::instance()->sendSlaves((char *)orient, 4*sizeof(orient[0]));
        }
        else
        {
            coVRMSController::instance()->readMaster((char *)bikeTrans.ptr(), sizeof(bikeTrans));
            coVRMSController::instance()->readMaster((char *)orient, 4*sizeof(orient[0]));
        }
       
       d_bikeTranslation.set(bikeTrans(3, 0), bikeTrans(3, 1), bikeTrans(3, 2));
       d_bikeRotation.set(orient[0], orient[1], orient[2], orient[3]);
    }
    else if (BicyclePlugin::plugin->isPlane)
    { 
        osg::Quat::value_type orient[4];
        osg::Vec3d newPos;
        if (BicyclePlugin::plugin->flightgear)
        {
            int buttonState = 0;
            if (BicyclePlugin::plugin->isBike)
                buttonState = BicyclePlugin::plugin->tacx->getButtons();

            osg::Vec3d currentPosition(BicyclePlugin::plugin->flightgear->getPosition());
            if ((currentPosition[0] != 0.0 && fgPlaneZeroPos[0] == 0.0) || flightgearReset)   //buttonState)
            {
                osg::Vec3d referenceCoordSys(0,1,0); 
                osg::Vec3d zeroPosition(BicyclePlugin::plugin->flightgear->getPosition());
                double EarthRadius=zeroPosition.normalize();
                osg::Vec3d rotationAxis(zeroPosition^referenceCoordSys);
                osg::Quat fgRot;			
                double rotationAngle((zeroPosition*referenceCoordSys)); 
                //fprintf(stderr, "\r");
                /*for (unsigned i = 0; i < 3; ++i)
                  fprintf(stderr, "Ax: %6f ", rotationAxis[i]);
                  fprintf(stderr, "An: %6f ", rotationAngle);*/
                fgPlaneRot.makeRotate(acos(rotationAngle),rotationAxis);
                flightgearReset=false;
                fgPlaneZeroPos=fgPlaneRot.preMult(osg::Vec3d(BicyclePlugin::plugin->flightgear->getPosition())); 
            } 
            newPos = fgPlaneRot.preMult(currentPosition)-fgPlaneZeroPos;

            /*	fprintf(stderr, "\r");
                for (unsigned i = 0; i < 3; ++i)
                fprintf(stderr, "CurrPos: %6f ", newPos[i]);*/

            osg::Vec3d currentOrientation(BicyclePlugin::plugin->flightgear->getOrientation());
            osg::Vec3d newOrientation = currentOrientation;
            newOrientation.normalize();
            osg::Matrix planeOrientationMatrix;
            double len = currentOrientation.length();
            if (len > 0)
            {
                planeOrientationMatrix.makeRotate(len, newOrientation);
            }
            else
            {
                planeOrientationMatrix.makeIdentity();
            }

            osg::Matrix viewcorrectionMatrix;
            viewcorrectionMatrix.makeRotate(M_PI/3*2,osg::Vec3d(-1,-1,1));

            planeOrientationMatrix.postMult(fgPlaneRot);
            planeOrientationMatrix.preMult(viewcorrectionMatrix);

            osg::Quat q=planeOrientationMatrix.getRotate();
            q.getRotate(orient[3], orient[0], orient[1], orient[2]);
            /*	fprintf(stderr, "\r");
                for (unsigned i = 0; i < 3; ++i)
                fprintf(stderr, "Pos: %6f ", newPos[i]);*/


            //     double timeStamp = System::the->time();

            // eventOut(timeStamp, "bikeTranslation", d_bikeTranslation);
            //  eventOut(timeStamp, "bikeRotation", d_bikeRotation);
            BicyclePlugin::plugin->flightgear->setThermal(d_thermal.get());
        }



        if (coVRMSController::instance()->isMaster())
        {
            coVRMSController::instance()->sendSlaves((char *)newPos.ptr(), 3*sizeof(newPos[0]));
            coVRMSController::instance()->sendSlaves((char *)orient, 4*sizeof(orient[0]));
        }
        else
        {
            coVRMSController::instance()->readMaster((char *)newPos.ptr(), 3*sizeof(newPos[0]));
            coVRMSController::instance()->readMaster((char *)orient, 4*sizeof(orient[0]));
        }

        d_bikeTranslation.set(newPos[0]+2595,newPos[1]+300,newPos[2]-50);
        d_bikeRotation.set(orient[0], orient[1], orient[2], orient[3]); 
    }
    
    else
    {
		float s = v * dT;
		osg::Vec3 V(0, 0, -s);

		float rotAngle = 0.0;
		//if((vehicleParameters->getWheelAngle()>-0.0001 && vehicleParameters->getWheelAngle()><0.0001 )|| (s < 0.0001 && s > -0.0001)) // straight
		if ((s < 0.0001 && s > -0.0001)) // straight
		{
		}
		else
		{
			//float r = tan(M_PI_2-vehicleParameters->getWheelAngle()) * wheelBase;
			float r = tan(M_PI_2 - BicyclePlugin::plugin->angle * 0.2 / (((v * 0.2) + 1))) * wheelBase;
			float u = 2.0 * r * M_PI;
			rotAngle = (s / u) * 2.0 * M_PI;
			V[2] = -r * sin(rotAngle);
			V[0] = r - r * cos(rotAngle);
		}

		osg::Matrix relTrans;
		osg::Matrix relRot;
		relRot.makeRotate(rotAngle, 0, 1, 0);
		relTrans.makeTranslate(V);
		bikeTrans = relRot * relTrans * bikeTrans;

		moveToStreet();


		osg::Quat q;
		q.set(bikeTrans);
		osg::Quat::value_type orient[4];
		q.getRotate(orient[3], orient[0], orient[1], orient[2]);
		d_bikeTranslation.set(bikeTrans(3, 0), bikeTrans(3, 1), bikeTrans(3, 2));
		d_bikeRotation.set(orient[0], orient[1], orient[2], orient[3]);
	}
    double timeStamp = System::the->time();
    
    int buttonState = 0;
    if (coVRMSController::instance()->isMaster())
    {
        if (BicyclePlugin::plugin->tacx != NULL)
        {
            buttonState = BicyclePlugin::plugin->tacx->getButtons();
        }
        else if (BicyclePlugin::plugin->skateboard != nullptr)
        {
            buttonState = BicyclePlugin::plugin->skateboard->getButton();
        }
        coVRMSController::instance()->sendSlaves((char *)&buttonState, sizeof(buttonState));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&buttonState, sizeof(buttonState));
    }
    d_button.set(buttonState);

    eventOut(timeStamp, "bikeTranslation", d_bikeTranslation);
    eventOut(timeStamp, "bikeRotation", d_bikeRotation);
    if (d_button.get() != oldButtonState)
    {
        oldButtonState = d_button.get();
        eventOut(timeStamp, "button", d_button);
    }
}

//-----------------------------------------------------------------------------
// Name: UpdateInputState()
// Desc: Get the input device's state and display it.
//-----------------------------------------------------------------------------
void BicyclePlugin::UpdateInputState()
{
    /*counters[0]=0;
counters[1]=0;
   //int i;
      if(coVRMSController::instance()->isMaster())
      {
#ifdef USE_LINUX
   if (mouse1 != -1) {
      while (read(mouse1, buffer, 4) == 4)
      {
      counters[0]+=buffer[2];
         //wheelcounter -= buffer[3];
         buffer[0] &= (1 << NUM_BUTTONS) - 1;
         if(buffer[0] & 2)
         {
            //wheelcounter = 0;
            cout << "reset wheelcounter: " << (int)buffer[0] << endl;
         }
         //cout << "button: " << (int)buffer[0] << endl;
      }
      //printf("buffer: %d %d %d %d\n", buffer[0], buffer[1], buffer[2], buffer[3]);
      //printf("wheelcounter: %d\n", wheelcounter);
   }if (mouse2 != -1) {
      while (read(mouse2, buffer+4, 4) == 4)
      {
         counters[1]+=buffer[2+4];
         //wheelcounter -= buffer[3+4];
         buffer[0+4] &= (1 << NUM_BUTTONS) - 1;
         if(buffer[0+4] & 2)
         {
            //wheelcounter = 0;
            cout << "reset wheelcounter: " << (int)buffer[0] << endl;
         }
         //cout << "button: " << (int)buffer[0] << endl;
      }
      //printf("buffer: %d %d %d %d\n", buffer[0], buffer[1], buffer[2], buffer[3]);
      //printf("buffer: %d %d %d %d\n", buffer[0+4], buffer[1+4], buffer[2+4], buffer[3+4]);
      //printf("wheelcounter: %d\n", wheelcounter);
   }
#endif

         coVRMSController::instance()->sendSlaves((char *)counters,sizeof(counters));
      
}
      else
         coVRMSController::instance()->readMaster((char *)counters,sizeof(counters));
  
  
   //angleCounter+=counters[0];
   
   //speedCounter+=counters[1];
   buffer[2]=0;
   buffer[2+4]=0;
   
   angle = (angleCounter/536.0)*(M_PI/2.0);
   static double oldTime=0.0;
   if(oldTime==0.0)
       oldTime = cover->frameTime();
       
   if(cover->frameTime() > oldTime+1.0)
   {
       speed = rpiSpeed; //speedCounter;///(cover->frameTime() - oldTime);
       oldTime = cover->frameTime();
   }
   if(cover->number_axes[2]>2)
   {
   angle = (-cover->axes[2][2])/2.0;
   }
   if((speed > 105) || (angle != 0) )
   {
   //printf("angleCounter: %d angle: %f\n", angleCounter,angle);
   //printf("speedCounter: %d speed: %f\n", speedCounter, speed);
   }
   
      if(coVRMSController::instance()->isMaster())
      {
         coVRMSController::instance()->sendSlaves((char *)&angle,sizeof(angle));
}
      else
         coVRMSController::instance()->readMaster((char *)&angle,sizeof(angle));
 */
    if (coVRMSController::instance()->isMaster())
    {
        if (tacx != NULL && flightgear != NULL)
        {
            float f= 0;
            float propellerforce=0.5; //set to constant value for now, to be update from flight dynamics model
            f += propellerforce * BicyclePlugin::plugin->forceFactor->getValue();
	    angle = tacx->getAngle() * 2.0;
            speed = tacx->getSpeed() * velocityFactor->getValue();
	    power = tacx->getRPM()*0.5*velocityFactor->getValue();//*f;
            BicyclePlugin::plugin->tacx->setForce(f);
        }
        else if (tacx != NULL)
        {
            angle = tacx->getAngle() * 2.0;
            speed = tacx->getSpeed() * velocityFactor->getValue();
        }
        else if (flightgear != NULL)
        {
        angle = 0.0;
        speed = 20.0;
        power= 0.0;
        }
        coVRMSController::instance()->sendSlaves((char *)&angle, sizeof(angle));
        coVRMSController::instance()->sendSlaves((char *)&speed, sizeof(speed));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&angle, sizeof(angle));
        coVRMSController::instance()->readMaster((char *)&speed, sizeof(speed));
    }
    return;
}

void BicyclePlugin::tabletPressEvent(coTUIElement * /*tUIItem*/)
{
}

void BicyclePlugin::tabletEvent(coTUIElement * /*tUIItem*/)
{
    //if(tUIItem == springConstant)
    {
    }
}

BicyclePlugin::BicyclePlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
{
    //startThread();
    running = true;
}

void VrmlNodeBicycle::moveToStreet()
{
    this->moveToStreet(bikeTrans);
}

void VrmlNodeBicycle::moveToStreet(osg::Matrix &carTrans)
{
    //float carHeight=200;
    //  just adjust height here

    osg::Matrix carTransWorld, tmp;
    osg::Vec3 pos, p0, q0;

    osg::Matrix baseMat;
    osg::Matrix invBaseMat;

    baseMat = cover->getObjectsScale()->getMatrix();

    osg::Matrix transformMatrix = cover->getObjectsXform()->getMatrix();

    baseMat.postMult(transformMatrix);
    invBaseMat.invert(baseMat);

    osg::Matrix invVRMLRotMat;
    invVRMLRotMat.makeRotate(-M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));
    osg::Matrix VRMLRotMat;
    VRMLRotMat.makeRotate(M_PI / 2.0, osg::Vec3(1.0, 0.0, 0.0));

    carTransWorld = carTrans * VRMLRotMat * baseMat;
    pos = carTransWorld.getTrans();

    //pos[2] -= carHeight;
    // down segment
    p0.set(pos[0], pos[1], pos[2] + 1500.0);
    q0.set(pos[0], pos[1], pos[2] - 40000.0);


    osg::ref_ptr<osgUtil::IntersectorGroup> igroup = new osgUtil::IntersectorGroup;
    osg::ref_ptr<osgUtil::LineSegmentIntersector> intersector[2];

    intersector[0] = coIntersection::instance()->newIntersector(p0, q0);
    igroup->addIntersector(intersector[0]);

    // down segment 2
    p0.set(pos[0], pos[1] + 1000, pos[2] + 1500.0);
    q0.set(pos[0], pos[1] + 1000, pos[2] - 40000.0);

    intersector[1] = coIntersection::instance()->newIntersector(p0, q0);
    igroup->addIntersector(intersector[1]);


    osg::Vec3 hitPoint[2];
    osg::Vec3 hitNormal[2];

    osgUtil::IntersectionVisitor visitor(igroup);
    visitor.setTraversalMask(Isect::Collision);

    cover->getObjectsXform()->accept(visitor);
    bool hit1 = intersector[0]->containsIntersections();
    bool hit2 = intersector[1]->containsIntersections();
    if (hit1 || hit2)
    {
        if (hit1 || hit2)
        {
            float dist = 0.0;
            osg::Vec3 normal(0, 0, 1);
            osg::Vec3 normal2(0, 0, 1);
            osg::Node *geode = NULL;
            if (hit1 && !hit2)
            {
                auto isect = intersector[0]->getFirstIntersection();
                normal = isect.getWorldIntersectNormal();
                dist = pos[2] - isect.getWorldIntersectPoint()[2];
                geode = *(--isect.nodePath.end());
            }
            else if (!hit1 && hit2)
            {
                auto isect = intersector[1]->getFirstIntersection();
                normal = isect.getWorldIntersectNormal();
                dist = pos[2] - isect.getWorldIntersectPoint()[2];
                geode = *(--isect.nodePath.end());
            }
            else if (hit1 && hit2)
            {

                auto isect1 = intersector[0]->getFirstIntersection();
                auto isect2 = intersector[1]->getFirstIntersection();
                normal = isect1.getWorldIntersectNormal();
                dist = pos[2] - isect1.getWorldIntersectPoint()[2];
                geode = *(--isect1.nodePath.end());

                normal2 = isect2.getWorldIntersectNormal();

                normal += normal2;
                normal *= 0.5;

                if (fabs(pos[2] - isect2.getWorldIntersectPoint()[2]) < fabs(dist))
                {
                    dist = pos[2] - isect2.getWorldIntersectPoint()[2];
                    geode = *(--isect2.nodePath.end());
                }
            }
            osg::Vec3 carNormal(carTransWorld(1, 0), carTransWorld(1, 1), carTransWorld(1, 2));
            tmp.makeTranslate(0, 0, -dist);
            osg::Matrix rMat;
            carNormal.normalize();
            osg::Vec3 upVec(0.0, 0.0, 1.0);
            float sprod = upVec * normal;
            if (sprod < 0)
                normal *= -1;
            sprod = upVec * normal;
            osg::Vec3 yVec(0.0, 1.0, 0.0);
            osg::Vec3 nwc = osg::Matrix::transform3x3(upVec, baseMat);
            nwc.normalize();
            //float s = yVec*nwc;
            float s = -(yVec * normal);
            //fprintf(stderr,"Steigung: %f",s*100000.0);
            if (coVRMSController::instance()->isMaster())
            {
                const float zero = 0.3;
                if (BicyclePlugin::plugin->tacx)
                {
                    float f = zero;
                    if (s < 0)
                    {
                        f += s * 3 * BicyclePlugin::plugin->forceFactor->getValue();
                    }
                    else
                    {
                        f += s * 9 * BicyclePlugin::plugin->forceFactor->getValue();
                    }
                    if (f < 0.)
                        f = 0.;
                    if (f > 1.)
                        f = 1.;


                    BicyclePlugin::plugin->tacx->setForce(f);
                    //cerr << "Slope: " << s << ", Force: " << f << endl;
                }
            }
            if (sprod > 0.8) // only rotate the car if the angle is not more the 45 degrees
            {
                carTrans.postMult(VRMLRotMat * baseMat * tmp * invBaseMat * invVRMLRotMat);
            }
            else
            {
                carTrans.postMult(VRMLRotMat * baseMat * tmp * invBaseMat * invVRMLRotMat);
            }
        }
    }
}

bool BicyclePlugin::init()
{
    fprintf(stderr, "BicyclePlugin::BicyclePlugin\n");
    if (plugin)
        return false;
    plugin = this;
    flightgear = nullptr;
    skateboard = nullptr;
    tacx = nullptr;
    mouse1 = 0;
    mouse2 = 0;

    isPlane=(coCoviseConfig::isOn("COVER.Plugin.Bicycle.FlightGear",false));
    isBike=(coCoviseConfig::isOn("COVER.Plugin.Bicycle.isBike",false));
    isParaglider = (coCoviseConfig::isOn("COVER.Plugin.Bicycle.isParaglider", false));
    isSkateboard = (coCoviseConfig::isOn("COVER.Plugin.Bicycle.isSkateboard", false));
   fprintf(stderr,"isParaglider %d\n",(int)isParaglider);
   fprintf(stderr,"Flightgear %d\n",(int)isPlane);
   fprintf(stderr,"COVER.Plugin.Bicycle.isSkateboard %d\n",(int)isSkateboard);
        if (isSkateboard)
        {
            skateboard = new Skateboard(this);
            skateboard->start();
        }
    if (coVRMSController::instance()->isMaster())
    {
        if (isBike)
        {
            tacx = new Tacx();
        }
        if (isPlane)
        {
            flightgear = new FlightGear(this);
            flightgear->start(); 
        }
        start();
    }



    angleCounter = 0;
    angle = 0;
    speedCounter = 0;
    speed = 0;

    BicycleTab = new coTUITab("Bicycle", coVRTui::instance()->mainFolder->getID());
    BicycleTab->setPos(0, 0);

    velocityFactor = new coTUIEditFloatField("velocity factor", BicycleTab->getID());
    velocityFactor->setEventListener(this);
    velocityFactor->setValue(2.0);
    //velocityFactor->setMin(1.0);
    //velocityFactor->setMax(100.0);
    velocityFactor->setPos(1, 0);

    velocityFactorLabel = new coTUILabel("velocity factor:", BicycleTab->getID());
    velocityFactorLabel->setPos(0, 0);
    
    wingArea = new coTUIEditFloatField("wing area", BicycleTab->getID());
    wingArea ->setEventListener(this);
    wingArea->setValue(70.0);
    wingArea->setPos(1, 2);

    wingAreaLabel = new coTUILabel("wing area:", BicycleTab->getID());
    wingAreaLabel->setPos(0, 2);

    forceFactor = new coTUIEditFloatField("force factor", BicycleTab->getID());
    forceFactor->setEventListener(this);
    forceFactor->setValue(1.0);
    //forceFactor->setMin(0.0);
    //forceFactor->setMax(1.0);
    forceFactor->setPos(1, 1);

    forceFactorLabel = new coTUILabel("force factor:", BicycleTab->getID());
    forceFactorLabel->setPos(0, 1);

    plugin = this;
    std::string Mouse1Dev = coCoviseConfig::getEntry("COVER.Plugin.Bicycle.LenkMaus");
    cout << "Mouse1Dev :" << Mouse1Dev << endl;
    std::string Mouse2Dev = coCoviseConfig::getEntry("COVER.Plugin.Bicycle.PedalDevice");
    cout << "Mouse2Dev  : " << Mouse2Dev << endl;

#ifndef WIN32

    // socket init (tx)

    if ((ssend = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == -1)
    {
        printf("bicyce: socket error\n");
    }

    memset((char *)&si_other, 0, sizeof(si_other));
    si_other.sin_family = AF_INET;
    si_other.sin_port = htons(REMOTE_PORT);
    if (inet_aton(REMOTE_IP, &si_other.sin_addr) == 0)
    {
        printf("bicyce: inet_aton failed\n");
    }

    // socket init (rx)

    s = socket(AF_INET, SOCK_DGRAM, 0);
    if (s < 0)
    {
        printf("Bicycle: Kann Socket nicht öffnen ...(%s)\n",
               strerror(errno));
        exit(EXIT_FAILURE);
    }

    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = htons(LOCAL_SERVER_PORT);
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &y, sizeof(int));
    rc = bind(s, (struct sockaddr *)&servAddr,
              sizeof(servAddr));
    if (rc < 0)
    {
        printf("Bicycle: Kann Portnummern %d nicht binden (%s)\n",
               LOCAL_SERVER_PORT, strerror(errno));
        exit(EXIT_FAILURE);
    }

    const char *message = "getSpeed";

    int iret1;
    iret1 = pthread_create(&thdGetBicycleSpeedIns, NULL, thdGetBicycleSpeed, (void *)message);
    if (iret1)
    {
        fprintf(stderr, "Bicycle: pthread_create() return code: %d\n", iret1);
        exit(EXIT_FAILURE);
    }

    mouse1 = -1;
    mouse2 = -1;
    //wheelcounter = 0;

    if (!Mouse1Dev.empty())
    {
        memset(buffer, 0, 4);
        mouse1 = open(Mouse1Dev.c_str(), O_RDWR | O_NONBLOCK);
        if (mouse1 == -1)
        {
            fprintf(stderr, "Bicycle: could not open Mouse1 device %s\n\n", Mouse1Dev.c_str());
        }
        else
        {

            char ch;
            unsigned char getdevtype = 0xf2, disableps2 = 0xf5, imps2[6] = { 0xf3, 200, 0xf3, 100, 0xf3, 80 }, resetps2 = 0xff;

            fprintf(stderr, "write disable\n");
            ssize_t iret = write(mouse1, &disableps2, 1);
            if (iret != 1)
                fprintf(stderr, "Bicycle: error reading 'disableps2', wrong no. of arguments\n");

            tcflush(mouse1, TCIFLUSH);
            iret = write(mouse1, &getdevtype, 1);
            if (iret != 1)
                fprintf(stderr, "Bicycle: error reading 'getdevtype', wrong no. of arguments\n");
            usleep(1000);
            ch = readps2(mouse1);

            iret = write(mouse1, &resetps2, 1);
            if (iret != 1)
                fprintf(stderr, "Bicycle: error reading 'resetps2', wrong no. of arguments\n");
            usleep(1000);
            ch = readps2(mouse1);
            tcflush(mouse1, TCIFLUSH);
            iret = write(mouse1, &getdevtype, 1);
            if (iret != 1)
                fprintf(stderr, "Bicycle: error reading 'getdevtype', wrong no. of arguments\n");
            usleep(1000);
            ch = readps2(mouse1);

            iret = write(mouse1, imps2, 6);
            if (iret != 6)
                fprintf(stderr, "Bicycle: error reading 'imps2',wrong no. of arguments\n");
        }
    }
    if (!Mouse2Dev.empty())
    {
        memset(buffer, 0, 4);
        mouse2 = open(Mouse2Dev.c_str(), O_RDWR | O_NONBLOCK);
        if (mouse2 == -1)
        {
            fprintf(stderr, "Bicycle: could not open Mouse2 device %s\n\n", Mouse2Dev.c_str());
        }
        else
        {

            char ch;
            unsigned char getdevtype = 0xf2, disableps2 = 0xf5, imps2[6] = { 0xf3, 200, 0xf3, 100, 0xf3, 80 }, resetps2 = 0xff;

            fprintf(stderr, "write disable\n");
            ssize_t iret = write(mouse2, &disableps2, 1);
            if (iret != 1)
                fprintf(stderr, "Bicycle: error reading 'disableps2', wrong no. of arguments\n");

            tcflush(mouse2, TCIFLUSH);
            iret = write(mouse2, &getdevtype, 1);
            if (iret != 1)
                fprintf(stderr, "Bicycle: error reading 'getdevtype', wrong no. of arguments\n");
            usleep(1000);
            ch = readps2(mouse2);

            iret = write(mouse2, &resetps2, 1);
            if (iret != 1)
                fprintf(stderr, "Bicycle: error reading 'resetps2', wrong no. of arguments\n");
            usleep(1000);
            ch = readps2(mouse2);
            tcflush(mouse2, TCIFLUSH);
            iret = write(mouse2, &getdevtype, 1);
            if (iret != 1)
                fprintf(stderr, "Bicycle: error reading 'getdevtype', wrong no. of arguments\n");
            usleep(1000);
            ch = readps2(mouse2);

            iret = write(mouse2, imps2, 6);
            if (iret != 6)
                fprintf(stderr, "Bicycle: error reading 'imps2',wrong no. of arguments\n");
        }
    }
#endif

    if (!Mouse2Dev.empty())
    {
        //AVRInit(Mouse2Dev.c_str(), 9600);
    }
    initUI();

    return true;
}

// this is called if the plugin is removed at runtime
// which currently never happens
BicyclePlugin::~BicyclePlugin()
{
    fprintf(stderr, "BicyclePlugin::~BicyclePlugin\n");
    if (flightgear)
    {
        flightgear->stop();
        delete flightgear;
    }
    if (skateboard)
    {
        skateboard->stop();
        delete skateboard;
    }
    if (mouse1 > 0)
    {
        close(mouse1);
    }
    running = false;
    //AVRClose();
}

int BicyclePlugin::initUI()
{
    VrmlNamespace::addBuiltIn(VrmlNodeBicycle::defineType());
    return 1;
}

void
BicyclePlugin::stop()
{
    doStop = true;
}
void
BicyclePlugin::run()
{
    running = true;
    doStop = false;
    while (running)
    {
        usleep(5000);
        if (tacx)
	{
	    tacx->update();
	}
       /* if (flightgear)
	{
            flightgear->update();
	}*/
    } 
}
void
BicyclePlugin::preFrame()
{
    if (coVRMSController::instance()->isMaster() && (tacx != NULL || flightgear != NULL || skateboard != NULL))
    {
    }
    UpdateInputState();
}

void BicyclePlugin::key(int type, int keySym, int mod)
{
	if (type == 32)
	{
		switch (keySym)
		{
		case 114:
			flightgearReset = true;
			break;
		case 'p':
		case 'P':
			if (flightgear)
			{
				if (flightgear->getPause() == 1.0)
				{
					flightgear->doPause(0.01);
					fprintf(stderr, "Pause\n");
				}
				else
				{
					flightgear->doPause(1.0);
					fprintf(stderr, "Resume\n");
				}
			}
			break;
		case 'u':
		case 'U':
			if (flightgear)
			{
				flightgear->doUp();
				fprintf(stderr, "Up\n");
			}
			break;
		case 'z':
		case 'Z':
			if (skateboard)
			{
				skateboard->Initialize();
			}
			break;
		}
	}

}

COVERPLUGIN(BicyclePlugin)
