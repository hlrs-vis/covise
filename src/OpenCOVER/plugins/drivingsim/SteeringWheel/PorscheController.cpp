/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "PorscheController.h"
#include "SteeringWheel.h"
#include "net/covise_connect.h"
#include "net/covise_host.h"
#include "net/covise_socket.h"

PorscheController::PorscheController()
{
    port = coCoviseConfig::getInt("port", "COVER.Plugin.SteeringWheel.PorscheServer", 30001);
    serverHost = NULL;
    localHost = new Host("localhost");
    std::string line = coCoviseConfig::getEntry("host", "COVER.Plugin.SteeringWheel.PorscheServer", "192.168.1.24");
    if (!line.empty())
    {
        if (strcasecmp(line.c_str(), "NONE") == 0)
            serverHost = NULL;
        else
            serverHost = new Host(line.c_str());
        cerr << "... on " << serverHost->getName() << endl;
    }

    conn = NULL;

    simulatorJoystick = -1;
    oldSimulatorJoystick = -1;

    numFloatsOut = -1;
    numIntsOut = -1;

    floatValuesOut = new float[60];
    intValuesOut = new int[60];

    floatValues = NULL;
    intValues = NULL;

    gearState1 = 0;
    gearState2 = 0;
    receiveBuffer.steeringWheelAngle = 0;
    receiveBuffer.clutchPedal = 0;
    receiveBuffer.accPedal = 0;
    receiveBuffer.brakePedal = 0;
    receiveBuffer.gearButtonUp = 0;
    receiveBuffer.gearButtonDown = 0;
    receiveBuffer.hornButton = false;
    receiveBuffer.resetButton = false;
    receiveBuffer.mirrorLightLeft = 0;
    receiveBuffer.mirrorLightRight = 0;

    updateRate = coCoviseConfig::getFloat("COVER.Plugin.SteeringWheel.UpdateRate", 0.1);

    doRun = false;
    if (coVRMSController::instance()->isMaster())
    {
        doRun = true;
        Init();
        startThread();
    }
}

PorscheController::~PorscheController()
{
    if (coVRMSController::instance()->isMaster())
    {
        fprintf(stderr, "~Porsche controller waiting!\n");
        doRun = false;
        fprintf(stderr, "waiting1\n");
        endBarrier.block(2); // wait until communication thread finishes
        fprintf(stderr, "done1\n");
    }

    delete localHost;
    delete serverHost;
    delete conn;
}

void PorscheController::update()
{
    if (coVRMSController::instance()->isMaster())
    {
        memcpy(&appReceiveBuffer, &receiveBuffer, sizeof(receiveBuffer));
        coVRMSController::instance()->sendSlaves((char *)&appReceiveBuffer, sizeof(receiveBuffer));
    }
    else
    {
        coVRMSController::instance()->readMaster((char *)&appReceiveBuffer, sizeof(receiveBuffer));
    }

    /*
   if(coVRMSController::instance()->isMaster())
   {
      //cerr << numFloats << endl;
      //cerr << numInts << endl;
      coVRMSController::instance()->sendSlaves((char *)&numFloats,sizeof(int));
      coVRMSController::instance()->sendSlaves((char *)&numInts,sizeof(int));
      if(numFloats)
         coVRMSController::instance()->sendSlaves((char *)floatValues,numFloats*sizeof(float));
      if(numInts)
         coVRMSController::instance()->sendSlaves((char *)intValues,numInts*sizeof(int));
   }
   else
   {
      int newNumFloats=0;
      int newNumInts=0;
      coVRMSController::instance()->readMaster((char *)&newNumFloats,sizeof(int));
      coVRMSController::instance()->readMaster((char *)&newNumInts,sizeof(int));
      //cerr << newNumFloats << endl;
      //cerr << newNumInts << endl;
               if(newNumFloats == 3 && (simulatorJoystick == -1))
               {
                   simulatorJoystick = cover->numJoysticks++;
               fprintf(stderr,"Connected to SimulatorHardware, storing Data as Joystick  %d\n",simulatorJoystick);
               cover->number_buttons[simulatorJoystick]=3;
               cover->number_sliders[simulatorJoystick]=0;
               cover->number_axes[simulatorJoystick]=3;
               cover->number_POVs[simulatorJoystick]=0;
               cover->buttons[simulatorJoystick]=new int[3];
               cover->sliders[simulatorJoystick]=NULL;
               cover->axes[simulatorJoystick]=new float[3];
               cover->POVs[simulatorJoystick]=NULL;
               fd[simulatorJoystick]=-1;
               } 
      if(newNumFloats>0 && newNumFloats != numFloats)
      {
         cerr << "resize" << endl;
         numFloats=newNumFloats;
         delete[] floatValues;
         floatValues = new float[numFloats];
      }
      if(newNumInts > 0 && newNumInts != numInts)
      {
         cerr << "resize" << endl;
         numInts=newNumInts;
         delete[] intValues;
         intValues = new int[numInts];
      }
      if(newNumFloats>0 && numFloats)
      {
         //cerr << "rf" << endl;
         coVRMSController::instance()->readMaster((char *)floatValues,numFloats*sizeof(float));
      }
      if(newNumFloats>0 && numInts)
      {
         //cerr << "ri" << endl;
         coVRMSController::instance()->readMaster((char *)intValues,numInts*sizeof(int));
      }
      if(newNumFloats>0 && numFloats && simulatorJoystick != -1)
      {
         for(int i=0;i<numFloats;i++)
         {
                if(i>=cover->number_axes[simulatorJoystick])
                {
                   cerr << endl << "more floats than axes " << endl;
                   cerr << endl << "simulatorJoystick " << endl;
                   cerr << endl << "numFloats " << endl;
                }
                else
                {
                   cover->axes[simulatorJoystick][i]=floatValues[i];
                }
         }
         for(int i=0;i<(int)(cover->number_buttons[simulatorJoystick]);i++)
         {
            if(intValues[0] & (1 << i))
               cover->buttons[simulatorJoystick][i]=1;
            else
               cover->buttons[simulatorJoystick][i]=0;
         }
         cerr << "Creceived numFloats: " << numFloats << " numInts " << numInts<< endl;
         cerr << "Creceived: ";
         for(int i=0;i<numFloats;i++)
         {
            cerr << floatValues[i] << " ";
         }
         cerr << endl;

         for(int i=0;i<numInts;i++)
         {
            cerr << intValues[i] << " ";
         }
         cerr << endl;
      }
   }
   */
}

void PorscheController::run()
{
    while (doRun)
    {

        if (conn)
        {
            sendValues();
            while (conn && conn->check_for_input())
            {
                int newNumFloats = 3; // should read these numbers from the server!!
                int newNumInts = 3; // should read these numbers from the server!!
                if (!readValues(&newNumFloats, sizeof(int)))
                {
                    delete conn;
                    conn = NULL;
                    newNumFloats = 0;
                    newNumInts = 0;
                    numFloats = 0;
                    numInts = 0;
                    cerr << "reset " << newNumInts << endl;
                }
                if (!readValues(&newNumInts, sizeof(int)))
                {
                    delete conn;
                    conn = NULL;
                    newNumFloats = 0;
                    newNumInts = 0;
                    numFloats = 0;
                    numInts = 0;
                    cerr << "reseti " << newNumInts << endl;
                }
                if (newNumFloats > 0 && newNumFloats != numFloats)
                {
                    numFloats = (int)newNumFloats;
                    delete[] floatValues;
                    floatValues = new float[numFloats];
                }
                if (newNumInts > 0 && newNumInts != numInts)
                {
                    numInts = (int)newNumInts;
                    delete[] intValues;
                    intValues = new int[numInts];
                }
                if (!readValues(floatValues, numFloats * sizeof(float)))
                {
                    delete conn;
                    conn = NULL;
                    newNumFloats = 0;
                    newNumInts = 0;
                    numFloats = 0;
                    numInts = 0;
                    cerr << "reseti2 " << newNumInts << endl;
                }
                if (!readValues(intValues, numInts * sizeof(int)))
                {
                    delete conn;
                    conn = NULL;
                    newNumFloats = 0;
                    newNumInts = 0;
                    numFloats = 0;
                    numInts = 0;
                    cerr << "reseti2 " << newNumInts << endl;
                }
                //int i;
                /*         
            cerr << "received numFloats: " << numFloats << endl;
            for(i=0;i<numFloats;i++)
            {
               cerr << floatValues[i] << " ";
            }
            cerr << endl;
            cerr << "received numInts: " << numInts << endl;
   
            for(i=0;i<numInts;i++)
            {
               cerr << intValues[i] << " ";
            }
            cerr << endl;
         for(int i=0;i<numFloats;i++)
         {
                if(i>=cover->number_axes[simulatorJoystick])
                {
                   cerr << endl << "more floats than axes " << endl;
                   cerr << endl << "simulatorJoystick " << endl;
                   cerr << endl << "numFloats " << endl;
                }
                else
                {
                   cover->axes[simulatorJoystick][i]=floatValues[i];
                }
         }
         for(int i=0;i<(int)(cover->number_buttons[simulatorJoystick]);i++)
         {
            if(intValues[0] & (1 << i))
               cover->buttons[simulatorJoystick][i]=1;
            else
               cover->buttons[simulatorJoystick][i]=0;
         }

            */
                if (numFloats <= 3)
                {
                    receiveBuffer.accPedal = floatValues[1];
                    receiveBuffer.brakePedal = floatValues[2];
                    receiveBuffer.clutchPedal = 0.0;
                    receiveBuffer.steeringWheelAngle = floatValues[0];
                }
                if (numInts <= 1)
                {
                    if (intValues[0] & (1 << 0))
                        receiveBuffer.gearButtonUp = 1;
                    else
                        receiveBuffer.gearButtonUp = 0;
                    if (intValues[0] & (1 << 1))
                        receiveBuffer.gearButtonDown = 1;
                    else
                        receiveBuffer.gearButtonDown = 0;

                    receiveBuffer.hornButton = false;
                    receiveBuffer.resetButton = false;
                    receiveBuffer.resetButton = (intValues[0] & 4);
                }
                if (numInts <= 3)
                {
                    if (intValues[0] & (1 << 0))
                        receiveBuffer.gearButtonUp = 1;
                    else
                        receiveBuffer.gearButtonUp = 0;
                    if (intValues[0] & (1 << 1))
                        receiveBuffer.gearButtonDown = 1;
                    else
                        receiveBuffer.gearButtonDown = 0;

                    receiveBuffer.hornButton = false;
                    receiveBuffer.resetButton = false;
                    receiveBuffer.resetButton = (intValues[0] & 4);

                    receiveBuffer.mirrorLightLeft = intValues[1];
                    receiveBuffer.mirrorLightRight = intValues[2];
                    //std::cout << "mirror :" << intValues[1] << ", " << intValues[2] << std::endl;
                }
            }
        }
        else if ((coVRMSController::instance()->isMaster()) && (serverHost != NULL))
        {
            if ((cover->frameTime() - oldTime) > 2)
            {
                connect();
                oldTime = cover->frameTime();
            }
        }

        usleep(1000);
    }
    fprintf(stderr, "waiting2\n");
    endBarrier.block(2);
    fprintf(stderr, "done2\n");
}

void PorscheController::connect()
{
    // try to connect to server every 2 secnods
    {
        conn = new SimpleClientConnection(serverHost, port, 0);

        if (!conn->is_connected()) // could not open server port
        {
#ifndef _WIN32
            if (errno != ECONNREFUSED)
            {
                fprintf(stderr, "Could not connect to Porsche on %s; port %d\n", serverHost->getName(), port);
                delete serverHost;
                serverHost = NULL;
            }
#endif
            delete conn;
            conn = NULL;
            conn = new SimpleClientConnection(localHost, port, 0);
            if (!conn->is_connected()) // could not open server port
            {
#ifndef _WIN32
                if (errno != ECONNREFUSED)
                {
                    fprintf(stderr, "Could not connect to Porsche on %s; port %d\n", localHost->getName(), port);
                }
#endif
                delete conn;
                conn = NULL;
            }
            else
            {
                fprintf(stderr, "Connected to Porsche on %s; port %d\n", localHost->getName(), port);
            }
        }
        else
        {
            fprintf(stderr, "Connected to Porsche on %s; port %d\n", serverHost->getName(), port);
        }
        if (conn && conn->is_connected())
        {
            int id = 2;
            int ret = conn->getSocket()->read(&id, sizeof(id));
            if (ret < sizeof(id))
            {
                fprintf(stderr, "reading Porsche id failed\n");
            }
            if (id == 1)
            {
                /*
               if(simulatorJoystick == -1)
               {
               simulatorJoystick = cover->numJoysticks++;
               }
               fprintf(stderr,"Connected to SimulatorHardware, storing Data as Joystick  %d\n",simulatorJoystick);
               cover->number_buttons[simulatorJoystick]=3;
               cover->number_sliders[simulatorJoystick]=0;
               cover->number_axes[simulatorJoystick]=3;
               cover->number_POVs[simulatorJoystick]=0;
               cover->buttons[simulatorJoystick]=new int[3];
               cover->sliders[simulatorJoystick]=NULL;
               cover->axes[simulatorJoystick]=new float[3];
               cover->POVs[simulatorJoystick]=NULL;
               fd[simulatorJoystick]=-1;
               */
                std::cerr << "Connected to Porsche Delphi Computer" << std::endl;
            }
        }
    }
}

bool PorscheController::readValues(void *buf, unsigned int numBytes)
{
    unsigned int toRead = numBytes;
    unsigned int numRead = 0;
    int readBytes = 0;
    if (conn == NULL)
        return false;
    while (numRead < numBytes)
    {
        readBytes = conn->getSocket()->Read(((unsigned char *)buf) + readBytes, toRead);
        if (readBytes < 0)
        {
            cerr << "error reading data from socket" << endl;
            return false;
        }
        numRead += readBytes;
        toRead = numBytes - numRead;
    }
    return true;
}

bool PorscheController::sendValues()
{
    if (conn == NULL || numFloatsOut < 0 || numIntsOut < 0)
        return false;
    int written;

    //std::cerr << "Sending: ";

    written = conn->getSocket()->write(&numFloatsOut, sizeof(numFloats));
    if (written < 0)
        return false;
    //for(int i=0; i<sizeof(numFloats); ++i)
    //   std::cerr << (int)(((char*)&numFloatsOut)[i]) << " ";

    written = conn->getSocket()->write(&numIntsOut, sizeof(numInts));
    if (written < 0)
        return false;
    //for(int i=0; i<sizeof(numInts); ++i)
    //   std::cerr << (int)(((char*)&numIntsOut)[i]) << " ";

    if (numFloatsOut > 0)
    {
        written = conn->getSocket()->write(floatValuesOut, numFloatsOut * sizeof(float));
        if (written < numFloatsOut * sizeof(float))
        {
            cerr << "float short write" << endl;
            return false;
        }
    }
    //for(int i=0; i<numFloatsOut*sizeof(float); ++i)
    //   std::cerr << (int)(((char*)floatValuesOut)[i]) << " ";

    if (numIntsOut > 0)
    {
        written = conn->getSocket()->write(intValuesOut, numIntsOut * sizeof(int));
        if (written < numIntsOut * sizeof(int))
        {
            cerr << "integer short write" << endl;
            return false;
        }
    }
    //for(int i=0; i<numIntsOut*sizeof(int); ++i)
    //   std::cerr << (int)(((char*)intValuesOut)[i]) << " ";

    //std::cerr << std::endl;

    return true;
}

double PorscheController::getSteeringWheelAngle()
{
    return appReceiveBuffer.steeringWheelAngle;
}
double PorscheController::getGas()
{
    return appReceiveBuffer.accPedal;
}
double PorscheController::getBrake()
{
    return appReceiveBuffer.brakePedal;
}
double PorscheController::getClutch()
{
    return appReceiveBuffer.clutchPedal;
}
int PorscheController::getGearButtonUp()
{
    return appReceiveBuffer.gearButtonUp;
}
int PorscheController::getGearButtonDown()
{
    return appReceiveBuffer.gearButtonDown;
}
bool PorscheController::getHorn()
{
    return appReceiveBuffer.hornButton;
}
bool PorscheController::getReset()
{
    return appReceiveBuffer.resetButton;
}
int PorscheController::getMirrorLightLeft()
{
    return appReceiveBuffer.mirrorLightLeft;
}
int PorscheController::getMirrorLightRight()
{
    return appReceiveBuffer.mirrorLightRight;
}
