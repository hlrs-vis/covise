/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	      (C)1999 RUS **
 **                                                                        **
 ** Description: Interface to Matthias Muellers Simulation    	          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner		                                          **
 **                                                                        **
 ** History:  								  **
 ** 27-Sep-00	v1						          **
 **                                                                        **
 **                                                                        **
\**************************************************************************/
#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

//lenght of a line
#define CONFIG_MSG 1
#define GRID_MSG 2
#define SCALAR_MSG 3
#define VECTOR_MSG 4
#define EXEC_MSG 10
#define COVISE_SEND_DATA 11
#define COVISE_SET_VEL 12

#include <config/CoviseConfig.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "mattDSMC.h"

inline void swap_int(int &d)
{
    unsigned int &data = (unsigned int &)d;
    data = ((data & 0xff000000) >> 24)
           | ((data & 0x00ff0000) >> 8)
           | ((data & 0x0000ff00) << 8)
           | ((data & 0x000000ff) << 24);
}

inline void swap_int(int *d, int num)
{
    unsigned int *data = (unsigned int *)d;
    int i;
    //fprintf(stderr,"swapping %d integers\n", num);
    for (i = 0; i < num; i++)
    {
        //fprintf(stderr,"data=%d\n", *data);

        *data = (((*data) & 0xff000000) >> 24)
                | (((*data) & 0x00ff0000) >> 8)
                | (((*data) & 0x0000ff00) << 8)
                | (((*data) & 0x000000ff) << 24);
        //fprintf(stderr,"data=%d\n", *data);
        data++;
    }
}

inline void swap_float(float *d, int num)
{
    unsigned int *data = (unsigned int *)d;
    int i;
    for (i = 0; i < num; i++)
    {
        *data = (((*data) & 0xff000000) >> 24)
                | (((*data) & 0x00ff0000) >> 8)
                | (((*data) & 0x0000ff00) << 8)
                | (((*data) & 0x000000ff) << 24);
        data++;
    }
}

int main(int argc, char *argv[])
{
    Application *application = new Application(argc, argv);
    application->run();

    return 0;
}

void Application::paramCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->paramCB(callbackData);
}

void Application::feedbackCallback(void *userData, int len, const char *buf)
{
    Application *thisApp = (Application *)userData;
    thisApp->feedbackCB(len, buf);
}

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
/////
/////            I M M E D I A T E   C A L L B A C K
/////
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

void Application::paramCB(void *)
{
    const char *paramname = Covise::get_reply_param_name();
    static int firsttime = 1;
    if (firsttime)
    {
        firsttime = 0;
        return;
    }
    cerr << "testit" << paramname << endl;
    if (strcmp("recalc", paramname) == 0)
    {
        if (toSimulation)
        {
            cerr << "requesting new data" << endl;
            sendInt(COVISE_SEND_DATA);
        }
    }
}

// Messages from the Sunface Plugin

void Application::feedbackCB(int len, const char *data)
{
    if (strcmp(data, "MOVE") == 0)
    {
        const char *layerName = data + 5;
    }
    else if (strcmp(data, "RECALC") == 0)
    {
        if (toSimulation)
        {
        }
    }
}

Application::Application(int argc, char *argv[])
{
    // this info appears in the module setup window
    Covise::set_module_description("Matthias Mueller Simulation Interface");

    // the output port
    Covise::add_port(OUTPUT_PORT, "grid", "coDoUniformGrid", "the Grid");
    // the output port
    Covise::add_port(OUTPUT_PORT, "density", "coDoFloat", "Density per node");
    Covise::add_port(OUTPUT_PORT, "velocity", "coDoVec3", "Velocity per node");

    // select the OBJ file name with a file browser
    Covise::add_port(PARIN, "hostName", "String", "Hostname");
    Covise::set_port_default("hostName", "hwwt3e.hww.de");
    Covise::add_port(PARIN, "recalc", "Boolean", "Start Calculation");
    Covise::set_port_default("recalc", "0");
    Covise::set_port_immediate("recalc", 1);

    // set up the connection to the controller and data manager
    Covise::init(argc, argv);

    // set the quit and the compute callback
    Covise::set_quit_callback(Application::quitCallback, this);
    Covise::set_start_callback(Application::computeCallback, this);
    Covise::set_param_callback(Application::paramCallback, this);
    Covise::set_feedback_callback(Application::feedbackCallback, this);
    toSimulation = NULL;
    serverSocket = NULL;
    numU = 0;
    byteSwap = 0;
    densityValues = NULL;
    vxValues = NULL;
    vyValues = NULL;
    vzValues = NULL;
}

void Application::run()
{

    ConnectionList *list_of_connections;
    list_of_connections = Covise::appmod->getConnectionList();
    while ((!serverSocket) || (!serverSocket->is_connected()))
    {
        int serverPort = coCoviseConfig::getInt("Module.MattDSMC.Port", 31553);
        if (serverPort < 100)
            serverPort = 31553;
        serverSocket = new ServerConnection(serverPort, 1, (sender_type)1);
        fprintf(stderr, "Opening server Socket on port %d\n", serverPort);
        serverSocket->listen();
        if ((!serverSocket) || (!serverSocket->is_connected()))
        {
            sleep(1);
            fprintf(stderr, "Could not open server Socket on port %d\n", serverPort);
            delete serverSocket;
            serverSocket = 0;
        }
    }
    if (serverSocket)
    {
        list_of_connections->add(serverSocket);
    }
    fprintf(stderr, "Done Opening server Socket\n");
    Connection *conn;
    while (1)
    {
        conn = list_of_connections->wait_for_input();
        if (conn == (Connection *)toSimulation) // handle Messages from Matthias
        {
            recvData();
        }
        else if (conn == (Connection *)serverSocket) // connection on server port
        {
            fprintf(stderr, "try to connect to client\n");
            toSimulation = serverSocket->spawn_connection();
            fprintf(stderr, "connection to client established\n");
            // check for endiness (if i receive 1, then we do not have to swap)
            int i;
            if ((toSimulation->receive(&i, sizeof(int))) < sizeof(int))
            {
                fprintf(stderr, "error receiving endianess information\n");
                delete toSimulation;
                toSimulation = NULL;
            }
            else
            {
                if (i != 1)
                {
                    byteSwap = 1;
                    cerr << "swaping bytes" << endl;
                    swap_int(i);

                    if (i != 1)
                    {
                        cerr << "oops, got " << i << " or ";
                        swap_int(i);
                        cerr << " orig: " << i << endl;
                        unsigned char *c = (unsigned char *)&i;
                        cerr << "Bytes: " << (int)c[0] << " " << (int)c[1] << " " << (int)c[2] << " " << (int)c[3] << endl;
                    }
                }
                list_of_connections->add(toSimulation);
            }
        }
        else
        {
            Covise::check_and_handle_event(0.01); // handle Covise Messages
        }
    }
}

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

void Application::selfExec()
{
    char buf[1000];
    sprintf(buf, "E%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
    Covise::set_feedback_info(buf);
    Covise::send_feedback_message("EXEC", "");
}

void Application::recvData()
{
    int msgType = -1;
    msgType = recvInt();
    switch (msgType)
    {
    case CONFIG_MSG:
        break;
    case GRID_MSG:
    {
        cerr << "reading grid information" << endl;
        readGrid();
    }
    break;
    case SCALAR_MSG:
    {
        cerr << "reading scalar datavalues" << endl;
        readScalars(densityValues);
    }
    break;
    case VECTOR_MSG:
    {
        readScalars(vxValues);
        readScalars(vyValues);
        readScalars(vzValues);
    }
    break;
    case EXEC_MSG:
    {
        selfExec();
    }
    case COVISE_SET_VEL:
    {
    }
    break;
    case -1:
    case -1234567:
    case 0:
    {
        Covise::sendInfo("Connection to Simulation closed");
        ConnectionList *list_of_connections;
        list_of_connections = Covise::appmod->getConnectionList();
        list_of_connections->remove(toSimulation);
        toSimulation = NULL;
    }
    break;
    default:
        Covise::sendInfo("Unknown Message Type from Simulation");
    }
}

void Application::readGrid()
{
    blockSizeX = recvInt();
    blockSizeY = recvInt();
    blockSizeZ = recvInt();
    cpuVecX = recvInt();
    cpuVecY = recvInt();
    cpuVecZ = recvInt();
    numU = blockSizeX * cpuVecX;
    numV = blockSizeY * cpuVecY;
    numW = blockSizeZ * cpuVecZ;
    recvFloat(gridLL, 3);
    recvFloat(gridUR, 3);
    //gridSize[0] = 1.0;
    //gridSize[1] = 1.0;
    //gridSize[2] = 1.0;
    cerr << "blockSizeX: " << blockSizeX << endl;
    cerr << "blockSizeY: " << blockSizeY << endl;
    cerr << "blockSizeZ: " << blockSizeZ << endl;
    cerr << "cpuVecX: " << cpuVecX << endl;
    cerr << "cpuVecY: " << cpuVecY << endl;
    cerr << "cpuVecZ: " << cpuVecZ << endl;
    tmpValues = new float[blockSizeX * blockSizeY * blockSizeZ];
    densityValues = new float[numU * numV * numW];
    vxValues = new float[numU * numV * numW];
    vyValues = new float[numU * numV * numW];
    vzValues = new float[numU * numV * numW];
}

void Application::readScalars(float *values)
{
    int numFloats = blockSizeX * blockSizeY * blockSizeZ;
    int i, j, k, bx, by, bz;
    for (i = 0; i < cpuVecX; i++)
    {
        for (j = 0; j < cpuVecY; j++)
        {
            for (k = 0; k < cpuVecZ; k++)
            {
                cerr << "reading " << numFloats << " floats" << endl;
                recvFloat(tmpValues, numFloats);
                cerr << "done reading " << numFloats << " floats" << endl;
                for (bz = 0; bz < blockSizeZ; bz++)
                {
                    for (by = 0; by < blockSizeY; by++)
                    {
                        for (bx = 0; bx < blockSizeX; bx++)
                        {
                            int index = i * (cpuVecZ * cpuVecY * blockSizeY * blockSizeZ * blockSizeX) + j * blockSizeY * (cpuVecZ * blockSizeZ) + k * blockSizeZ + (cpuVecZ * cpuVecY * blockSizeZ * blockSizeY) * bx + (cpuVecZ * blockSizeZ) * by + bz;
                            if (index >= numFloats * cpuVecX * cpuVecY * cpuVecZ)
                            {
                                cerr << "scheisse... " << endl;
                            }
                            else
                                values[index] = tmpValues[(blockSizeZ * blockSizeY) * bx + (blockSizeZ)*by + bz];
                            if ((blockSizeZ * blockSizeY) * bx + (blockSizeZ)*by + bz >= blockSizeZ * blockSizeY * blockSizeX)
                            {
                                cerr << "Matthias bekommt ein Mittagessen von mir " << endl;
                            }
                        }
                    }
                }
            }
        }
    }
}

char *Application::recvString()
{
    int i;
    i = recvInt();
    if ((i < 0) || (i > 10000))
    {
        Covise::sendInfo("Matthias schickt scheisse!");
        return NULL;
    }
    char *buf = new char[i + 1];
    buf[0] = 'X';
    int done = 0, ret;
    while (done < i)
    {
        ret = toSimulation->receive(buf, i - done);
        done += ret;
    }
    buf[i] = '\0';
    return buf;
}

void Application::sendString(const char *buf)
{
    int len = strlen(buf);
    sendInt(len);
    if (len)
    {
        toSimulation->send(buf, len);
    }
}

int Application::recvInt()
{
    int i = 0;
    if ((toSimulation->receive(&i, sizeof(int))) < sizeof(int))
        return -1234567;
    if (byteSwap)
        swap_int(i);
    return i;
}

void Application::sendInt(int i)
{
    int li = i;
    if (byteSwap)
        swap_int(li);
    toSimulation->send(&li, sizeof(int));
}

void Application::sendFloat(float *floatArray, int num)
{
    float *lf;
    int i;
    lf = new float[num];
    for (i = 0; i < num; i++)
        lf[i] = floatArray[i];
    if (byteSwap)
        swap_float(lf, num);
    toSimulation->send(lf, sizeof(float) * num);
    delete[] lf;
}

void Application::recvInt(int *intArray, int num)
{
    int done = 0, ret;
    while (done < num)
    {
        ret = toSimulation->receive(intArray + done, sizeof(int) * (num - done));
        done += ret / 4;
    }
    if (byteSwap)
        swap_int(intArray, num);
}

void Application::recvFloat(float *floatArray, int num)
{
    int done = 0, ret;
    while (done < num * sizeof(float))
    {
        ret = toSimulation->receive(((char *)floatArray) + done, (sizeof(float) * num) - done);
        done += ret;
    }
    if (byteSwap)
        swap_float(floatArray, num);
}

void Application::quit(void *)
{
}

void Application::compute(void *callbackData)
{

    if (numU == 0)
    {
        Covise::sendInfo("No Data received from Simulation");
        return;
    }
    char *gObjectName = Covise::get_object_name("grid");
    char *dObjectName = Covise::get_object_name("density");
    char *vObjectName = Covise::get_object_name("velocity");

    coDoUniformGrid *gridObject = new coDoUniformGrid(gObjectName, numU, numV, numW, gridLL[0], gridUR[0], gridLL[1], gridUR[1], gridLL[2], gridUR[2]);
    delete gridObject;
    coDoFloat *densityObject = new coDoFloat(dObjectName, numU, numV, numW, densityValues);
    delete densityObject;
    coDoVec3 *velocityObject = new coDoVec3(vObjectName, numU, numV, numW, vxValues, vyValues, vzValues);
    delete velocityObject;
}
