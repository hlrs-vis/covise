/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	      (C)1999 RUS **
 **                                                                        **
 ** Description: Interface to Frieder	    	          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author: Uwe Woessner		                                          **
 **                                                                        **
 ** History:  								  **
 ** 08-April-99	v1						          **
 **                                                                        **
 **                                                                        **
\**************************************************************************/

//lenght of a line
#define LINE_SIZE 8192

// portion for resizing data
#define CHUNK_SIZE 4096

#include <config/CoviseConfig.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "Frieder.h"
#include <do/coDoSet.h>
#include <do/coDoData.h>
#include <do/coDoPolygons.h>
#include <covise/covise_appproc.h>
#include <net/covise_host.h>

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

void Application::paramCallback(bool /*inMapLoading*/, void *userData, void *callbackData)
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
    ////// Selected new input data file /////////////////////
    if (strcmp("recalc", paramname) == 0)
    {
        if (master)
        {
            transformPolygons();
            sendPolygons();
            sendInt(2); // start calculation
        }
    }
}

// Messages from the Sunface Plugin

void Application::feedbackCB(int /*len*/, const char *data)
{
    if (strcmp(data, "MOVE") == 0)
    {
        const char *layerName = data + 5;
        objectGroup *group = groups.findObject(layerName);
        if (group)
        {
            for (int i = 0; i < 4; ++i)
            {
                for (int j = 0; j < 4; ++j)
                    group->mat.set(i, j, ((float *)(data + 5 + strlen(layerName)))[i * 4 + j]);
            }

            group->mat.print(stderr, (char *)layerName);
            transformPolygons();
            sendPolygons();
            sendInt(2); // start calculation
        }
        else
        {
            cerr << "Layer " << layerName << " not found!" << endl;
        }
    }
    else if (strcmp(data, "RECALC") == 0)
    {
        if (master)
        {
            transformPolygons();
            sendPolygons();
            sendInt(2); // start calculation
        }
    }
}

Application::Application(int argc, char *argv[])
{
    // this info appears in the module setup window
    Covise::set_module_description("Frieder Simulation Interface");

    // the output port
    Covise::add_port(OUTPUT_PORT, "polygons", "coDoPolygons", "geometry polygons");
    // the output port
    Covise::add_port(OUTPUT_PORT, "energy", "coDoFloat", "Energy per face");

    // select the OBJ file name with a file browser
    Covise::add_port(PARIN, "hostName", "String", "Hostname");
    Covise::set_port_default("hostName", "visuw");
    Covise::add_port(PARIN, "recalc", "Boolean", "Start sunface Calculation");
    Covise::set_port_default("recalc", "0");

    // set up the connection to the controller and data manager
    Covise::init(argc, argv);

    // set the quit and the compute callback
    Covise::set_quit_callback(Application::quitCallback, this);
    Covise::set_start_callback(Application::computeCallback, this);
    Covise::set_param_callback(Application::paramCallback, this);
    Covise::set_feedback_callback(Application::feedbackCallback, this);
    master = 0;
}

void Application::run()
{

    ConnectionList *list_of_connections;
    list_of_connections = Covise::appmod->getConnectionList();
    std::string entry = coCoviseConfig::getEntry("value", "Module.Sunface.Hostname", "visuw");
    host.reset(new Host(entry.c_str()));
    while (!(master = list_of_connections->tryAddNewConnectedConn<ClientConnection>(&*host, 31552, 1, (sender_type)1, 10)))
    {
        sleep(1);
    }
    while (1)
    {
        const Connection *conn = list_of_connections->wait_for_input();
        if (conn == (Connection *)master) // handle Messages from Frieder
        {
            fprintf(stderr, "Message from Frieder\n");
            recvData();
        }
        else
        {
            Covise::check_and_handle_event(0.01f); // handle Covise Messages
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

void Application::recvData()
{
    int msgType = -1;
    msgType = recvInt();
    switch (msgType)
    {
    case 1:
        readPolygons();
        break;
    case 2:
    {
        readScalars(0);
        char buf[600];
        sprintf(buf, "E%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
        Covise::set_feedback_info(buf);
        Covise::send_feedback_message("EXEC", "");
    }
    break;
    case -1:
    case -1234567:
    case 0:
    {
        Covise::sendInfo("Connection to Simulation closed");
        ConnectionList *list_of_connections;
        list_of_connections = Covise::appmod->getConnectionList();
        list_of_connections->remove(master);
        master = nullptr;
        std::string entry = coCoviseConfig::getEntry("value", "Module.Sunface.Hostname", "visuw");
        host.reset(new Host(entry.c_str()));
        while (!master)
        {
            master = list_of_connections->tryAddNewConnectedConn<ClientConnection>(&*host, 31552, 1, (sender_type)1, 10);
            if (!master)
            {
                sleep(1);
            }
        }
    }
    break;
    default:
        Covise::sendInfo("Unknown Message Type from Simulation");
    }
}

void Application::readPolygons()
{
    int numGroups = recvInt();
    int i, n;
    for (i = 0; i < numGroups; i++)
    {
        char *name = recvString();
        if (name == NULL)
            return;
        objectGroup *group = groups.findObject(name);
        if (group == NULL)
        {
            group = new objectGroup(name);
            cerr << "new Layer: " << name << endl;
            groups.append(group);
        }
        group->numCoords = recvInt() * 4;
        delete[] group -> xCoords;
        delete[] group -> yCoords;
        delete[] group -> zCoords;
        group->xCoords = new float[group->numCoords];
        group->yCoords = new float[group->numCoords];
        group->zCoords = new float[group->numCoords];
        float *tmpbuf = new float[group->numCoords * 3];
        recvFloat(tmpbuf, group->numCoords * 3);
        for (n = 0; n < group->numCoords; n++)
        {
            group->xCoords[n] = tmpbuf[n * 3];
            group->yCoords[n] = tmpbuf[n * 3 + 1];
            group->zCoords[n] = tmpbuf[n * 3 + 2];
        }
    }
}

void Application::readScalars(int which)
{
    int numGroups = recvInt();
    int i;
    for (i = 0; i < numGroups; i++)
    {
        char *name = recvString();
        if (name == NULL)
            return;
        objectGroup *group = groups.findObject(name);
        if (group == NULL)
        {
            group = new objectGroup(name);
            cerr << "new Layer: " << name << endl;
            groups.append(group);
        }
        group->numScalars[which] = recvInt();
        delete[] group -> scalars[which];
        group->scalars[which] = new float[group->numScalars[which]];
        recvFloat(group->scalars[which], group->numScalars[which]);
    }
}

char *Application::recvString()
{
    int i;
    i = recvInt();
    if ((i < 0) || (i > 10000))
    {
        Covise::sendInfo("Sunface schickt scheisse!");
        return NULL;
    }
    char *buf = new char[i + 1];
    buf[0] = 'X';
    int done = 0, ret;
    while (done < i)
    {
        ret = master->receive(buf, i - done);
        done += ret;
    }
    buf[i] = '\0';
    return buf;
}

void Application::sendString(const char *buf)
{
    int len = (int)strlen(buf);
    sendInt(len);
    if (len)
    {
        master->send(buf, len);
    }
}

void Application::sendPolygons()
{
    int i, n;
    sendInt(1);
    int numGroups = groups.num();
    sendInt(numGroups);
    for (i = 0; i < numGroups; i++)
    {
        objectGroup *group = groups.item(i);
        sendInt(group->numCoords / 4);
        sendString(group->getName());

        float *tmpbuf = new float[group->numCoords * 3];
        for (n = 0; n < group->numCoords; n++)
        {
            tmpbuf[n * 3] = group->xCoords[n];
            tmpbuf[n * 3 + 1] = group->yCoords[n];
            tmpbuf[n * 3 + 2] = group->zCoords[n];
        }
        if (group->numCoords)
            sendFloat(tmpbuf, group->numCoords * 3);
    }
}

void Application::transformPolygons()
{
    coMatrix ident;
    ident.unity();
    int numGroups = groups.num();
    for (int i = 0; i < numGroups; i++)
    {
        objectGroup *group = groups.item(i);
        if (!(group->mat == ident))
        {
            //float *tmpbuf = new float[group->numCoords*3];
            for (int n = 0; n < group->numCoords; n++)
            {
                coVector vec;

                vec[0] = group->xCoords[n];
                vec[1] = group->yCoords[n];
                vec[2] = group->zCoords[n];
                vec = group->mat * vec; //vec.fullXformPt(vec,group->mat);
                group->xCoords[n] = (float)vec[0];
                group->yCoords[n] = (float)vec[1];
                group->zCoords[n] = (float)vec[2];
            }
            group->mat.unity();
        }
    }
}

int Application::recvInt()
{
    int i = 0;
    if ((master->receive(&i, sizeof(int))) < sizeof(int))
        return -1234567;
    swap_int(i);
    return i;
}

void Application::sendInt(int i)
{
    int li = i;
    swap_int(li);
    master->send(&li, sizeof(int));
}

void Application::sendFloat(float *floatArray, int num)
{
    float *lf;
    int i;
    lf = new float[num];
    for (i = 0; i < num; i++)
        lf[i] = floatArray[i];
    swap_float(lf, num);
    master->send(lf, sizeof(float) * num);
    delete[] lf;
}

void Application::recvInt(int *intArray, int num)
{
    int done = 0, ret;
    while (done < num)
    {
        ret = master->receive(intArray + done, sizeof(int) * (num - done));
        done += ret / 4;
    }
    swap_int(intArray, num);
}

void Application::recvFloat(float *floatArray, int num)
{
    int done = 0, ret, i;
    for (i = 0; i < num; i++)
        floatArray[i] = (float)(i + 1);
    while (done < num)
    {
        ret = master->receive(floatArray + done, sizeof(float) * (num - done));
        done += ret / 4;
    }
    swap_float(floatArray, num);
    //for(i=0;i<num;i++)
    //{
    //    fprintf(stderr,"%f\n",floatArray[i]);
    //}
}

void Application::quit(void *)
{
}

void Application::compute(void * /*callbackData*/)
{

    if (groups.num() == 0)
    {
        Covise::sendInfo("No Data received from Simulation");
    }
    char *pObjectName = Covise::get_object_name("polygons");
    char *sObjectName = Covise::get_object_name("energy");

    coDistributedObject **pobjects = new coDistributedObject *[groups.num() + 1];
    coDistributedObject **sobjects = new coDistributedObject *[groups.num() + 1];
    int i = 0;
    groups.reset();
    while (groups.current())
    {
        pobjects[i] = groups.current()->makePlygons(pObjectName);
        sobjects[i] = groups.current()->makeScalarObject(0, sObjectName);
        i++;
        pobjects[i] = NULL;
        sobjects[i] = NULL;
        groups.next();
    }

    coDoSet *pSet = new coDoSet(pObjectName, pobjects);
    coDoSet *sSet = new coDoSet(sObjectName, sobjects);
    pSet->addAttribute("MODULE", "VRSunface");
    Covise::addInteractor(pSet, "Sunface", "Sunface Movable Objects");
    for (i = 0; i < groups.num(); i++)
    {
        delete pobjects[i];
        delete sobjects[i];
    }
    delete pSet;
    delete sSet;

    delete[] pobjects;
    delete[] sobjects;
}

objectGroup::objectGroup(const char *nam)
{
    name = new char[strlen(nam) + 1];
    strcpy(name, nam);
    xCoords = yCoords = zCoords = NULL;
    numCoords = 0;
    int i;
    for (i = 0; i < NUMSCALARS; i++)
    {
        scalars[i] = NULL;
    }
    mat.unity();
}

objectGroup::~objectGroup()
{
    delete[] name;
    delete[] xCoords;
    delete[] yCoords;
    delete[] zCoords;
}

coDistributedObject *objectGroup::makePlygons(const char *objectName)
{
    char *objName = new char[strlen(objectName) + 100];
    sprintf(objName, "%s_P_%s", objectName, name);

    int np = numCoords / 4;
    coDoPolygons *p = new coDoPolygons(objName, numCoords, numCoords, np);

    if (p->objectOk())
    {
        p->addAttribute("vertexOrder", "2");
        char buf[1024];
        sprintf(buf, "SunfaceLayer_%s", name);
        p->addAttribute("OBJECTNAME", buf);
        float *x_coord, *y_coord, *z_coord;
        int *vl, *el, i;
        p->getAddresses(&x_coord, &y_coord, &z_coord, &vl, &el);
        for (i = 0; i < numCoords; i++)
        {
            vl[i] = i;
            x_coord[i] = xCoords[i];
            y_coord[i] = yCoords[i];
            z_coord[i] = zCoords[i];
        }
        for (i = 0; i < np; i++)
        {
            el[i] = i * 4;
        }
    }
    else
    {
        delete p;
        p = NULL;
    }

    delete[] objName;
    return p;
}

coDistributedObject *objectGroup::makeScalarObject(int which, const char *objectName)
{
    char *objName = new char[strlen(objectName) + 100];
    sprintf(objName, "%s_S%d_%s", objectName, which, name);

    coDoFloat *s = new coDoFloat(objName, numScalars[which], scalars[which]);

    if (s->objectOk())
    {
    }
    else
    {
        delete s;
        s = NULL;
    }

    delete[] objName;
    return s;
}

objectGroup *objectGroupList::findObject(const char *name)
{
    reset();
    while (current())
    {
        if (strcmp(current()->getName(), name) == 0)
        {
            return current();
        }
        next();
    }
    return NULL;
}
