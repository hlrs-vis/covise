/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_OBJ_SIMPLE_H
#define _READ_OBJ_SIMPLE_H
/**************************************************************************\
**                                                   	      (C)1999 RUS **
**                                                                        **
** Description: Interface to Frieder            	                  **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
**                                                                        **
** Author: Uwe Woessner 		                                          **
**                                                                        **
** History:  								  **
** 08-April-99 v1					       		  ** 
**                                                                        **
**                                                                        **
\**************************************************************************/

#include <net/covise_connect.h>
#include <appl/ApplInterface.h>
using namespace covise;
#include <util/DLinkList.h>
#include <util/coMatrix.h>

#define NUMSCALARS 10
#define MAXLAYERS 200
class objectGroup
{
private:
    char *name;

public:
    int numCoords;
    float *xCoords;
    float *yCoords;
    float *zCoords;
    float *scalars[NUMSCALARS];
    int numScalars[NUMSCALARS];
    coMatrix mat;

    const char *getName()
    {
        return name;
    };
    coDistributedObject *makePlygons(const char *objectName);
    coDistributedObject *makeScalarObject(int which, const char *objectName);
    objectGroup(const char *nam);
    ~objectGroup();
};

class objectGroupList : public DLinkList<objectGroup *>
{
public:
    objectGroup *findObject(const char *name);
};

class Application
{

private:
    objectGroupList groups;
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);
    void paramCB(void *callbackData);
    void feedbackCB(int len, const char *data);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);
    static void paramCallback(bool inMapLoading, void *userData, void *callbackData);
    static void feedbackCallback(void *userData, int len, const char *data);

    //  member data
    char *hostname; // obj file name
    FILE *fp;
    const ClientConnection *master = nullptr;
    std::unique_ptr<Host> host;
    int recvInt();
    void recvFloat(float *floatArray, int num);
    void recvInt(int *intArray, int num);
    void recvData();
    char *recvString();
    void readPolygons();
    void readScalars(int which);
    void sendInt(int i);
    void sendString(const char *buf);
    void sendFloat(float *floatArray, int num);
    void sendPolygons();
    void transformPolygons();

public:
    Application(int argc, char *argv[]);
    ~Application();

    void run();
};

#endif
