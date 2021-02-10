/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef READ_OBJ_MATT_DSMC_SIMPLE_H_SIMPLE_H
#define READ_OBJ_MATT_DSMC_SIMPLE_H_SIMPLE_H
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

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/DLinkList.h>

class Application
{

private:
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
    const ServerConnection *serverSocket;
    const ServerConnection *toSimulation;
    int numU, numV, numW;
    int gridSizeX;
    int gridSizeY;
    int gridSizeZ;
    int cpuVecX;
    int cpuVecY;
    int cpuVecZ;
    int byteSwap;
    float gridLL[3];
    float gridUR[3];
    float *tmpValues;
    float *densityValues;
    float *vxValues;
    float *vyValues;
    float *vzValues;

    int recvInt();
    void selfExec();
    void recvFloat(float *floatArray, int num);
    void recvInt(int *intArray, int num);
    void recvData();
    char *recvString();
    void readGrid();
    void readScalars(float *values);
    void sendInt(int i);
    void sendString(const char *buf);
    void sendFloat(float *floatArray, int num);

public:
    Application(int argc, char *argv[]);
    ~Application();

    void run();
};

#endif
