/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_FLOWER_H
#define _READ_FLOWER_H
/************************************************************************
 *									*
 *          								*
 *                            (C) 1997					*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30a				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			ReadFlower.h	 			*
 *									*
 *	Description		Read FLowet file 			*
 *									*
 *	Author			Tobias Schweickhardt             	*
 *									*
 *	Date			July 21th, 1997				*
 *									*
 *	Status			finished				*
 *									*
 ************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <stdlib.h>
#include <stdio.h>

#define MAX_VARS 50

class Application
{

private:
#define POINT 1
#define BLOCK 2
#define FEPOINT 3
#define FEBLOCK 4
    struct DynZoneElement
    {
        DynZoneElement *prev, *next;
        float *value;
    };
    struct DynZoneDescr
    {
        DynZoneDescr *prev, *next;
        // allg.
        char Title[100], Color[100], DataTypeList[200];
        int Format, duplist[MAX_VARS], ndup;
        int i, j, k; // fuer structured Zones
        int n, e, nv, et; // fuer FE Zones
    };

    //  member functions
    void paramChange(void *callbackData);
    void execute(void *callbackData);
    void quit(void *callbackData);

    void init_Vars();
    void delete_chain(DynZoneElement *element);
    void delete_chain(DynZoneDescr *element);
    void delete_data();
    int getFilename();
    int openFile();
    int fnextstring(long *fpos, char *str);
    char *upStr(char *strParam);
    void fileformaterror(char *msg);
    int readFileHeader(FILE *fp, char *Title, int *n, char **VarNames);
    int getOutputObjectNames();
    int getInputParams();
    int readFile();
    int readZoneHeader(DynZoneDescr **curDescr);
    int fnextfloat(long *fpos, float *f);
    int readStructVal(DynZoneElement **curZone, int Var, int Val);
    int readZoneRecord(DynZoneElement **curZone, DynZoneDescr **curDescr);
    void createOutputObjects();

    //  Static callback stubs
    static void executeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);
    static void paramCallback(void *userData, void *callbackData);

    //  Local data
    char *filename;
    FILE *fp;
    char Title[100], *VarNames[MAX_VARS];
    int preInitOK;
    struct DynZoneElement *data;
    struct DynZoneDescr *zone;
    int nZones, nVars, line, usedVars[6], nused, *isused;
    int header_size;
    long fsize;
    long fpos; // letzte Datei-Position
    char *grid_name, *data_name[6];
    coDoSet *GRID_Set, *DATA_Set;
    coDoStructuredGrid *GRID;
    coDoFloat *DATA;
    coDoUnstructuredGrid *uGRID;
    coDoFloat *uDATA;

    enum // sl: 10 is the originally assumed value
    {
        MAX_N_VARS = 10
    };

public:
    Application(int argc, char *argv[]);
    ~Application();
    void run();
};
#endif // _READ_DNW_DATA_OFFLINE_H
