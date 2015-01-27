/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_POWER_FLOW_ASCII_H
#define _READ_POWER_FLOW_ASCII_H
/************************************************************************
 *									*
 *          								*
 *                            (C) 2000					*
 *                 VirCinity IY-Consulting GmbH				*
 *                         Nobelstrasse 15				*
 *                        D-70569 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 *	File			ReadPowerFlowASCII.h	 		*
 *									*
 *	Description		Read PowerFlow ASCII TECPLOT file 	*
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
#define BRICK 5
#define QUADRILATERAL 6

    struct DynZoneElement
    {
        DynZoneElement *prev, *next;
        float *value;
        int *connectivity; // for element connectivity
    };
    struct DynZoneDescr
    {
        DynZoneDescr *prev, *next;
        // allg.
        char Title[100], Color[100], DataTypeList[200];
        int Format, ElementType, duplist[MAX_VARS], ndup;
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
    int getTSFilename(int ts);
    int openFile();
    int fnextstring(long *fpos, char *str);
    char *upStr(char *strParam);
    void fileformaterror(char *msg);
    int isVarName(char *hstr);
    int readFileHeader(FILE *fp, char *Title, int *n, char **VarNames);
    int getOutputObjectNames();
    int getInputParams();
    int readFile();
    int readZoneHeader(DynZoneDescr **curDescr);
    int fnextfloat(long *fpos, float *f);
    int fnextint(long *fpos, int *i);
    int readStructVal(DynZoneElement **curZone, int Var, int Val);
    int readElement(int *Var, int no_of_elements);
    int readZoneRecord(DynZoneElement **curZone, DynZoneDescr **curDescr);
    void createOutputObjects();

    //  Static callback stubs
    static void executeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);
    static void paramCallback(void *userData, void *callbackData);

    //  Local data
    char *filename;
    char *base_filename;
    FILE *fp;
    char Title[100], *VarNames[MAX_VARS];
    int preInitOK;
    struct DynZoneElement *data;
    struct DynZoneDescr *zone;
    int nZones, nVars, line, usedVars[8], nused, *isused;
    int header_size;
    int from_ts, to_ts; // for time depedent data sets: start and end timestep
    int time_dependent; // flag: yes/no?
    int current_ts; // index variable
    long fsize;
    long fpos; // letzte Datei-Position
    char *grid_name, *data_name[8];
    coDoSet *GRID_Set, *DATA_Set;
    coDoSet *GRID_TS_Set, *DATA_TS_Set[3];
    coDoStructuredGrid *GRID;
    coDoFloat *DATA;
    coDoVec3 *VDATA;
    coDoUnstructuredGrid *uGRID;
    coDoFloat *uDATA;
    coDoVec3 *uVDATA;

public:
    Application(int argc, char *argv[]);
    ~Application();
    void run();
};
#endif // _READ_POWER_FLOW_ASCII_H
