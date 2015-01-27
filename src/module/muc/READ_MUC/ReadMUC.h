/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_DNW_DATA__OFFLINE_H
#define _READ_DNW_DATA__OFFLINE_H
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
 *	File			ReadTecplot.h	 			*
 *									*
 *	Description		Read Tecplot file 			*
 *									*
 *	Author			Tobias Schweickhardt             	*
 *									*
 *	Date			July 21th, 1997				*
 *									*
 *	Status			finished				*
 *      Modified by Ralph Bruckschen to read in timedependent data	*
 *	with every timestep from different files			*
 *	Date 6. 10. 99							*
 *									*
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
    int isVarName(char *hstr);
    int readFileHeader(FILE *fp, char *Title, int *n, char **VarNames);
    int getOutputObjectNames();
    int getInputParams();
    int readFile();
    int readZoneHeader(DynZoneDescr **curDescr);
    int fnextfloat(long *fpos, float *f);
    int readStructVal(DynZoneElement **curZone, int Var, int Val);
    int readZoneRecord(DynZoneElement **curZone, DynZoneDescr **curDescr);
    void create_time_OutputObjects();
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
    int nZones, nVars, line, usedVars[8], nused, *isused;
    int header_size;
    long fsize;
    long fpos; // letzte Datei-Position
    char *grid_name, *data_name[8];
    char *ugrid_name, *udata_name[8];
    coDoSet *GRID_Set, *DATA_Set[2], *TIME_GRID_Set, *TIME_DATA_Set[3], *VECTOR_Set, *TIME_VECTOR_Set;
    coDoSet *uGRID_Set, *uSCALAR_Set[2], *TIME_uGRID_Set, *TIME_uSCALAR_Set[3], *uVECTOR_Set, *TIME_uVECTOR_Set;
    coDistributedObject **GRID_sets, **DATA_sets[2], **VECTOR_sets;
    coDistributedObject **uGRID_sets, **uSCALAR_sets[2], **uVECTOR_sets;

    coDoStructuredGrid *GRID;
    coDoFloat *DATA;
    coDoVec3 *VECTOR;
    coDoUnstructuredGrid *uGRID;
    coDoFloat *uSCALAR;
    coDoVec3 *uVECTOR;
    int timestep, filesets, current_file;
    int is_timedependent;

public:
    Application(int argc, char *argv[]);
    ~Application();
    void run();
};
#endif // _READ_DNW_DATA_OFFLINE_H
