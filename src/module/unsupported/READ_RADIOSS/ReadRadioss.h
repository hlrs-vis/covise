/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_RADIOSS_H
#define _READ_RADIOSS_H
/************************************************************************
 *									*
 *          								*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30a				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 ************************************************************************/

#define ProgrammName "Generic ASCII-File Reader for Radioss"

#define Kurzname "ReadRadioss"

#define Copyright "(c) 2000 RUS Rechenzentrum der Uni Stuttgart"

#define Autor "M. Wierse (SGI)"

#define letzteAenderung "23.4.2000"

/************************************************************************/

#include <api/coModule.h>
using namespace covise;

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

const int MAXLINE = 1000;

// values for reading connectivity
const int I2 = 2;
const int I7 = 7;
const int I8 = 8;
const int E16 = 16;
const int E12 = 12;
const int MAX_DATA_COMPONENTS = 25;
const int MAX_PORTS_RADIOSS = 4;

int split[6][4] = {
    { 0, 1, 2, 3 },
    { 1, 5, 6, 2 },
    { 4, 5, 6, 7 },
    { 0, 4, 7, 3 },
    { 0, 1, 5, 4 },
    { 3, 2, 6, 7 }
};

typedef struct Polygonstmp // space to fill in the polygons of the different subsets
{
    int num_points, num_polygons, num_corners;
    float *x, *y, *z;
    int *corner_list, *polygon_list;
} polygonstmp;

typedef struct Linestmp // space to fill in the lines of the different subsets
{
    int num_points, num_lines, num_corners;
    float *x, *y, *z;
    int *corner_list, *line_list;
} linestmp;

class ReadRadioss : public coModule
{

private:
    //  member functions
    virtual int compute(void);
    virtual void quit(void);

    int openFiles();
    void readGeoFile();
    void readInputFile();
    void readResultFile();
    void createCovisePolygon(int);
    void createCovisePolygonShell3nQuad(int);
    void createCoviseLine(int);
    void param(const char *name);
    void readHead();
    void readControl();
    void readCoordinates();
    void readSolids();
    void readQuad();
    void readShell();
    void readTruss();
    void readBeam();
    void readSpring();
    void readShell3n();
    void createPolygon();
    void createPolygonShell3nQuad();
    void createLine();
    int compressNodes(int, int *, float **, float **, float **);
    void splitSolid(int *);
    int getLastI8(); // Reading of formatted ASCII (fortran I8) data
    float getLastE16(); // Reading of formatted ASCII (fortran E16) data
    float getLastE12(); // Reading of formatted ASCII (fortran E12) data
    void getDisplacements();
    void getScalarData(int, int, int);
    void createScalarDataSubsets();
    void createDisplacementsDataSubsets(int);
    void createCoviseV3D(int);
    void createCoviseUnsgrid();
    //  member data: File handling
    const char *geofilename; // geo file name
    FILE *fpgeo;
    const char *inputfilename; // input file name for Radioss
    FILE *fpinput;
    const char *resfilename; // result file name
    FILE *fpres;

    //  member data: covise variables
    coOutputPort *polygonsPort, *unsgridPort, *linesPort, *dataPort[MAX_PORTS], *polygonsShell3nQuadPort;
    coFileBrowserParam *geoFileParam;
    coFileBrowserParam *inputFileParam;
    coFileBrowserParam *resFileParam;
    coChoiceParam *choiceData[MAX_PORTS_RADIOSS];

    // output object for mesh
    coDoPolygons *polygonsObject, *polygonsShell3nQuadObject;
    // output object name assigned by the controller
    char *polygonsObjectName, *polygonsShell3nQuadObjectName;
    coDoLines *linesObject; // output object for mesh
    char *linesObjectName; // output object name assigned by the controller
    coDoVec3 *VdataObject; // output object for vector data
    const char *VdataName; // output object name assigned by the controller
    coDoFloat *SdataObject; // output object for scalar data
    const char *SdataName; // output object name assigned by the controller
    polygonstmp polys;
    linestmp lines;
    float *displacementsfield[3];

    //  member data: auxiliary variables
    char infobuf[500]; // buffer for COVISE info and error messages
    char buffer[MAXLINE]; // line in an obj file
    int *global_to_local, *local_to_global; // mapping of nodes to compress version
    float *Data[MAX_PORTS_RADIOSS + 2];

    //  member data: variables for input data
    int nummid, numpid, numnod;
    int numsol, numquad, numshel, numtrus, numbeam, numspri, numsh3n;
    float *xglobal, *yglobal, *zglobal;
    char date[11]; // date where data produced
    char casename[20];
    int *mapping_propid_to_subset, num_subsets, *collect_subsetids;
    struct Solids
    {
        int sysnod[8];
        int syssol, usrsol, sysmid, syspid;
    } solids;
    Solids *solidslist;

    struct Quads
    {
        int sysnod[4];
        int syspid, usrquad, sysmid, sysquad;
    } quads;
    Quads *quadslist;

    struct Shells
    {
        int sysnod[4];
        int sysshel, usrshel, sysmid, syspid;
    } shells;
    Shells *shellslist;

    struct Trusss
    {
        int sysnod[2];
        int systrus, usrtrus, sysmid, syspid;
    } trusss;
    Trusss *trussslist;

    struct Beams
    {
        int sysnod[2];
        int sysbeam, usrbeam, sysmid, syspid;
    } beams;
    Beams *beamslist;

    struct Springs
    {
        int sysnod[2];
        int sysspri, usrspri, sysmid, syspid;
    } springs;
    Springs *springslist;

    struct Shell3ns
    {
        int sysnod[3];
        int syssh3n, usrsh3n, sysmid, syspid;
    } shell3ns;
    Shell3ns *shell3nslist;

public:
    ReadRadioss();
    virtual ~ReadRadioss();
};
#endif
