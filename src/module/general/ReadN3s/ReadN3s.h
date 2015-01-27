/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_N3S_H
#define _READ_N3S_H
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

#define ProgrammName "Generic ASCII-File Reader for N3S 3.2"

#define Kurzname "ReadN3s"

#define Copyright "(c) 2000 RUS Rechenzentrum der Uni Stuttgart"

#define Autor "M. Wierse (SGI)"

#define letzteAenderung "13.3.2000"

/************************************************************************/

#include <util/coviseCompat.h>
#include <api/coSimpleModule.h>
#include <do/coDoData.h>
#include <do/coDoUnstructuredGrid.h>
using namespace covise;

#ifndef FALSE
#define FALSE 0
#endif

#ifndef TRUE
#define TRUE 1
#endif

const int MAXLINE = 82;

// values for reading coordinates
const int DIGITALS_OF_COORD = 13;

// values for reading values
const int VALUES_IN_LINE = 8;
const int DIGITALS_OF_VALUE = 10;

// values for reading connectivity
const int I2 = 2;
const int I7 = 7;
const int I8 = 8;
const int EDGES_2D = 3;
const int EDGES_3D = 6;

const int MAX_PORTS_N3S = 1;
const int LINES_OF_HEADER = 11;
const int MAX_DATA_COMPONENTS = 25;
const int MAX_TIME_STEPS = 100;

class ReadN3s : public coSimpleModule
{

private:
    //  member functions
    virtual int compute(const char *port);

    int openFiles();
    void readGeoFile();
    void readResultFile();
    void createCoviseUnsgrd();
    void scanResultFile();
    void param(const char *name, bool inMapLoading);
    //  member data
    const char *geofilename; // geo file name
    FILE *fpgeo;
    const char *resfilename; // result file name
    FILE *fpres;

    coOutputPort *unsgridPort, *dataPort[MAX_PORTS];
    coFileBrowserParam *geoFileParam;
    coFileBrowserParam *resFileParam;
    coChoiceParam *choiceData[MAX_PORTS_N3S];

    char infobuf[500]; // buffer for COVISE info and error messages

    int num_nodes, num_elements;
    int num_nodes_P2, dim;
    int *vl, *el, *tl;
    float *x, *y, *z;
    char dnames[MAX_DATA_COMPONENTS][4]; // name of variables stored
    char version[10]; // version of N3S
    char date[10]; // date where data produced
    int dset[MAX_DATA_COMPONENTS]; // if data is set and at which position in the data list
    // position in file (time dependency??)
    int position_in_file[MAX_TIME_STEPS][MAX_DATA_COMPONENTS];
    // first one is 'none'
    char *choice_of_data[MAX_DATA_COMPONENTS + 1];
    // first one is 'none'
    int mapping_of_choice_data[MAX_DATA_COMPONENTS + 1];
    int velocity_components[3]; // position of the velociy components in the data list
    coDoUnstructuredGrid *unsgridObject; // output object for mesh
    char *unsgridObjectName; // output object name assigned by the controller
    coDoVec3 *VdataObject; // output object for vector data
    const char *VdataName; // output object name assigned by the controller
    coDoFloat *SdataObject; // output object for scalar data
    const char *SdataName; // output object name assigned by the controller

public:
    ReadN3s(int argc, char **argv);
    virtual ~ReadN3s();
};
#endif
