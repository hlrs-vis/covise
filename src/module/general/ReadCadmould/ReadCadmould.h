/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READ_ELMER_H
#define _READ_ELMER_H

#include <do/coDoSet.h>
#include <api/coModule.h>
using namespace covise;
#include <string>

class CadmouldGrid;
class CadmouldData;

class ReadCadmould : public coModule
{
public:
    //c'tor + d'tor
    ReadCadmould(int argc, char *argv[]);
    virtual ~ReadCadmould();

private:
    enum
    {
        NUM_PORTS = 8,
        MAX_DATASETS = 1024
    };
    enum
    {
        TYPE_CAR,
        TYPE_STD
    } d_type;

    //    member functions
    virtual int compute(const char *port);
    virtual void postInst();
    void param(const char *paramName, bool inMapLoading);
    void createParam();
    void readGrid();
    void readThick();
    void readData(coOutputPort *port, int choice);
    void readData(coOutputPort *port, int useDataSet, int useField);
    void fillingParams();
    int fillingAnimation();
    void getMinMax(const coDistributedObject *fillTime, float &min, float &max);
    coDistributedObject *FillStep(const std::string &stepName,
                                  float realtime, float max,
                                  const coDistributedObject *ogrid, const coDistributedObject *fillTime);

    // Check required files: return false if filename is not valid
    bool checkFiles(const std::string &filename);
    /// set d_type according to input filename
    void setCaseType(const std::string &filename);
    void openFiles();
    int retOpenFile;

    coDoSet *readField(const std::string &objName, CadmouldData *data,
                       int field, int step);

    // ports and param
    coOutputPort *p_mesh, *p_fillMesh, *p_fillData, *p_stepMesh, *p_thick;
    coFileBrowserParam *p_filename;
    coOutputPort *p_data[NUM_PORTS];
    coChoiceParam *p_choice[NUM_PORTS];
    coChoiceParam *p_fillField;

    coIntScalarParam *p_no_time_steps;
    coStringParam *p_no_data_color;
    //      coBooleanParam *p_byteswap;
    bool byteswap_;
    int fillChoice;
    int fillTimeDataSets; // data set where fill_time is found
    int fillField;
    coOutputPort *whereisFillTime;

    // Cadmould mesh and data objects
    CadmouldGrid *d_grid;
    CadmouldData *d_data[MAX_DATASETS];
    int d_numDataSets;

    // map label no to data set / field
    struct
    {
        int datasetNo, fieldNo;
    } dataLoc[6 * MAX_DATASETS];

    //      void FlipState(coBooleanParam*);
};
#endif
