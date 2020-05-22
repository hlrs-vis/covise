/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


 /**************************************************************************\
  **                                                           (C)1995 RUS  **
  **                                                                        **
  ** Description: Read module for Nek5000 data                              **
  **                                                                        **
  **                                                                        **
  **                                                                        **
  **                                                                        **
  **                                                                        **
  ** Author:                                                                **
  **                                                                        **
  **                             Dennis Grieger                             **
  **                                 HLRS                                   **
  **                            Nobelstraße 19                              **
  **                            70550 Stuttgart                             **
  **                                                                        **
  ** Date:  08.07.19  V1.0                                                  **
 \**************************************************************************/
#ifndef _READNEK5000_H
#define _READNEK5000_H

#include <util/coRestraint.h>
#include <api/coModule.h>

#include <vector>
#include <map>
#include <string>
#include <iostream>    
#include <fstream>
#include <cstdio>

namespace covise {
class coDoSet;
}

typedef struct {
    std::string var;
    int         element;
    int         timestep;
} PointerKey;

class     avtIntervalTree;

class KeyCompare {
public:
    bool operator()(const PointerKey& x, const PointerKey& y) const {
        if (x.element != y.element)
            return (x.element > y.element);
        if (x.timestep != y.timestep)
            return (x.timestep > y.timestep);
        return (x.var > y.var);
    }
};

class ReadNek : public covise::coModule {
public:
    int compute(const char* port) override;
    void param(const char* name, bool inMapLoading) override;
private:

    //      This method is called as part of initialization.  It parses the text
//      file which is a companion to the series of .fld files that make up a
//      dataset.
    bool parseMetaDataFile();
//      This method is called as part of initialization.  Some of the file
//      metadata is written in the header of each timestep, and this method
//      reads and parses that metadata.
    bool ParseNekFileHeader();
//      Parses the characters in a binary Nek header file that indicate which
//  fields are present.  Tags can be separated by 0 or more spaces.  Parsing
//  stops when an unknown character is encountered, or when the file pointer
//  moves beyond this->iHeaderSize characters.
//
//  X or X Y Z indicate a mesh.  Ignored here, UpdateCyclesAndTimes picks this up.
//  U indicates velocity
//  P indicates pressure
//  T indicates temperature
//  1 2 3 or S03  indicate 3 other scalar fields
    void ParseFieldTags(std::ifstream& f);

    void ReadCombined();

    //      Gets the mesh associated with this file.  The mesh is returned as a
//      derived type of vtkDataSet (ie vtkRectilinearGrid, vtkStructuredGrid,
//      vtkUnstructuredGrid, etc).
    bool ReadMesh(int timestep, int block, float* x = nullptr, float* y = nullptr, float* z = nullptr);
    //read velocity in x, y and z
    bool ReadVelocity(int timestep, int block, float* x = nullptr, float* y = nullptr, float* z = nullptr);
    //read var with varname in data
    bool ReadVar(const char* varname, int timestep, int block, float* data = nullptr);
    void ReadBlockLocations();
    void UpdateCyclesAndTimes();
//      Create a filename from the template and other info.
    std::string GetFileName(int rawTimestep, int pardir);
    //      If the data is ascii format, there is a certain amount of deprecated
//      data to skip at the beginning of each file.  This method tells where to
//      seek to, to read data for the first block.
    void FindAsciiDataStart(FILE* fd, int& outDataStart, int& outLineLen);

    void makeConectivityList(int* connectivityList, int numBlocks);

    struct DomainParams {
        int domSizeInFloats = 0;
        int varOffsetBinary = 0;
        int varOffsetAscii = 0;
        bool timestepHasMesh = 0;
    };
    DomainParams GetDomainSizeAndVarOffset(int iTimestep, const char* varn);
    template<class T>
    void ByteSwapArray(T* val, int size)
    {
        std::cerr << "ByteSwapArray not implemented" << std::endl;
    }

    // Ports
    covise::coOutputPort* p_grid = nullptr;
    covise::coOutputPort* p_velocity = nullptr;
    covise::coOutputPort* p_pressure = nullptr;
    covise::coOutputPort* p_temperature = nullptr;


    // Parameters
    covise::coFileBrowserParam* p_data_path = nullptr;
    covise::coIntScalarParam* p_partitions = nullptr;
    covise::coIntScalarParam* p_numberOfGrids = nullptr;
    covise::coBooleanParam* p_combineBlocks = nullptr;

    covise::coDoSet* grids = nullptr;
    std::vector<covise::coDoSet*> velocities, pressures;
    covise::coDoSet* velocitiesAllTimes = nullptr, * pressuresAllTime = nullptr;
    float minVeclocity = 0, maxVelocity = 0;
    int numReads = 0; 
    // This info is embedded in the .nek3d text file 
        // originally specified by Dave Bremer
    std::string fileTemplate;
    int iFirstTimestep = 1;
    int iNumTimesteps = 1;
    bool bBinary = false;         //binary or ascii
    int iNumOutputDirs = 0;  //used in parallel format
    bool bParFormat = false;

    int numberOfTimePeriods = 1;
    double gapBetweenTimePeriods = 0.0;

    // This info is embedded in, or derived from, the file header
    bool bSwapEndian = false;
    int iNumBlocks = 0;
    int iBlockSize[3]{ 1,1,1 };
    int iTotalBlockSize = 0;
    bool bHasVelocity = false;
    bool bHasPressure = false;
    bool bHasTemperature = false;
    int iNumSFields = 0;
    int iHeaderSize = 0;
    int iDim = 3;
    int iPrecision = 4; //4 or 8 for float or double
    //only used in parallel binary
    std::vector<int> vBlocksPerFile;
    int iNumberOfRanks = 1;
    // This info is distributed through all the dumps, and only
    // computed on demand
    std::vector<int> aCycles;
    std::vector<double> aTimes;
    std::vector<bool> readTimeInfoFor;
    std::vector<bool> vTimestepsWithMesh;
    int curTimestep = 1;
    int timestepToUseForMesh = 0;

    // Cached data describing how to read data out of the file.
    FILE* fdMesh = nullptr, * fdVar = nullptr;
    std::string  curOpenMeshFile;
    std::string  curOpenVarFile;

    int  iCurrTimestep;        //which timestep is associated with fdVar
    int  iCurrMeshProc;        //For parallel format, proc associated with fdMesh
    int  iCurrVarProc;         //For parallel format, proc associated with fdVar  
    int  iAsciiMeshFileStart;  //For ascii data, file pos where data begins, in mesh file
    int  iAsciiCurrFileStart;  //For ascii data, file pos where data begins, in current timestep
    int  iAsciiMeshFileLineLen; //For ascii data, length of each line, in mesh file
    int  iAsciiCurrFileLineLen; //For ascii data, length of each line, in current timestep

    int* aBlockLocs = nullptr;           //For parallel format, make a table for looking up blocks.
                               //This has 2 ints per block, with proc # and local block #.

    // This info is for managing which blocks are read on which processors
    // and caching blocks that have been read. 
    std::vector<int>                                     myElementList;
    std::map<PointerKey, float*, KeyCompare>            cachedData;
    std::map<int, avtIntervalTree*>                     boundingBoxes;
    std::map<PointerKey, avtIntervalTree*, KeyCompare>  dataExtents;
    int                                                  cachableElementMin;
    int                                                  cachableElementMax;

public:
    ReadNek(int argc, char* argv[]);
    ~ReadNek() override;

    bool makeRandomValues(int timestep, int block);

};

#endif // _READNEK5000_H
