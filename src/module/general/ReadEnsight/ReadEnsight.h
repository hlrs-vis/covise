/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// MODULE   ReadEnsight
//
// Description: New Technology Ensight read-module
//
// Initial version: 15.04.2002
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef READENSIGHT_H
#define READENSIGHT_H

#include "reader/coReader.h"
#include "reader/ReaderControl.h"
#include "CaseFile.h"
#include "EnFile.h"

#ifdef __sgi
using namespace std;
#endif

const int Success(0);
const int Failure(1);

/////////////////////
const int MAX_LIST_SIZE(500000000);
//////////////////////

class ReadEnsight : public coReader
{
public:
    
    /// default CONSTRUCTOR
    ReadEnsight(int argc, char *argv[]);

    /// DESTRUCTOR
    virtual ~ReadEnsight();

    virtual void param(const char *paramName, bool inMapLoading);

    /// compute call-back
    virtual int compute(const char *port);

    // create distributed objects using parts preserving the part Structure
    // geometry
    coDistributedObject **createGeoOutObj(const string &baseName2d,
        const string &baseName3d, const int &step);
    // data

    coDistributedObject **createDataOutObj(EnFile::dimType dim, const string &baseName,
        DataCont &dc,
        const int &step, const bool &perVertex = true);

    vector<PartList> globalParts_;
private:
    // write a list of parts to the map editor (info channel)
    void createMasterPL();

    // parse parts string given in the partStringParam_
    bool evalPartString(const string &partStr);

    // create file names according to the Ensight file name convention
    vector<string> mkFileNames(const string &baseName, int &realNumTs);

    // read geometry
    int readGeometry(const int &portTok2d, const int &portTok3d3d);

    // read Measured geometry
    int readMGeometry(const int &portTok1d);

    // the new general data read method
    int readData(const int &portTok2d, const int &portTok3d,
                 const string &fileNameBase,
                 const bool &pV, const int &dim = 1, const string &desc = "");

    int readData1d(const int &portTok1d,
                   const string &fileNameBase,
                   const bool &pV, const int &dim = 1, const string &desc = "");

    int readData2d(const int &portTok2d,
                   const string &fileNameBase,
                   const bool &pV, const int &dim = 1, const string &desc = "");

    int readData3d(const int &portTok3d,
                   const string &fileNameBase,
                   const bool &pV, const int &dim = 1, const string &desc = "");

    EnFile *createDataFilePtr(const string &filename, const int &d, const int &numCoord);

    // helper for static geometry / transinet data
    void extendArrays(const int &numTimesteps);

    coDistributedObject *createDataOutObj(const DataCont &dc,
                                          const bool &pv,
                                          const string &name,
                                          const int &dim = 1,
                                          const int &idx = -1);



    void incrRefCnt(const coDistributedObject *obj);

    const coDistributedObject *getGeoObject(const int &step, const int &iPart, const int &dimFlag);

    CaseFile case_;

    vector<int> numCoords_;
    vector<int> numCoordsM_;
    vector<int> numElem_;
    vector<int *> idxMaps_;
    int geoTimesetIdx_;
    bool readGeo_;
    EnFile::BinType binType_;
    EnFile::BinType MbinType_;

    bool dataByteSwap_;

    coBooleanParam *transformToVert_;

    coBooleanParam *repairConnectivity_;

    coBooleanParam *dataByteSwapParam_;

    coBooleanParam *storeGrpParam_;

    coBooleanParam *autoColoring_;

    coStringParam *partStringParam_;

    coBooleanParam *includePolyederParam_;

    PartList masterPL_;

    coDistributedObject *geoObj_;
    coDistributedObject *geoObjs_[3];
};
#endif
