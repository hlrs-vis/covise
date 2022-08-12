/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __READTRK_H
#define __READTRK_H

/*=========================================================================
 *   Program:   Covise
 *   Module:    ReadTRK
 *   Language:  C++
 *   Date:      $Date: 2007/09/07 14:17:42 $
 *   Version:   $Revision:  $
 *=========================================================================*/

#include <api/coModule.h>
using namespace covise;
#include <do/coDoGeometry.h>
#include <do/coDoData.h>

using namespace std;
enum TRKdataType {
    DOUBLE = 0,
    VEC3D = 1,
    SINGLE = 2,
    VEC3F = 3,
    UNSIGNED_INT = 4
};

class DataField
{
public:
    std::string name;
    int size=0;
    int type;
    TRKdataType dataType=DOUBLE;
    float* fData = nullptr;
    double* dData = nullptr;
};

class ReadTRK : public coModule
{
private:

    FILE* fp = nullptr;
    int numChunks = 0;
    // ports
    coOutputPort *poLines;

    coOutputPort* poParticleResidenceTime;
    coOutputPort* poVelocity;
    coOutputPort* poParticleFlowRate;
    

    // Parameter & IO:
    coFileBrowserParam * trkFilePath;
    vector< DataField> dataFields;

    vector<coDoFloat*> dosParticleResidenceTime;
    vector<coDoVec3*> dosVelocity;
    vector<coDoFloat*> dosParticleFlowRate;
    vector<coDoLines*> dosLines;

    bool readChunk();
    std::string genName(const std::string& bn);

public:
    ReadTRK(int argc, char *argv[]);
    virtual ~ReadTRK();

    // main-callback
    virtual int compute(const char *port);

};

#endif // __ReadTRK_H
