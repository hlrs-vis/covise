/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS  EnGeoFile
// CLASS  En6GeoASC
//
// Description: Ensight 6 geo-file representation
//
// Initial version: 01.06.2002 by RM
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef GEOFILEBIN_H
#define GEOFILEBIN_H

#include <util/coviseCompat.h>
#ifdef __sgi
using namespace std;
#endif

#include "EnFile.h"
#include "EnElement.h"

//
// read binary Ensight 6 geometry files
//
class En6GeoBIN : public EnFile
{
public:
    /// default CONSTRUCTOR
    En6GeoBIN(ReadEnsight *mod);

    // creates file-rep. and opens the file
    En6GeoBIN(ReadEnsight *mod, const string &name,
              EnFile::BinType binType = EnFile::CBIN);

    // read the file
    void read(dimType dim, coDistributedObject **outObjects2d, coDistributedObject **outObjects3d, const string &actObjNm2d, const string &actObjNm3d, int &timeStep, int numTimeSteps);

    // destructor
    virtual ~En6GeoBIN();

    virtual void parseForParts();

private:
    // read header
    int readHeader();

    // read coordinates
    int readCoords();

    // forwards file pointer i.e. reads data but does nothing with it
    int readCoordsDummy();

    // read connectivity
    int readConn();

    // enter value i to index map (incl realloc)
    void fillIndexMap(const int &i, const int &natIdx);

    void checkAllocOffset();

    int numCoords_; // number of coordinates
    int *indexMap_; // index map array if node id: GIVEN
    int maxIndex_; // max possible  index of indexmap

    //vector<EnPart> parts_; // contains all parts of the current geometry
    bool resetAllocInc_;
    int allocOffset_;
    bool debug_;
};
#endif
