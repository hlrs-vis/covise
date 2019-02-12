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

#ifndef MGEOFILEBIN_H
#define MGEOFILEBIN_H

#include <util/coviseCompat.h>
#include <do/coDoPoints.h>
#ifdef __sgi
using namespace std;
#endif

#include "EnFile.h"
#include "EnElement.h"

//
// read binary Ensight 6 geometry files
//
class En6MGeoBIN : public EnFile
{
public:
    /// default CONSTRUCTOR
    En6MGeoBIN(const coModule *mod);

    // creates file-rep. and opens the file
    En6MGeoBIN(const coModule *mod, const string &name,
               EnFile::BinType binType = EnFile::CBIN);

    // read the file
    void read(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects2d, coDistributedObject **outObjects3d, const string &actObjNm2d, const string &actObjNm3d, int &timeStep, int numTimeSteps);

    // destructor
    virtual ~En6MGeoBIN();

    virtual coDistributedObject *getDataObject(std::string s);

private:
    // read header
    int readHeader();

    // read coordinates
    int readCoords();

    // forwards file pointer i.e. reads data but does nothing with it
    int readCoordsDummy();

    // enter value i to index map (incl realloc)
    void fillIndexMap(const int &i, const int &natIdx);

    int numCoords_; // number of coordinates
    int *indexMap_; // index map array if node id: GIVEN
    int maxIndex_; // max possible  index of indexmap
    coDoPoints *pointObj;

    vector<EnPart> parts_; // contains all parts of the current geometry
};
#endif
