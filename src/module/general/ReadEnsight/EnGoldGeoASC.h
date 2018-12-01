/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    EnGoldgeoAsc
//
// Description: Abstraction of Ensight Gold geometry Files
//
// Initial version: 08.04.2003
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 / 2003 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  Changes:
//

#ifndef ENGOLDGEOASC_H
#define ENGOLDGEOASC_H

#include "EnFile.h"

class EnGoldGeoASC : public EnFile
{
public:
    /// default CONSTRUCTOR
    EnGoldGeoASC(const coModule *mod);

    // creates file-rep. and opens the file
    EnGoldGeoASC(const coModule *mod, const string &name);

    /// read the file
    void read(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects2d, coDistributedObject **outObjects3d, const string &actObjNm2d, const string &actObjNm3d, int &timeStep);

    // get part info
    void parseForParts();

    /// DESTRUCTOR
    ~EnGoldGeoASC();

private:
    int allocateMemory();

    // read header
    int readHeader();

    // read Ensight part information
    int readPart(EnPart &actPart);

    // read part connectivities (ENSIGHT Gold only)
    int readPartConn(EnPart &actPart);

    // read bounding box (ENSIGHT Gold)
    int readBB();

    // skip part
    int skipPart();

    int lineCnt_; // actual linecount
    int numCoords_; // number of coordinates
    int *indexMap_; // index map array if node id: GIVEN
    int maxIndex_; // max possible  index of indexmap
    int lastNc_;
    int globalCoordIndexOffset_;
    int actPartNumber_;
    int currElementIdx_;
    int currCornerIdx_;

    vector<EnPart> parts_; // contains all parts of the current geometry

    bool allocated_;
};
#endif
