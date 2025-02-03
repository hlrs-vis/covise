/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// CLASS  MEnGoldMPGASC
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

#ifndef MEnGoldMPGASC_H
#define MEnGoldMPGASC_H

#include <util/coviseCompat.h>
#include <do/coDoPoints.h>
#include "EnFile.h"
#include "EnElement.h"
#include "EnPart.h"

// helper strip off spaces
string strip(const string &str);

//
// read Ensight 6 geometry files
//
class MEnGoldMPGASC : public EnFile
{
public:
    /// default CONSTRUCTOR
    MEnGoldMPGASC(ReadEnsight *mod);

    // creates file-rep. and opens the file
    MEnGoldMPGASC(ReadEnsight *mod, const string &name);

    // read the file (Ensight Particle)
    void read(dimType dim, coDistributedObject** outObjects1d, const string& actObjNm1d, int& timeStep, int numTimeSteps);
    /// DESTRUCTOR
    virtual ~MEnGoldMPGASC();

    // helper converts char buf containing num ints of length int_leng to int-array arr
    // -> should better go to EnFile
    static void atoiArr(const int &int_leng, char *buf, int *arr, const int &num);

    virtual coDistributedObject *getDataObject(std::string s);

private:
    // read header (ENSIGHT 6, Gold)
    int readHeader();

    // read coordinates (ENSIGHT 6)
    int readCoords();

    // enter value i to index map (incl realloc)
    void fillIndexMap(const int &i, const int &natIdx);

    int lineCnt_; // actual linecount
    int numCoords_; // number of coordinates
    int *indexMap_; // index map array if node id: GIVEN
    int maxIndex_; // max possible  index of indexmap
    int lastNc_;
    coDoPoints *pointObj;

    //vector<EnPart> parts_; // contains all parts of the current geometry
};
#endif
