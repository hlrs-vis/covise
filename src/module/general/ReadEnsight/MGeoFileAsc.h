/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// CLASS  En6MGeoASC
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

#ifndef MGEOFILEASC_H
#define MGEOFILEASC_H

#include <util/coviseCompat.h>
#include <do/coDoPoints.h>
#ifdef __sgi
using namespace std;
#endif

#include "EnFile.h"
#include "EnElement.h"
#include "EnPart.h"

// helper strip off spaces
string strip(const string &str);

//
// read Ensight 6 geometry files
//
class En6MGeoASC : public EnFile
{
public:
    /// default CONSTRUCTOR
    En6MGeoASC(const coModule *mod);

    // creates file-rep. and opens the file
    En6MGeoASC(const coModule *mod, const string &name);

    // read the file (Ensight 6)
    void read(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects2d, coDistributedObject **outObjects3d, const string &actObjNm2d, const string &actObjNm3d, int &timeStep, int numTimeSteps);

    /// DESTRUCTOR
    virtual ~En6MGeoASC();

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

    vector<EnPart> parts_; // contains all parts of the current geometry
};
#endif
