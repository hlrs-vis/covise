/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    DataFileGold
//
// Description: Abstraction of a Ensight Gold ASCII Data File
//
// Initial version: 07.04.2003
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002/2003 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  Changes:
//

#ifndef DATAFILEGOLDBIN_H
#define DATAFILEGOLDBIN_H

#include "EnFile.h"
#include "GeoFileAsc.h"

class DataFileGoldBin : public EnFile
{
public:
    /// default CONSTRUCTOR
    DataFileGoldBin(const coModule *mod,
                    const string &name,
                    const int &dim,
                    const int &numVals,
                    const EnFile::BinType &binType = CBIN);

    void read(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep, int numTimeSteps);

    void readCells(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep, int numTimeSteps);

    /// DESTRUCTOR
    ~DataFileGoldBin();

private:
    int lineCnt_; // actual linecount
    int numVals_; // number of values
    int *indexMap_; // may contain indexMap
    int actPartIndex_;
};

#endif
