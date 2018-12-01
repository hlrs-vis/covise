/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    DataFileBin
//
// Description: Read binary data files (Ensight 6)
//
// Initial version: RM 17.07.002
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef DATAFILEBIN_H
#define DATAFILEBIN_H

#include "EnFile.h"
#include "ReadEnsight.h"

class DataFileBin : public EnFile
{
public:
    /// default CONSTRUCTOR
    DataFileBin(const coModule *mod);
    DataFileBin(const coModule *mod,
                const string &name,
                const int &dim,
                const int &numVals,
                const EnFile::BinType &binType = CBIN);

    void read(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep);

    void readCells(ReadEnsight *ens, dimType dim, coDistributedObject **outObjects, const string &baseName, int &timeStep);

    virtual coDistributedObject *getDataObject(std::string s);

private:
    int dim_;
    int lineCnt_; // actual linecount
    int numVals_; // number of values
    int *indexMap_; // may contain indexMap
};
#endif
