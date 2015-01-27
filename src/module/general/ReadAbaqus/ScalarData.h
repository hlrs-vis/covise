/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS ScalarData
//
//  Scalar data representation.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _ABAQUS_SCALAR_DATA_H_
#define _ABAQUS_SCALAR_DATA_H_

#include "Data.h"

class ScalarData : public Data
{
public:
    ScalarData(float scalar);
    virtual ~ScalarData();
    virtual Data *Copy() const;
    virtual Data *Average(const vector<Data *> &other_data) const;

protected:
    virtual coDistributedObject *GetNoDummy(const char *name,
                                            const vector<const Data *> &datalist) const;

private:
    float _scalar;
};
#endif
