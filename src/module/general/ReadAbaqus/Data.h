/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  CLASS Data
//
//  Abstract class for all data types.
//
//  Initial version: 25.09.2003, Sergio Leseduarte
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//  (C) 2003 by VirCinity IT Consulting
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#ifndef _ABAQUS_DATA_H_
#define _ABAQUS_DATA_H_

#include <util/coviseCompat.h>
#include <api/coModule.h>
using namespace covise;
#include "odb_Enum.h"

class Data
{
public:
    Data();
    virtual ~Data();
    virtual Data *Copy() const = 0;
    virtual Data *Average(const vector<Data *> &other_data) const = 0;
    void SetPosition(odb_Enum::odb_ResultPositionEnum position);
    void SetNode(int node);

    static coDistributedObject *GetObject(const char *name,
                                          const vector<const Data *> &datalist);
    static enum MyValueType
    {
        SCALAR,
        VECTOR,
        TENSOR,
        REFERENCE_SYSTEM,
        UNDEFINED_TYPE
    } TYPE;
    static string SPECIES;
    static string REALTIME;
    static coDistributedObject *GetDummy(const char *name);
    static Data *Invisible();

protected:
    virtual coDistributedObject *GetNoDummy(const char *name,
                                            const vector<const Data *> &datalist) const = 0;
    odb_Enum::odb_ResultPositionEnum _position;
    int _node;

private:
};
#endif
