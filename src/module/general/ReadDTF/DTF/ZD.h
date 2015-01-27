/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __DTF_ZD_H_
#define __DTF_ZD_H_

#include "../DTF_Lib/libif.h"
#include "../DTF_Lib/zone.h"
#include "../DTF_Lib/DataTypes.h"
#include "../DTF_Lib/virtualzone.h"

namespace DTF
{
class ClassInfo_DTFZD;

class ZD : public Tools::BaseObject
{
    friend class ClassInfo_DTFZD;

private:
    static ClassInfo_DTFZD classInfo;

    ZD();
    ZD(string className, int objectID);

    string dataName;
    int dataType;

    map<string, vector<double> > dblData;

public:
    virtual ~ZD();

    bool read(int simNum);
    map<string, vector<double> > getData();
    virtual void clear();
    virtual void print();
};

CLASSINFO(ClassInfo_DTFZD, ZD);
};
#endif
