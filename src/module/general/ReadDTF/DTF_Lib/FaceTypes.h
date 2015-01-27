/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __DTF_FACETYPES_H_
#define __DTF_FACETYPES_H_

#include "../Tools/EnumTypes.h"

using namespace std;

namespace DTF_Lib
{
class FaceTypes : public Tools::EnumTypes
{
    friend class Tools::Singleton<FaceTypes>::InstanceHolder;

private:
    FaceTypes();

    map<int, int> elementLength;

public:
    virtual ~FaceTypes();
};
};
#endif
