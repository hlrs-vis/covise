/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_EXPRESSION_H
#define __GAALET_EXPRESSION_H

#include "configuration_list.h"

#include <iostream>
#include <iomanip>

namespace gaalet
{

//Wrapper class for CRTP
template <class E>
struct expression
{
    operator const E &() const
    {
        return *static_cast<const E *>(this);
    }
};

} //end namespace gaalet

#endif
