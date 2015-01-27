/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_SCALAR_H
#define __GAALET_SCALAR_H

#include "grade.h"
#include "geometric_product.h"

template <class L, class R>
inline gaalet::Grade<0, gaalet::geometric_product<L, R> >
scalar(const gaalet::expression<L> &l, const gaalet::expression<R> &r)
{
    return grade<0>(l * r);
}

#endif
