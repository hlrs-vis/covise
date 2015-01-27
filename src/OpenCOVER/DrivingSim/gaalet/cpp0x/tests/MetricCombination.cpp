/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

typedef gaalet::algebra<gaalet::signature<3, 0> > em;
typedef gaalet::algebra<gaalet::signature<4, 1> > cm;

int main()
{
    typedef gaalet::metric_combination_traits<em::metric, cm::metric>::metric emcm;

    std::cout << "Combined em, cm: p: " << emcm::p << ", q: " << emcm::q << ", emcm.r: " << emcm::r << std::endl;
}
