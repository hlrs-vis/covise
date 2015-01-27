/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <iostream.h>
#include "ChoiceList.h"

ostream &operator<<(ostream &str, const ChoiceList &cl)
{
    const char *const *list = cl.get_strings();
    int num = cl.get_num();
    int i;
    str << "ChoiceList with " << num << " choices:" << endl;
    for (i = 0; i < num; i++)
    {
        str << '(' << cl.get_orig_num(i + 1) << ")  " << list[i];
    }
    return str;
}
