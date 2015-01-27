/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <covise/covise.h>
#include <invent.h>

void main()
{
    inventory_t *inv;
    while ((inv = getinvent()) != NULL)
    {
        if (inv->inv_class == INV_SERIAL)
        {
            cout << "Type: " << inv->inv_type << "\n controller: " << inv->inv_controller << "\n unit: " << inv->inv_unit << "\n state: " << inv->inv_state << "\n";
        }
    }
}
