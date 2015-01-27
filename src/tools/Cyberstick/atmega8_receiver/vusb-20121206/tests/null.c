/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* Name: null.c
 * Project: Testing driver features
 * Author: Christian Starkjohann
 * Creation Date: 2008-04-29
 * Tabsize: 4
 * Copyright: (c) 2008 by OBJECTIVE DEVELOPMENT Software GmbH
 * License: GNU GPL v2 (see License.txt), GNU GPL v3 or proprietary (CommercialLicense.txt)
 */

/*
This is a NULL main() function to find out the code size required by libusb's
startup code, interrupt vectors etc.
*/
#include <avr/io.h>

/* ------------------------------------------------------------------------- */

int main(void)
{
    for (;;)
        ;
    return 0;
}

/* ------------------------------------------------------------------------- */
