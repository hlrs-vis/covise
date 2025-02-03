/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#if !defined(__BIRDRECEIVER_H)
#define __BIRDRECEIVER_H
#include <util/coExport.h>
class INPUT_LEGACY_EXPORT birdReceiver
{
protected:
public:
    birdReceiver(){};
    ~birdReceiver(){};

    unsigned int address;
    unsigned short int buttons;
    int buttons_flag;

    int add_button_data;
    signed short int range;

    float x, y, z;
    float u, v, w, a;

    float h, p, r;
    float m[3][3];
};
#endif
