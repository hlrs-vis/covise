/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_BUTTONDEVICE_H
#define CO_BUTTONDEVICE_H

#include <util/coExport.h>

namespace opencover
{
class INPUT_LEGACY_EXPORT ButtonDevice
{
public:
    virtual ~ButtonDevice()
    {
    }
    virtual void getButtons(int station, unsigned int *status) = 0;
    virtual int getWheel(int station) = 0;
};
}
#endif
