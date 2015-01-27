/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef IREMOTEDATA_H_
#define IREMOTEDATA_H_

#include "IData.h"
#include <net/message.h>

namespace opencover
{
/**
 * The class provides an abstract interface for the FileBrowser
 * This way access to different data objects from the FileBrowser
 * itself can be handled. Some methods which are common for different
 * data access methods are implemented directly in this abstract class
 * thereby the implementation does not completely stay to the pattern
 * of interfaces.
 * @author Michael Braitmaier
 * @date 2007-01-12
 */
class IRemoteData : public IData
{
public:
    virtual void reqClientList(int pId) = 0;
    virtual void setClientList(covise::Message &msg) = 0;
};
}
#endif
