/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_SYNCACTION_H
#define CO_SYNCACTION_H

#include <util/coExport.h>
namespace opencover
{
/** Allow all clients to performe a synchronized Action
 */
class COVEREXPORT coSynchronizedAction
{
protected:
    static int sequenceNumber;

public:
    int mySequenceNumber;
    int host;
    int type;
    int blocking;
    int numberOfConfirmations;
    coSynchronizedAction();
    coSynchronizedAction(int sequenceNumber, int host, int type, int blocking);
    virtual ~coSynchronizedAction();
    void fireAction(int type);
    //virtual void executeInSync(int initiatingHostID, int type, int blocking);
};
}
#endif
