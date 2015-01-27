/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "coSynchronizedAction.h"
using namespace covise;
using namespace opencover;
int coSynchronizedAction::sequenceNumber = 0;
coSynchronizedAction::coSynchronizedAction()
{
    sequenceNumber++;
    mySequenceNumber = sequenceNumber;
    host = 0;
    type = 0;
    blocking = 0;
}

coSynchronizedAction::~coSynchronizedAction()
{
}

coSynchronizedAction::coSynchronizedAction(int s, int h, int t, int b)
{
    mySequenceNumber = s;
    host = h;
    type = t;
    blocking = b;
}

void coSynchronizedAction::fireAction(int type)
{
    (void)type;
}

/*virtual void coSynchronizedAction::executeInSync(int initiatingHostID, int t, int b)
{
    host = initiatingHostID;
    type = ttype;
    blocking = b;
}*/
