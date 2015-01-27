/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef REMOTEREBOOTCONSTANTS_H
#define REMOTEREBOOTCONSTANTS_H

enum RemoteBootMethod
{
    RB_AUTO = -1,
    RB_REMOTE_DAEMON = 1,
    RB_SSH,
    RB_WMI
};
#endif
