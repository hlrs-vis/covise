/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "dmgr.h"

using namespace covise;

AddressOrderedTree *coShmAlloc::used_list = 0L;
AddressOrderedTree *coShmAlloc::free_list = 0L;
SizeOrderedTree *coShmAlloc::free_size_list = 0L;
int DataManagerProcess::max_t = 0;
