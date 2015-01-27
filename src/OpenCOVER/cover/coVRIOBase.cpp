/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRIOBase.h"

using namespace opencover;
coVRIOBase::coVRIOBase()
    : progress(-1.0f)
{
}

coVRIOBase::~coVRIOBase()
{
}

/**
 * Gets the current file loading / saving progress.
 * @returns The current file progress between 0 and 100, or -1 if progress cannot be determined.
 */

float coVRIOBase::getIOProgress() const
{
    return this->progress;
}

/**
 * Sets the current file loading / saving progress
 * @param progress The current file progress.
 */

void coVRIOBase::setIOProgress(float progress)
{
    this->progress = progress;
}
