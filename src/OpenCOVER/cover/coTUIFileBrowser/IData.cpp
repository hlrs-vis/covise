/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "IData.h"
using namespace opencover;

IData::IData(void)
{
}

IData::~IData(void)
{
}

void IData::setLocation(std::string ip)
{
    this->mIP = ip;
}

std::string IData::getCurrentPath()
{
    return this->mCurrentPath;
}

void IData::setCurrentPath(std::string path)
{
    this->mCurrentPath = path;
}

void IData::setFilter(std::string filter)
{
    this->mFilter = filter;
}
