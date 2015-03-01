/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "ScalDataObject.h"
#include <iostream>
#include "covise/covise.h"
#include <do/coDoData.h>

static ssize_t COVISE_write(int fd, const void *buf, size_t count)
{
#if defined(WIN32) || defined(WIN64)
    ssize_t ret = _write(fd, buf, count);
#else //WIN32 || WIN64
    ssize_t ret = write(fd, buf, count);
#endif //WIN32 || WIN64
    if (ret == -1)
    {
        std::cerr << "RWCoviseBlock: write error: " << strerror(errno) << std::endl;
    }
    else if (ret != count)
    {
        std::cerr << "RWCoviseBlock: short write, " << ret << " instead of " << count << std::endl;
    }
    return ret;
}

ScalDataObject::ScalDataObject()
    : OutputObject("USTSDT")
{
}

ScalDataObject::ScalDataObject(const ScalDataObject &o)
    : OutputObject(o)
{
}

bool ScalDataObject::process(const int &fd)
{
    if (!distrObj_)
        return false;

    if (!distrObj_->isType("USTSDT"))
    {
        std::cerr << "ScalDataObject::process() object mismatch " << type_ << " expecxted" << std::endl;
        std::cerr << "ScalDataObject::process() got " << distrObj_->getType() << std::endl;

        return false;
    }

    if (!distrObj_->objectOk())
    {
        std::cerr << "ScalDataObject::process() object has a shm problem" << std::endl;

        return false;
    }

    coDoFloat *ustData = (coDoFloat *)distrObj_;
    int numData = ustData->getNumPoints();
    float *data(NULL);
    ustData->getAddress(&data);

    COVISE_write(fd, "DATA-1", 6 * sizeof(char));
    COVISE_write(fd, &numData, sizeof(int));
    COVISE_write(fd, data, numData * sizeof(float));
    COVISE_write(fd, "FI", 2 * sizeof(char));

    return true;
}

ScalDataObject *ScalDataObject::clone() const
{
    std::cerr << "ScalDataObject::clone() called  type: " << type_ << std::endl;

    return new ScalDataObject(*this);
}

ScalDataObject::~ScalDataObject()
{
}
