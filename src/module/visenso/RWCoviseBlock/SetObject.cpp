/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "SetObject.h"
#include "OutputObjectFactory.h"
#include <iostream>

#if defined(WIN32) || defined(WIN64)
#define COVISE_write _write
#else //WIN32 || WIN64
#include <unistd.h>
#define COVISE_write write
#endif //WIN32 || WIN64

#include <do/coDoSet.h>
#include <do/coDistributedObject.h>

SetObject::SetObject()
    : OutputObject("SETELE")
{
}

SetObject::SetObject(const SetObject &o)
    : OutputObject(o)
{
}

SetObject::SetObject(const OutputObject &o)
    : OutputObject("SETELE")
{
    std::string x = o.type();
}

bool SetObject::process(const int &fd)
{
    if (!distrObj_)
        return false;

    if (!distrObj_->isType("SETELE"))
    {
        std::cerr << "SetObject::process() object mismatch SETELE expecxted" << std::endl;
        std::cerr << "SetObject::process() got " << distrObj_->getType() << std::endl;

        return false;
    }

    if (!distrObj_->objectOk())
    {
        std::cerr << "SetObject::process() object has a shm problem" << std::endl;

        return false;
    }

    coDoSet *set = (coDoSet *)distrObj_;
    int no_elems;
    const coDistributedObject *const *setList = set->getAllElements(&no_elems);

    COVISE_write(fd, "SET", 3 * sizeof(char));
    COVISE_write(fd, &no_elems, sizeof(int));

    // add varname for data values
    const char *varName = set->getAttribute("SPECIES");
    if (varName)
    {
        COVISE_write(fd, varName, (strlen(varName) + 1) * sizeof(char));
    }

    // add part name if some exist
    const char *partName;
    for (int i = 0; i < no_elems; i++)
    {
        partName = setList[i]->getAttribute("PART");
        if (partName)
        {
            COVISE_write(fd, partName, (strlen(partName) + 1) * sizeof(char));
        }
    }

    for (int i = 0; i < no_elems; i++)
    {
        OutputObject *out = OutputObjectFactory::create(setList[i]);
        out->process(fd);
    }

    return true;
}

SetObject *SetObject::clone() const
{
    std::cerr << "SetObject::clone() called  type: " << type_ << std::endl;

    return new SetObject(*this);
}

SetObject::~SetObject()
{
}
