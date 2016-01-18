/* This file is part of COVISE.

You can use it under the terms of the GNU Lesser General Public License
version 2.1 or later, see lgpl-2.1.txt.

* License: LGPL 2+ */

#include <oscBody.h>


using namespace OpenScenario;


sexType::sexType()
{
    addEnum("male", oscBody::male);
    addEnum("female", oscBody::female);
}

sexType *sexType::instance()
{
    if(inst == NULL)
    {
        inst = new sexType();
    }
    return inst;
}

sexType *sexType::inst = NULL;
