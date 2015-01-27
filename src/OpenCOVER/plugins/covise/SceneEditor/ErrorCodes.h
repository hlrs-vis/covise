/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef ERROR_CODES_H
#define ERROR_CODES_H

class EventErrors
{
public:
    enum Type
    {
        SUCCESS,
        FAIL,
        EPIC_FAIL,
        UNHANDLED
    };
};

#endif
