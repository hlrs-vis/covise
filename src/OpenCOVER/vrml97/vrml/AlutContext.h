/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _VRML_ALUT_CONTEXT_H
#define _VRML_ALUT_CONTEXT_H

namespace vrml
{

class AlutContext
{
public:
    AlutContext();
    ~AlutContext();

    bool is_initialized;
    bool has_context;

private:
    static int _refcount;
};

}

#endif
