/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    coRenderer
//
// Description: Base class for renderers build ion the common coModule API
//
// Initial version: 03.09.2002
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef CORENDERER_H
#define CORENDERER_H

#include "coModule.h"

namespace covise
{

class APIEXPORT coRenderer : public coModule
{
public:
    /// default CONSTRUCTOR
    coRenderer(int argc, char *argv[]);

    /// DESTRUCTOR
    virtual ~coRenderer();

    const coDistributedObject *getInObj(const coObjInfo &info);

    virtual int compute(const char *port);

private:
    coInputPort *inPort_;
};
}
#endif
