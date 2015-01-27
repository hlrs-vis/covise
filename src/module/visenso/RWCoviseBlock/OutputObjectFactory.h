/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    OutputObjectFactory
//
// Description:
//
// Initial version: 04.2007
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2009 by VISENSO GmbH
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  Changes:
//     Ported: 05.2009
//
#ifndef OUTPUTOBJECT_FACTORY_H
#define OUTPUTOBJECT_FACTORY_H

#include <string>
#include <map>

#include "OutputObject.h"
#include "covise/covise.h"

class OutputObjectFactory : OutputObject
{
public:
    static OutputObject *create(const coDistributedObject *d);

private:
    // *this should not be created
    OutputObjectFactory();
    OutputObjectFactory(const OutputObjectFactory &o);
    virtual ~OutputObjectFactory();

    void initialize();
    OutputObject *createObj(const coDistributedObject *d) const;

    static OutputObjectFactory *me_;
    std::map<std::string, OutputObject *> prototypes_;
};

#endif
