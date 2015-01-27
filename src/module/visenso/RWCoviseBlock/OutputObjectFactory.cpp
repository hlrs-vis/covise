/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "OutputObjectFactory.h"
#include "LinesObject.h"
#include "PolygonObject.h"
#include "ScalDataObject.h"
#include "SetObject.h"
#include <iostream>
#include "covise/covise.h"

OutputObjectFactory *OutputObjectFactory::me_ = NULL;

OutputObjectFactory::OutputObjectFactory(const OutputObjectFactory &o)
    : OutputObject(o)
{
}

OutputObjectFactory::OutputObjectFactory()
{
    initialize();
}

void OutputObjectFactory::initialize()
{
    prototypes_["LINES"] = new LinesObject();
    prototypes_["POLYGN"] = new PolygonObject();
    prototypes_["USTSDT"] = new ScalDataObject();
    prototypes_["SETELE"] = new SetObject();
}

OutputObject *OutputObjectFactory::createObj(const coDistributedObject *d) const
{
    if (!d)
    {
        return OutputObject().clone();
    }

    std::string type = d->getType();

    std::cerr << "OutputObjectFactory::createObj(..) should create obj for type " << type << std::endl;

    std::map<std::string, OutputObject *>::const_iterator ret = prototypes_.find(type);

    if (ret != prototypes_.end())
    {
        std::cerr << "OutputObjectFactory::createObj(..) *found* " << ret->second->type() << std::endl;

        OutputObject *oOut = ret->second->clone();
        oOut->setDO(d);

        return oOut;
    }
    else
    {
        std::cout << "OutputObjectFactory::createObj(..) *NOT found* NONE " << std::endl;

        return OutputObject().clone();
    }
}

OutputObject *OutputObjectFactory::create(const coDistributedObject *d)
{
    if (me_ == NULL)
    {
        std::cerr << "OutputObjectFactory::create() Factory created" << std::endl;

        me_ = new OutputObjectFactory();
    }

    OutputObject *oo = me_->createObj(d);

    std::cerr << "OutputObjectFactory::create() TYPE " << oo->type() << std::endl;

    return oo;
}

OutputObjectFactory::~OutputObjectFactory()
{
}
