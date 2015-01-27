/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                           (C)2008 HLRS **
 **                                                                        **
 ** Description: 					                                       **
 **  this module allows to group other modules into so called variants.    **
 **                                                                        **
 **                                                                        **
 ** Name:        VariantMarker                                             **
 ** Category:    Filter                                                    **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "VariantMarker.h"
#include <util/coVector.h>
#include <do/coDoSet.h>

/*! \brief constructor
 *
 * create In/Output Ports and module parameters here
 */
VariantMarker::VariantMarker(int argc, char **argv)
    : coModule(argc, argv, "Set as variant")
{
    p_Inport = addInputPort("inObject", "coDistributedObject", "input object");
    p_Outport = addOutputPort("outObject", "coDistributedObject", "output object");
    p_varName = addStringParam("varName", "name of variant");
    p_varName->setValue("name of variant");
}

VariantMarker::~VariantMarker()
{
}

void VariantMarker::param(const char * /* name */, bool /* inMapLoading */)
{
}

int VariantMarker::compute(const char * /* port */)
{
    const coDistributedObject *object = p_Inport->getCurrentObject();

    //int ret = recurse(object);
    //if(ret == CONTINUE_PIPELINE)
    // {
    object->incRefCount(); // object is reused
    coDoSet *set = new coDoSet(p_Outport->getObjName(), 1, &object);
    set->addAttribute("VARIANT", p_varName->getValue());
    p_Outport->setCurrentObject(set);

    //}

    return 1;
}

MODULE_MAIN(Filter, VariantMarker)
