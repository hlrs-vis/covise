/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                        (C)2005 ZAIK ++
// ++ Description: Add attribute                                          ++
// ++                                                                     ++
// ++ Author: Martin Aumueller (aumueller@uni-koeln.de)                   ++
// ++                                                                     ++
// ++**********************************************************************/

#include <do/coDoSet.h>
// this includes our own class's headers
#include "AddAttribute.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
AddAttribute::AddAttribute(int argc, char *argv[])
    : coModule(argc, argv, "Attach an attribute to an object")
{
    // Parameters

    // Ports
    p_inPort = addInputPort("inObject", "coDistributedObject", "input object");
    p_outPort = addOutputPort("outObject", "coDistributedObject", "output object");

    p_attrName = addStringParam("attrName", "name of attribute");
    p_attrName->setValue("LABEL");

    p_attrVal = addStringParam("attrVal", "value of attribute");
    p_attrVal->setValue("This is a label");
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int AddAttribute::compute(const char *)
{
    const coDistributedObject *object = p_inPort->getCurrentObject();

    int ret = recurse(object);
    if (ret == CONTINUE_PIPELINE)
    {
        object->incRefCount(); // object is reused
        coDoSet *set = new coDoSet(p_outPort->getObjName(), 1, &object);
        p_outPort->setCurrentObject(set);
    }

    return ret;
}

int AddAttribute::recurse(const coDistributedObject *obj)
{
    if (!obj)
        return FAIL;

    if (const coDoSet *set = dynamic_cast<const coDoSet *>(obj))
    {
        int no;
        const coDistributedObject *const *elems = set->getAllElements(&no);

        for (int i = 0; i < no; i++)
        {
            int ret = recurse(elems[i]);
            if (ret != CONTINUE_PIPELINE)
                return ret;
        }
    }
    ((covise::coDistributedObject *)obj)->addAttribute(p_attrName->getValue(), p_attrVal->getValue());

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(Tools, AddAttribute)
