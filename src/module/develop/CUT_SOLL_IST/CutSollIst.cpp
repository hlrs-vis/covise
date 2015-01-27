/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                    (C) 2002 VirCinity  **
 ** Description: extract a layer out of an object                          **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
\**************************************************************************/

//#include <iostream.h>
#include "CutSollIst.h"

CutSollIst::CutSollIst()
    : coModule("cut something out of an object")
{
    // ports
    p_geo_in = addInputPort("geo_in", "coDoPolygons", "geometry");
    p_geo_out = addOutputPort("geo_out", "coDoPolygons", "geometry");
    p_cut1 = addOutputPort("cut1", "coDoText", "cut 1 params");
    p_cut2 = addOutputPort("cut2", "coDoText", "cut 2 params");

    //params
    distance = addFloatParam("distance", "distance of plane");
    normal = addFloatVectorParam("normal", "normal of plane");
    thick = addFloatParam("thickness", "thickness of layer");

    // parameters are changed by COVER, so must be immediate
    distance->setImmediate(1);
    normal->setImmediate(1);

    // set default values
    distance->setValue(0.0);
    normal->setValue(0.0, 0.0, 1.0);
    thick->setValue(0.1);
}

int main(int argc, char *argv[])
{
    CutSollIst *application = new CutSollIst;
    application->start(argc, argv);
    return 0;
}

////// this is called whenever we have to do something

int CutSollIst::compute()
{
    float pdistance = distance->getValue();
    float pthickness = thick->getValue();
    float pnormal[3];
    normal->getValue(pnormal[0], pnormal[1], pnormal[2]);

    if (p_geo_in->getCurrentObject()->isType("SETELE"))
    {
        coDoSet *s_in = (coDoSet *)p_geo_in->getCurrentObject();

        int i, num = 0;
        coDistributedObject *const *elems = s_in->getAllElements(&num);

        coDistributedObject **new_elems = new coDistributedObject *[num + 1];
        new_elems[num] = 0;

        for (i = 0; i < num; i++)
        {
            elems[i]->incRefCount();
            new_elems[i] = elems[i];
        }
        // add feedback attribute
        char feedback[512];
        sprintf(feedback, "G%s\n%s\n%s\n", Covise::get_module(),
                Covise::get_instance(),
                Covise::get_host());
        new_elems[0]->addAttribute("FEEDBACK", feedback);

        coDoSet *s_out = new coDoSet(p_geo_out->getObjName(), new_elems);
        s_out->copyAllAttributes(s_in);

        p_geo_out->setCurrentObject(s_out);
        delete[] new_elems;

        // create params for CutGeometry
        char param[512];
        char *text;

        sprintf(param, "distance %f\nnormal %f %f %f", pdistance - pthickness,
                pnormal[0], pnormal[1], pnormal[2]);
        coDoText *cut1 = new coDoText(p_cut1->getObjName(), strlen(param) + 1, param);
        p_cut1->setCurrentObject(cut1);

        sprintf(param, "distance %f\nnormal %f %f %f", (-1.) * (pdistance + pthickness),
                (-1.) * pnormal[0], (-1.) * pnormal[1], (-1.) * pnormal[2]);
        coDoText *cut2 = new coDoText(p_cut2->getObjName(), strlen(param) + 1, param);
        p_cut2->setCurrentObject(cut2);
    }
    else
    {
        Covise::sendError("Only set elements as input implemented right now.");
        return STOP_PIPELINE;
    }

    return CONTINUE_PIPELINE;
}
