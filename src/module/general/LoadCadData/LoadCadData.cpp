/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                           (C)2008 HLRS **
 **                                                                        **
 ** Description: 					                   **
 **                                                                        **
 ** Name:        LoadCadData                                               **
 ** Category:    I/O Module                                                **
 **                                                                        **
 **                                                                        **
\****************************************************************************/

#include "LoadCadData.h"
#include <util/coVector.h>

/*! \brief constructor
 *
 * create In/Output Ports and module parameters here
 */
LoadCadData::LoadCadData(int argc, char **argv)
    : coModule(argc, argv, "Read CAD data")
{

    p_pointName = addOutputPort("model", "Points", "Model");
    p_modelPath = addFileBrowserParam("modelPath", "modelPath");
    p_scale = addFloatParam("scale", "global Scale factor used for OpenCover session");
    p_resize = addFloatVectorParam("resize", "Resize factor");
    p_rotangle = addFloatParam("rotangle", "angle for rotation");
    p_tansvec = addFloatVectorParam("transvec", "Vector for translation");
    p_rotvec = addFloatVectorParam("rotsvec", "Vector for rotation");
    p_backface = addBooleanParam("backface", "Backface Culling");
    p_orientation_iv = addBooleanParam("orientation_iv", "Orientation of iv models like in Inventor Renderer");
    p_convert_xforms_iv = addBooleanParam("convert_xforms_iv", "create LoadCadData DCS nodes");

    p_scale->setValue(-1.0);
    p_rotangle->setValue(0);
    p_resize->setValue(1, 1, 1);
}

LoadCadData::~LoadCadData()
{
}

void LoadCadData::param(const char * /* name */, bool /* inMapLoading */)
{
}

int LoadCadData::compute(const char * /* port */)
{
    //
    // ...... do work here ........
    //

    char buf[2000]; //,*b;

    const char *fname = p_modelPath->getValue();

    float scale = p_scale->getValue();
    bool backface = p_backface->getValue();
    bool orientation_iv = p_orientation_iv->getValue();
    bool convertXforms_iv = p_convert_xforms_iv->getValue();

    float c = 0.f;
    point = new coDoPoints(p_pointName->getObjName(), 1, &c, &c, &c);
    if (point->objectOk())
    {
        point->addAttribute("CAD_FILE", fname);
        sendInfo("Setting CADFile to [%s]\n", fname);

        point->addAttribute("COLOR", "black");

        if (backface)
            point->addAttribute("BACKFACE", "ON");
        else
            point->addAttribute("BACKFACE", "OFF");

        if (orientation_iv)
            point->addAttribute("PFIV_CONVERT_ORIENTATION", "ON");
        else
            point->addAttribute("PFIV_CONVERT_ORIENTATION", "OFF");

        if (convertXforms_iv)
            point->addAttribute("PFIV_CONVERT_XFORMS", "ON");
        else
            point->addAttribute("PFIV_CONVERT_XFORMS", "OFF");

        if (scale < 0.0)
        {
            point->addAttribute("SCALE", "viewAll");
        }
        else if (scale == 0.0)
        {
            point->addAttribute("SCALE", "keep");
        }
        else
        {
            sprintf(buf, "%f", scale);
            point->addAttribute("SCALE", buf);
        }

        point->addAttribute("OBJECTNAME", this->getTitle());

        sprintf(buf, "%f %f %f", p_tansvec->getValue(0), p_tansvec->getValue(1), p_tansvec->getValue(2));
        point->addAttribute("TRANSLATE_OBJECT", buf);

        sprintf(buf, "%f", p_rotangle->getValue() * M_PI / 180.);
        point->addAttribute("ROTANGLE_OBJECT", buf);

        sprintf(buf, "%f %f %f", p_rotvec->getValue(0), p_rotvec->getValue(1), p_rotvec->getValue(2));
        point->addAttribute("ROTATE_OBJECT", buf);

        sprintf(buf, "%f %f %f", p_resize->getValue(0), p_resize->getValue(1), p_resize->getValue(2));
        point->addAttribute("RESIZE_OBJECT", buf);

        p_pointName->setCurrentObject(point);
    }
    else
    {
        sendError("ERROR: Could not create Point Object");
        return 0;
    }
    return 0;
}

MODULE_MAIN(IO, LoadCadData)
