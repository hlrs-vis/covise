/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CuttingSurface3DTex.h"
#include <api/coFeedback.h>

#undef VERBOSE

#ifdef VERBOSE
inline const char *name(coDistributedObject *obj)
{
    return obj ? obj->getName() : "NULL";
}
#endif

CuttingSurface3DTex::CuttingSurface3DTex(int argc, char *argv[])
    : coModule(argc, argv, "Combine sampled data and cuttingsurface")
{

    // Ports
    port_cuttingsurface = addInputPort("cuttingsurface", "Polygons|TriangleStrips", "Cuttingsurface geometry");
    port_cuttingsurface->setRequired(1);

    port_cuttingsurface_colors = addInputPort("cuttingsurface_colors", "RGBA", "Colors on the Cuttingsurface");
    port_cuttingsurface_colors->setRequired(1);

    port_samplegrid = addInputPort("unigrid", "UniformGrid", "The sampled grid");
    port_samplegrid->setRequired(0);

    port_samplegrid_colors = addInputPort("unigrid_colors", "RGBA", "Colors on the sampled grid");
    port_samplegrid_colors->setRequired(0);

    port_cut3DTex = addOutputPort("cuttingsurface3DTex", "Geometry", "set consisting of 2 geometry objects");
}

int CuttingSurface3DTex::compute(const char *)
{

    // get input objects
    coDistributedObject *cuttingsurface = port_cuttingsurface->getCurrentObject();
    coDistributedObject *cuttingsurface_colors = port_cuttingsurface_colors->getCurrentObject();
    coDistributedObject *samplegrid = port_samplegrid->getCurrentObject();
    coDistributedObject *samplegrid_colors = port_samplegrid_colors->getCurrentObject();

    // output objects
    coDoGeometry *samplegridGeom = NULL;
    coDoGeometry *cuttingsurfaceGeom = NULL;

    if (!samplegrid || !samplegrid->isType("UNIGRD"))
    {
        sendError("Input object at port '%s' is not a uniform grid",
                  port_samplegrid->getName());
        return STOP_PIPELINE;
    }

    if (!samplegrid_colors || !samplegrid_colors->isType("RGBADT"))
    {
        sendError("Input object at port '%s' is not a RGBA data",
                  samplegrid_colors->getName());
        return STOP_PIPELINE;
    }

    //
    char *cuttingsurfaceGeomName;
    char *samplegridGeomName;
    const char *set_name;

    char buf[1000], modulename[1000], moduleinstance[1000], hostname[1000];

    if (!cuttingsurface || !cuttingsurface_colors)
        return STOP_PIPELINE; // should never get here...

    if (!cuttingsurface->isType("POLYGN") && !cuttingsurface->isType("TRIANG"))
    {
        sendError("Wrong object type on port '%s'", port_cuttingsurface->getName());
        return STOP_PIPELINE;
    }

    // check species attribute
    // ...

    // get the name of the output object
    set_name = port_cut3DTex->getObjName();

    // we re-use the objects
    if (samplegrid && samplegrid_colors)
    {
        samplegrid->incRefCount();
        samplegrid_colors->incRefCount();
    }
    cuttingsurface->incRefCount();
    cuttingsurface_colors->incRefCount();

    // we want to start a plugin for feedback
    // and at the
    //const char *str = cuttingsurface->getAttribute("IGNORE");
    //coFeedback feedback("CuttingSurface3DTexPlugin");
    //feedback.addString(str);
    //feedback.apply(cuttingsurface);

    // New-Style interaction attribute available?
    const char *interactorAttr = cuttingsurface->getAttribute("INTERACTOR");
    if (interactorAttr)
    {
    }
    // Old-Style interaction attributes available?
    else
    {
        const char *value = cuttingsurface->getAttribute("IGNORE");
        const char *str = cuttingsurface->getAttribute("FEEDBACK");

        if (!value || !str)
        {
            sendError("Polygon object lacks required attributes IGNORE and FEEDBACK");
            return STOP_PIPELINE;
        }

        str++;
        sscanf(str, "%s %s %s", modulename, moduleinstance, hostname);
        cuttingsurface->addAttribute("MODULE", "CuttingSurface3DTexPlugin");
        sprintf(buf, "X%s\n%s\n%s\n%s\n%s", modulename, moduleinstance, hostname, "CuttingSurface", value);
        cuttingsurface->addAttribute("INTERACTOR", buf);
    }

    // create the set elements, each set element is a coDoGeometry
    if (samplegrid && samplegrid_colors)
    {
        cuttingsurfaceGeomName = new char[strlen(set_name) + 3];
        strcpy(cuttingsurfaceGeomName, set_name);
        strcat(cuttingsurfaceGeomName, "_0");
        cuttingsurfaceGeom = new coDoGeometry(cuttingsurfaceGeomName, cuttingsurface);
    }
    else
    {
        cuttingsurfaceGeom = new coDoGeometry(set_name, cuttingsurface);
    }

    cuttingsurfaceGeom->setColor(NONE, cuttingsurface_colors);

    if (samplegrid && samplegrid_colors)
    {
        samplegridGeomName = new char[strlen(set_name) + 3];
        strcpy(samplegridGeomName, set_name);
        strcat(samplegridGeomName, "_1");

        samplegridGeom = new coDoGeometry(samplegridGeomName, samplegrid);
        samplegridGeom->setColor(NONE, samplegrid_colors);
    }

    // append attribute CREATOR_MODULE_NAME CuttingsSurface3dTex_1
    char *creatorModuleName = new char[strlen(get_module()) + 100];
    sprintf(creatorModuleName, "%s_%s", get_module(), get_instance());

    if (samplegridGeom)
        samplegridGeom->addAttribute("CREATOR_MODULE_NAME", creatorModuleName);

    cuttingsurfaceGeom->addAttribute("CREATOR_MODULE_NAME", creatorModuleName);

    // create the output object

    if (samplegridGeom)
    {
        coDistributedObject *elements[2];
        elements[0] = cuttingsurfaceGeom;
        elements[1] = samplegridGeom;

        coDoSet *cut3DTex = new coDoSet(set_name, 2, elements);
        if (!cut3DTex || !cut3DTex->objectOk())
        {
            sendError("Could not create output object");
            return FAIL;
        }
#ifdef VERBOSE
        cerr << "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" << endl;
        cerr << "CuttingSurface3DTex objects:" << endl;
        cerr << "----------------------------" << endl;
        cerr << "Set: " << cut3DTex->getName() << endl;
        cerr << "|" << endl;
        if (cuttingsurfaceGeom)
        {
            cerr << "+--Geometry Container: " << name(cuttingsurfaceGeom) << endl;
            cerr << "|  |" << endl;
            cerr << "|  +--Geomet: " << name(cuttingsurface) << endl;
            cerr << "|  +--Colors: " << name(cuttingsurface_colors) << endl;
            cerr << "|" << endl;
        }
        else
        {
            cerr << "+--Geometry Container: NULL" << endl;
            cerr << "|" << endl;
        }

        if (samplegridGeom)
        {
            cerr << "+--Sample Grid Container: " << name(samplegridGeom) << endl;
            cerr << "|  |" << endl;
            cerr << "|  +--Geomet: " << name(samplegrid) << endl;
            cerr << "|  +--Colors: " << name(samplegrid_colors) << endl;
            cerr << "|" << endl;
        }
        else
        {
            cerr << "+--Geometry Container: NULL" << endl;
            cerr << "|" << endl;
        }
#endif
        // finally, assign object to port
        port_cut3DTex->setCurrentObject(cut3DTex);

        return SUCCESS;
    }
    else
    {
        port_cut3DTex->setCurrentObject(cuttingsurfaceGeom);
        return SUCCESS;
    }
}

MODULE_MAIN(Obsolete, CuttingSurface3DTex)
