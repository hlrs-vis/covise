/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description:  COVISE CuttingSurface  CuttingSurface module                **
 **                                                                        **
 **                                                                        **
 **                             (C) 1995                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Uwe Woessner                                                  **
 **                                                                        **
 **                                                                        **
 ** Date:  23.02.95  V1.0                                                  **
\**************************************************************************/

#include "CuttingSurface.h"
#include <alg/coCuttingSurface.h>
#include <util/covise_version.h>
#include <config/CoviseConfig.h>
#include <do/coDoSet.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUnstructuredGrid.h>
#include <do/coDoText.h>
#include <util/coWristWatch.h>

#ifdef _COMPLEX_MODULE_
#include <covise/covise_objalg.h>
#endif

coWristWatch ww_;

void
CuttingSurfaceModule::copyAttributesToOutObj(coInputPort **input_ports,
                                             coOutputPort **output_ports, int port)
{
#ifdef _COMPLEX_MODULE_
    if (port == 0) // do not treat here the coDoGeometry port
    {
        return;
    }
#endif
    if (port > 1 + shiftOut)
    {
        // isolines get the same attributes as the pertinent input scalar
        if (port == shiftOut + 3 && output_ports[port]->getCurrentObject())
        {
            if (input_ports[2]->getCurrentObject())
            {
                copyAttributes(output_ports[port]->getCurrentObject(),
                               input_ports[2]->getCurrentObject());
            }
            else if (input_ports[1]->getCurrentObject())
            {
                copyAttributes(output_ports[port]->getCurrentObject(),
                               input_ports[1]->getCurrentObject());
            }
        }
        return;
    }
    else if (input_ports[port - shiftOut] && output_ports[port] && output_ports[port]->getCurrentObject() && input_ports[port - shiftOut]->getCurrentObject())
    {

        if (output_ports[port])
        {
            copyAttributes(output_ports[port]->getCurrentObject(),
                           input_ports[port - shiftOut]->getCurrentObject());
        }
    }
}

CuttingSurfaceModule::CuttingSurfaceModule(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Extract a plane from a data set", true)
    ,
#ifdef _COMPLEX_MODULE_
    shiftOut(1)
#else
    shiftOut(0)
#endif
{
#ifdef _COMPLEX_MODULE_
    autoTitleConfigured = coCoviseConfig::isOn("System.AutoName.CuttingSurfaceModuleComp", false);
#else
    autoTitleConfigured = coCoviseConfig::isOn("System.AutoName.CuttingSurfaceModule", false);
#endif

    // initially we do what is configured in covise.config - but User may override
    // by setting his own title: done in param()
    autoTitle = autoTitleConfigured;

    p_MeshIn = addInputPort("GridIn0", "UnstructuredGrid|UniformGrid|StructuredGrid|RectilinearGrid", "input mesh");
    p_DataIn = addInputPort("DataIn0", "Byte|Float|Vec3", "input data");
    p_DataIn->setRequired(1);
    p_IBlankIn = addInputPort("DataIn3", "Text", "this char Array marks cells to be processed or not");
    p_IBlankIn->setRequired(0);
#ifdef _COMPLEX_MODULE_
    p_ColorMapIn = addInputPort("ColormapIn0", "ColorMap", "color map to create geometry");
    p_ColorMapIn->setRequired(0);
    p_SampleGeom_ = addInputPort("GridIn1", "UniformGrid", "Sample grid");
    p_SampleGeom_->setRequired(0);
    p_SampleData_ = addInputPort("DataIn4", "Float|Vec3", "Sample data");
    p_SampleData_->setRequired(0);
    p_GeometryOut = addOutputPort("GeometryOut0", "Geometry", "Cutting plane");
#endif

    p_MeshOut = addOutputPort("GridOut0", "Polygons|TriangleStrips", "Cuttingplane");
    p_DataOut = addOutputPort("DataOut0", "Float|Vec3", "interpolated data");
    p_NormalsOut = addOutputPort("DataOut1", "Vec3", "Surface normals");

    p_vertex = addFloatVectorParam("vertex", "Normal of cuttingplane, center of sphere or point on cylinder axis");
    p_vertex->setValue(1., 0., 0.);

    p_point = addFloatVectorParam("point", "Point on cuttingplane, or on sphere or Point on a cylinder");
    p_point->setValue(.5, 0., 0.);

    p_scalar = addFloatParam("scalar", "Distance from the origin to the cuttingplane or cylinder radius or radius of the sphere");
    p_scalar->setValue(.5);

    p_skew = addBooleanParam("skew", "Modify vertex and point slightly in order to avoid cutting exactly on cell boundaries");
    p_skew->setValue(false);

    const char *option_labels[] = { "Plane", "Sphere", "Cylinder-X", "Cylinder-Y", "Cylinder-Z" };
    p_option = addChoiceParam("option", "Plane or sphere");
    p_option->setValue(5, option_labels, 0);

    p_gennormals = addBooleanParam("gennormals", "Supply normals");
    p_gennormals->setValue(0);

    p_genstrips = addBooleanParam("genstrips", "convert triangles to strips");
    p_genstrips->setValue(1);

    p_genDummyS = addBooleanParam("genDummyS", "generate a dummy surface if the object hasn't been cut");
    p_genDummyS->setValue(1);

#ifdef _COMPLEX_MODULE_
    p_color_or_texture = addBooleanParam("color_or_texture", "colors or texture");
    p_color_or_texture->setValue(1);

    const char *ChoiseVal1[] = { "1*scale", "length*scale" /*,"according_to_data"*/ };
    p_scale = addFloatSliderParam("scale", "Scale factor");
    p_scale->setValue(0.0, 1.0, 1.0);
    p_length = addChoiceParam("length", "Length of vectors");
    p_length->setValue(2, ChoiseVal1, 0);
    p_num_sectors = addInt32Param("num_sectors", "number of lines for line tip");
    p_num_sectors->setValue(0);
    p_arrow_head_factor = addFloatParam("arrow_head_factor", "Relative length of arrow head");
    p_arrow_head_factor->setValue(0.2f);
    p_arrow_head_angle = addFloatParam("arrow_head_angle", "Opening angle of arrow head");
    p_arrow_head_angle->setValue(9.5f);
    p_project_lines = addBooleanParam("project_lines", "project lines onto the surface");
    p_project_lines->setValue(0);

    const char *vector_labels[] = { "SurfaceAndLines", "OnlySurface", "OnlyLines" };
    p_vector = addChoiceParam("vector", "SurfaceOrLines");
    p_vector->setValue(3, vector_labels, 0);
#endif

    p_vertex->show();
    p_point->show();
    p_scalar->show();
    p_option->show();

    p_vertexratio = addFloatParam("vertex_ratio", "Vertex Alloc Ratio");
    p_vertexratio->setValue(4.);

    //vertexAllocRatio=4;
    Polyhedra = coCoviseConfig::isOn("Module.CuttingSurfaceModule.SupportPolyhedra", true);
    maxPolyPerVertex = coCoviseConfig::getInt("Module.CuttingSurfaceModule.PolyPerVertex", 17);
    pointMode = true;

    /// Send old-style or new-style feedback: Default values different HLRS/Vrc
    fbStyle_ = FEED_NEW;
    std::string tmpStr = coCoviseConfig::getEntry("System.FeedbackStyle.CuttingSurfaceModule");
    const char *fbStyleStr = tmpStr.c_str();
    if (fbStyleStr)
    {
        if (0 == strncasecmp("NONE", fbStyleStr, 4))
            fbStyle_ = FEED_NONE;
        if (0 == strncasecmp("OLD", fbStyleStr, 3))
            fbStyle_ = FEED_OLD;
        if (0 == strncasecmp("NEW", fbStyleStr, 3))
            fbStyle_ = FEED_NEW;
        if (0 == strncasecmp("BOTH", fbStyleStr, 4))
            fbStyle_ = FEED_BOTH;
    }

#ifdef _COMPLEX_MODULE_
    static float defaultMinmaxValues[2] = { 0.0, 0.0 };
    p_minmax = addFloatVectorParam("MinMax", "Minimum and Maximum value");
    p_minmax->setValue(2, defaultMinmaxValues);
    p_autoScale = addBooleanParam("autoScales", "Automatically adjust Min and Max");
    p_autoScale->setValue(0);
#endif
}

//================called by compute
void CuttingSurfaceModule::ini_borders()
{
    x_minb = y_minb = z_minb = FLT_MAX; // use float.h instead of values.h
    x_maxb = y_maxb = z_maxb = -FLT_MAX;
}

void CuttingSurfaceModule::comp_borders(int nb_elem, float *x, float *y, float *z)
{
    if (!p_genDummyS->getValue())
        return;

    int i;
    if (x)
    {
        for (i = 0; i < nb_elem; i++)
        {
            if (x[i] > x_maxb)
                x_maxb = x[i];
            if (x[i] < x_minb)
                x_minb = x[i];
        }
    }
    if (y)
    {
        for (i = 0; i < nb_elem; i++)
        {
            if (y[i] > y_maxb)
                y_maxb = y[i];
            if (y[i] < y_minb)
                y_minb = y[i];
        }
    }
    if (z)
    {
        for (i = 0; i < nb_elem; i++)
        {
            if (z[i] > z_maxb)
                z_maxb = z[i];
            if (z[i] < z_minb)
                z_minb = z[i];
        }
    }
}

//======================================================================
// Called when immediate parameters change
//======================================================================

inline float norm(float val[3])
{
    return sqrt(val[0] * val[0]
                + val[1] * val[1]
                + val[2] * val[2]);
}

void CuttingSurfaceModule::param(const char *paramName, bool inMapLoading)
{
    // title: If user sets it, we have to de-activate auto-names
    if (strcmp(paramName, "SetModuleTitle") == 0)
    {
        // find out "real" module name
        char realTitle[1024];
        sprintf(realTitle, "%s_%s", get_module(), get_instance());

        // if it differs from the title - disable automatig settings
        if (strcmp(realTitle, getTitle()) != 0)
            autoTitle = false;
        else
            autoTitle = autoTitleConfigured; // otherwise do whatever configured

        return;
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    // +++ Map loading: save values after all parameters are read
    if (inMapLoading)
    {
        if (strcmp(paramName, "option") == 0)
        {
            // last option - save all values into array
            p_vertex->getValue(param_vertex[0], param_vertex[1], param_vertex[2]);
            p_point->getValue(param_point[0], param_point[1], param_point[2]);
            param_scalar = p_scalar->getValue();

            // make sure everything fits together even if saved wrong
            pointMode = false;
            UpdatePoint(p_option->getValue());
        }
        if (strcmp(paramName, "vertex_ratio") == 0)
        {
            vertexAllocRatio = coCoviseConfig::getFloat("Module.CuttingSurfaceModule.VertexRatio", vertexAllocRatio);
        }
        return;
    }
    else
    {
        if (strcmp(paramName, "vertex_ratio") == 0)
        {
            vertexAllocRatio = p_vertexratio->getValue();
        }
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if (strcmp(paramName, "vertex") == 0)
    {
        if (pointMode)
        {
            UpdateScalar(p_option->getValue());
        }
        else
        {
            UpdatePoint(p_option->getValue());
        }
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    else if (strcmp(paramName, "point") == 0)
    {
        UpdateScalar(p_option->getValue());
        pointMode = true;
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    else if (strcmp(paramName, "scalar") == 0)
    {
        UpdatePoint(p_option->getValue());
        pointMode = false;
    }

    // ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    else if (strcmp(paramName, "option") == 0)
    {
        param_option = p_option->getValue();
        if (pointMode)
        {
            UpdateScalar(param_option);
        }
        else
        {
            UpdatePoint(param_option);
        }
    }

#ifdef _COMPLEX_MODULE_
    else if (!inMapLoading && strcmp(paramName, p_minmax->getName()) == 0)
    {
        p_autoScale->setValue(0);
    }
#endif
}

void
CuttingSurfaceModule::UpdatePoint(int option)
{
    p_vertex->getValue(param_vertex[0], param_vertex[1], param_vertex[2]);
    param_scalar = p_scalar->getValue();

    switch (option)
    {
    case 0: // plane
    {
        float normLen = norm(param_vertex);
        if (normLen != 0.0)
            normLen = 1.0f / normLen;

        param_point[0] = param_scalar * param_vertex[0] * normLen;
        param_point[1] = param_scalar * param_vertex[1] * normLen;
        param_point[2] = param_scalar * param_vertex[2] * normLen;
        p_point->setValue(3, param_point);
    }
    break;
    case 1: // sphere
        param_point[0] = param_vertex[0] + param_scalar;
        param_point[1] = param_vertex[1];
        param_point[2] = param_vertex[2];
        p_point->setValue(3, param_point);
        break;
    default: // cylinder
    {
        float direc[3] = { 0.0, 0.0, 0.0 };
        int ind = (option - 3 + 1 + 1) % 3;
        if (ind >= 3)
            ind = 0;
        direc[ind] = 1.0;
        param_point[0] = param_vertex[0] + direc[0] * param_scalar;
        param_point[1] = param_vertex[1] + direc[1] * param_scalar;
        param_point[2] = param_vertex[2] + direc[2] * param_scalar;
        p_point->setValue(3, param_point);
    }
    break;
    }
}

void
CuttingSurfaceModule::UpdateScalar(int option)
{
    p_point->getValue(param_point[0], param_point[1], param_point[2]);
    p_vertex->getValue(param_vertex[0], param_vertex[1], param_vertex[2]);
    switch (option)
    {
    case 0: // plane
    {
        float vertNorm = norm(param_vertex);
        if (vertNorm != 0.0)
            vertNorm = 1.0f / vertNorm;
        param_scalar = (param_vertex[0] * param_point[0]
                        + param_vertex[1] * param_point[1]
                        + param_vertex[2] * param_point[2]) * vertNorm;
        p_scalar->setValue(param_scalar);
    }
    break;
    case 1: // sphere
    {
        float deltax = param_vertex[0] - param_point[0];
        float deltay = param_vertex[1] - param_point[1];
        float deltaz = param_vertex[2] - param_point[2];
        param_scalar = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
        p_scalar->setValue(param_scalar);
    }
    break;
    default: // cylinder
    {
        float direc[3] = { 1.0, 1.0, 1.0 };
        int ind = option - 3 + 1;
        direc[ind] = 0.0;
        // if in scalar mode - set point
        float deltax = direc[0] * (param_vertex[0] - param_point[0]);
        float deltay = direc[1] * (param_vertex[1] - param_point[1]);
        float deltaz = direc[2] * (param_vertex[2] - param_point[2]);
        param_scalar = sqrt(deltax * deltax + deltay * deltay + deltaz * deltaz);
        p_scalar->setValue(param_scalar);
    }
    break;
    }
}

void
CuttingSurfaceModule::preHandleObjects(coInputPort **)
{
    ww_.reset();
    // Automatically adapt our Module's title to the species
    if (autoTitle)
    {
        const coDistributedObject *obj = p_DataIn->getCurrentObject();
        if (obj)
        {
            const char *species = obj->getAttribute("SPECIES");

            size_t len = 0;
            if (species)
                len += strlen(species) + 3;
            char *buf = new char[len + 64];
            if (species)
                sprintf(buf, "Cut-%s:%s", get_instance(), species);
            else
                sprintf(buf, "Cut-%s", get_instance());
            setTitle(buf);
            delete[] buf;
        }
    }

    param_option = p_option->getValue();
    p_vertex->getValue(param_vertex[0], param_vertex[1], param_vertex[2]);
    p_point->getValue(param_point[0], param_point[1], param_point[2]);
    param_scalar = p_scalar->getValue();

    param_vertex[0] += 0.00001f;
    param_vertex[1] += 0.00001f;
    param_vertex[2] += 0.00001f;

    DoPostHandle = true;
    ini_borders();
#ifdef _COMPLEX_MODULE_
    // check for 3dTex usage:
    // p_ColorMapIn, p_SampleGeom_ and p_SampleData_ are all connected
    // p_ColorMapIn has got a COLMAP,
    // p_SampleGeom_ a UNIGRD
    // and p_SampleData_ a USTSDT or USTVDT (in the latter case
    // adjust p_vector to SurfaceOnly)
    if (p_SampleGeom_->isConnected() || p_SampleData_->isConnected())
    {
        if (!p_SampleGeom_->isConnected() || !p_SampleData_->isConnected() || !p_ColorMapIn->isConnected())
        {
            string message("If either ");
            message += p_SampleGeom_->getName();
            message += " or ";
            message += p_SampleData_->getName();
            message += " are connected, then these 2 parameters and  ";
            message += p_ColorMapIn->getName();
            message += " should be connected.";
            sendError("%s", message.c_str());
            DoPostHandle = false;
            return;
        }
        // VECTOR
        if (coObjectAlgorithms::containsType<const coDoVec3 *>(p_SampleData_->getCurrentObject()))
        {
            p_vector->setValue(1); // only surfaces
        }
    }
#endif
}

bool
isVisible(const coDistributedObject *inObj)
{
    if (!inObj)
        return false;
    if (const coDoSet *p_set = dynamic_cast<const coDoSet *>(inObj))
    {
        //coDoSet *p_set = new coDoSet(inObj->getName(),SET_CREATE);
        int how_many_elems;
        const coDistributedObject *const *T_i_sets = p_set->getAllElements(&how_many_elems);
        int i;
        for (i = 0; i < how_many_elems; ++i)
        {
            if (isVisible(T_i_sets[i]))
                return true;
        }
        //delete p_set;
    }
    else if (const coDoTriangleStrips *p_triang = dynamic_cast<const coDoTriangleStrips *>(inObj))
    {
        if (p_triang->getNumStrips() > 0)
            return true;
    }
    else if (const coDoPolygons *p_poly = dynamic_cast<const coDoPolygons *>(inObj))
    {
        // something visible
        if (p_poly->getNumPolygons() > 0)
            return true;
    }
    return false;
}

coDistributedObject *
CuttingSurfaceModule::dummy_polygons(string name, int noSteps,
                                     float **dummyX, float **dummyY, float **dummyZ)
{
    if (name == "")
    {
        *dummyX = NULL;
        *dummyY = NULL;
        *dummyZ = NULL;
        return NULL;
    }
    string polyName = name;
    if (noSteps > 0)
    {
        polyName += "_dummy";
    }
    coDoPolygons *poly = Plane::dummy_polygons(polyName.c_str(), x_minb, x_maxb,
                                               y_minb, y_maxb, z_minb, z_maxb,
                                               param_option, param_vertex[0], param_vertex[1],
                                               param_vertex[2], param_scalar, // FIXME
                                               param_point[0], param_point[1], param_point[2]);
    int *vl, *ll;
    poly->getAddresses(dummyX, dummyY, dummyZ, &vl, &ll);
    if (noSteps > 0)
    {
        coDistributedObject **list = new coDistributedObject *[noSteps + 1];
        list[noSteps] = NULL;
        list[0] = poly;
        int i;
        for (i = 1; i < noSteps; ++i)
        {
            list[i] = poly;
            poly->incRefCount();
        }
        coDoSet *set = new coDoSet(name.c_str(), list);
        delete[] list;
        return set;
    }
    return poly;
}

coDistributedObject *
CuttingSurfaceModule::dummy_normals(string name, int noSteps,
                                    float *dummyX, float *dummyY, float *dummyZ)
{
    if (name == "" || !p_gennormals->getValue())
        return NULL;
    string polyName = name;
    if (noSteps > 0)
    {
        polyName += "_dummy";
    }
    coDoVec3 *poly = Plane::dummy_normals(polyName.c_str(),
                                          dummyX, dummyY, dummyZ, param_option,
                                          param_vertex, param_scalar, param_point); // FIXME
    if (noSteps > 0)
    {
        coDistributedObject **list = new coDistributedObject *[noSteps + 1];
        list[noSteps] = NULL;
        list[0] = poly;
        int i;
        for (i = 1; i < noSteps; ++i)
        {
            list[i] = poly;
            poly->incRefCount();
        }
        coDoSet *set = new coDoSet(name.c_str(), list);
        delete[] list;
        return set;
    }
    return poly;
}

coDistributedObject *
CuttingSurfaceModule::dummy_data(string name, int noSteps)
{
    if (name == "")
        return NULL;
    string polyName = name;
    if (noSteps > 0)
    {
        polyName += "_dummy";
    }
    coDistributedObject *dummy = NULL;
    if (DataType)
    {
        dummy = new coDoFloat(polyName.c_str(), 0);
    }
    else
    {
        dummy = new coDoVec3(polyName.c_str(), 0);
    }
    if (noSteps > 0)
    {
        coDistributedObject **list = new coDistributedObject *[noSteps + 1];
        list[noSteps] = NULL;
        list[0] = dummy;
        int i;
        for (i = 1; i < noSteps; ++i)
        {
            list[i] = dummy;
            dummy->incRefCount();
        }
        coDoSet *set = new coDoSet(name.c_str(), list);
        delete[] list;
        return set;
    }
    return dummy;
}

coDistributedObject *
CuttingSurfaceModule::dummy_tr_strips(string name, int noSteps,
                                      float **dummyX, float **dummyY, float **dummyZ)
{
    isDummy_ = true;
    if (name == "")
    {
        *dummyX = NULL;
        *dummyY = NULL;
        *dummyZ = NULL;
        return NULL;
    }
    string polyName = name;
    if (noSteps > 0)
    {
        polyName += "_dummy";
    }
    coDoTriangleStrips *poly = Plane::dummy_tr_strips(polyName.c_str(), x_minb, x_maxb,
                                                      y_minb, y_maxb, z_minb, z_maxb,
                                                      param_option, param_vertex[0], param_vertex[1],
                                                      param_vertex[2], param_scalar,
                                                      param_point[0], param_point[1], param_point[2]);
    int *vl, *ll;
    poly->getAddresses(dummyX, dummyY, dummyZ, &vl, &ll);
    if (noSteps > 0)
    {
        coDistributedObject **list = new coDistributedObject *[noSteps + 1];
        list[noSteps] = NULL;
        list[0] = poly;
        int i;
        for (i = 1; i < noSteps; ++i)
        {
            list[i] = poly;
            poly->incRefCount();
        }
        coDoSet *set = new coDoSet(name.c_str(), list);
        delete[] list;
        return set;
    }
    return poly;
}

#include "attributeContainer.h"

#include <utility>
#include <vector>

#include <api/coFeedback.h>
void
CuttingSurfaceModule::addFeedbackParams(coDistributedObject *obj)
{
    if (!obj)
    {
        return;
    }
    // add new-style feedback messages
    if (fbStyle_ == FEED_NEW || fbStyle_ == FEED_BOTH)
    {
        coFeedback feedback("CuttingSurface");

        feedback.addPara(p_gennormals);
        feedback.addPara(p_genstrips);
        feedback.addPara(p_genDummyS);
        feedback.addPara(p_scalar);
        feedback.addPara(p_vertex);
        feedback.addPara(p_point);
        feedback.addPara(p_option);
#ifdef _COMPLEX_MODULE_
        feedback.addPara(p_scale);
        feedback.addPara(p_length);
        feedback.addPara(p_num_sectors);
        feedback.addPara(p_color_or_texture);
#endif

        char *t = new char[strlen(getTitle()) + 1];
        strcpy(t, getTitle());

        for (char *c = t + strlen(t); c > t; c--)
        {
            if (*c == '_')
            {
                *c = '\0';
                break;
            }
        }
        char *ud = new char[strlen(t) + 20];
        strcpy(ud, "SYNCGROUP=");
        strcat(ud, t);
        if (strcmp(t, "CuttingSurface") != 0)
        {
            feedback.addString(ud);
        }
        delete[] t;
        delete[] ud;
        feedback.apply(obj);
    }

    // add old-style feedback messages
    if (fbStyle_ == FEED_OLD || fbStyle_ == FEED_BOTH)
    {
        char buf[1024];
        sprintf(buf, "C%s\n%s\n%s\n",
                Covise::get_module(),
                Covise::get_instance(), Covise::get_host());
        if (p_option->getValue() > 1)
            buf[0] = 'Z';
        obj->addAttribute("FEEDBACK", buf);

        char ignore[1024];
        sprintf(ignore, "%f %f %f %f %d", param_vertex[0], param_vertex[1], param_vertex[2], p_scalar->getValue() * (p_skew->getValue() ? 1.00001 : 1.), p_option->getValue());
        obj->addAttribute("IGNORE", ignore);

#ifdef _COMPLEX_MODULE_
        // Complex module: add interactors for 3D-Textures if required input is there
        if (p_SampleGeom_->getCurrentObject() && p_SampleData_->getCurrentObject())
        {
            obj->addAttribute("MODULE", "CuttingSurfaceModule3DTexPlugin");
            sprintf(buf, "X%s\n%s\n%s\n%s\n%s",
                    Covise::get_module(),
                    Covise::get_instance(), Covise::get_host(),
                    "CuttingSurface", ignore);
            obj->addAttribute("INTERACTOR", buf);
        }
#endif
    }
}

#ifdef _COMPLEX_MODULE_
#include <alg/coComplexModules.h>
coDoGeometry *
CuttingSurfaceModule::SampleToGeometry(const coDistributedObject *grid,
                                       const coDistributedObject *data)
{
    string name = p_GeometryOut->getObjName();
    name += "_Sample";
    coDistributedObject *color = ComplexModules::DataTexture(string(name + "_Color"),
                                                             data, p_ColorMapIn->getCurrentObject(), false);
    grid->incRefCount();
    coDoGeometry *do_geo = new coDoGeometry(name.c_str(), grid);

    string creatorModuleName = get_module();
    creatorModuleName += '_';
    creatorModuleName += get_instance();

    do_geo->addAttribute("CREATOR_MODULE_NAME", creatorModuleName.c_str());

    do_geo->setColors(PER_VERTEX, color);
    return do_geo;
}
#endif

class Terminator
{
private:
    bool silent_;

public:
    Terminator()
        : silent_(false)
    {
    }
    ~Terminator()
    {
        if (!silent_)
            Covise::sendInfo("complete run: %6.3f s", ww_.elapsed());
    }
    void silent()
    {
        silent_ = true;
    }
};

void
CuttingSurfaceModule::postHandleObjects(coOutputPort **outPorts)
{
    if (!DoPostHandle)
    {
        return;
    }

    Terminator terminator;

#ifndef _COMPLEX_MODULE_ // in the complex case, only for scalar data
    addFeedbackParams(outPorts[shiftOut]->getCurrentObject());
#endif

    // if we have not asked for a dummy, then we may quit
    if (p_genDummyS->getValue())
    {
        // we have to check if there is at least one
        // visible object for output
        bool smthVisible = isVisible(outPorts[0 + shiftOut]->getCurrentObject());
        if (!smthVisible)
        {
            // we have to create dummies,
            // but before doing that,
            // we have to decide the attributes for the final objects
            attributeContainer *meshOutAttr = new attributeContainer(outPorts[0 + shiftOut]->getCurrentObject());
            attributeContainer *dataOutAttr = new attributeContainer(outPorts[1 + shiftOut]->getCurrentObject());
            attributeContainer *normalsOutAttr = new attributeContainer(outPorts[2 + shiftOut]->getCurrentObject());

            // and destroy invisible objects
            meshOutAttr->clean();
            dataOutAttr->clean();
            normalsOutAttr->clean();

            // now we may create dummy polygons (or tri-strips)
            coDistributedObject *dummy = NULL;
            float *dummyX, *dummyY, *dummyZ;
            if (!p_genstrips->getValue())
            {
                dummy = dummy_polygons(meshOutAttr->dummyName(),
                                       meshOutAttr->timeSteps(),
                                       &dummyX, &dummyY, &dummyZ);
            }
            else
            {
                dummy = dummy_tr_strips(meshOutAttr->dummyName(),
                                        meshOutAttr->timeSteps(),
                                        &dummyX, &dummyY, &dummyZ);
            }
            vector<pair<string, string> > vertexOrder;
            string attributeV = "vertexOrder";
            string contentV;
            if (p_gennormals->getValue())
            {
                contentV = "2";
            }
            else
            {
                contentV = "1";
            }
            vertexOrder.push_back(pair<string, string>(attributeV, contentV));
            vertexOrder.push_back(pair<string, string>("COLOR", "white"));
            meshOutAttr->addAttributes(dummy, vertexOrder);
            // now we may create dummy data
            coDistributedObject *dummyData = dummy_data(dataOutAttr->dummyName(),
                                                        dataOutAttr->timeSteps());
            dataOutAttr->addAttributes(dummyData, vector<pair<string, string> >());
            // now we may create dummy normals
            coDistributedObject *dummyNormals = dummy_normals(normalsOutAttr->dummyName(),
                                                              normalsOutAttr->timeSteps(),
                                                              dummyX, dummyY, dummyZ);
            normalsOutAttr->addAttributes(dummyNormals, vector<pair<string, string> >());

            delete meshOutAttr;
            delete dataOutAttr;
            delete normalsOutAttr;

            outPorts[0 + shiftOut]->setCurrentObject(dummy);
#ifndef _COMPLEX_MODULE_ // in the complex case, only for scalar data
            addFeedbackParams(dummy);
#endif
            outPorts[1 + shiftOut]->setCurrentObject(dummyData);
            outPorts[2 + shiftOut]->setCurrentObject(dummyNormals);
        }
    }
#ifdef _COMPLEX_MODULE_
    coDistributedObject *filth = p_GeometryOut->getCurrentObject();
    if (filth)
    {
        filth->destroy();
    }
    delete filth;
    p_GeometryOut->setCurrentObject(NULL);
    // we have to generate a coDoGeometry object for
    // outPorts[0], we have to distinguish the scalar and
    // vector case

    // get the output objects from the ports
    coDistributedObject *geo = p_MeshOut->getCurrentObject();
    coDistributedObject *data = p_DataOut->getCurrentObject();
    const coDistributedObject *norm = p_NormalsOut->getCurrentObject();
    const coDistributedObject *cmap = p_ColorMapIn->getCurrentObject();

    float min = FLT_MAX, max = -FLT_MAX;
    getMinMax(data, min, max);
    p_minmax->setValue(0, min);
    p_minmax->setValue(1, max);

    // SCALAR
    if (geo && (coObjectAlgorithms::containsType<const coDoFloat *>(p_DataIn->getCurrentObject()) || coObjectAlgorithms::containsType<const coDoByte *>(p_DataIn->getCurrentObject())))
    {
        // this is the easiest case...
        addFeedbackParams(geo);
        if (geo)
            geo->incRefCount();

        string geo_name = p_GeometryOut->getObjName();
        if (p_SampleGeom_->isConnected())
            geo_name += "_Cut";

        coDoGeometry *do_geom = new coDoGeometry(geo_name.c_str(), geo);
        if (norm)
        {
            norm->incRefCount();
            do_geom->setNormals(PER_VERTEX, norm);
        }

        if (p_color_or_texture->getValue()) // colors
        {
            string color_name = p_GeometryOut->getObjName();
            color_name += "_Color";
            coDistributedObject *color = ComplexModules::DataTexture(color_name, data, cmap, false);
            if (color)
                do_geom->setColors(PER_VERTEX, color);
        }
        else // texture
        {
            string texture_name = p_GeometryOut->getObjName();
            texture_name += "_Texture";
            coDistributedObject *texture = ComplexModules::DataTexture(texture_name, data, cmap, true);
            if (texture)
                do_geom->setTexture(0, texture);
        }
        if (!p_SampleGeom_->isConnected())
        {
            p_GeometryOut->setCurrentObject(do_geom);
        }
        else
        {
            const coDistributedObject **setList = new const coDistributedObject *[3];
            setList[0] = do_geom;
            string creatorModuleName = get_module();
            creatorModuleName += '_';
            creatorModuleName += get_instance();
            do_geom->addAttribute("CREATOR_MODULE_NAME", creatorModuleName.c_str());
            setList[1] = SampleToGeometry(p_SampleGeom_->getCurrentObject(), p_SampleData_->getCurrentObject());
            setList[2] = NULL;
            coDoSet *do_geometries = new coDoSet(p_GeometryOut->getObjName(), setList);
            delete do_geom;
            delete setList[1];
            delete[] setList;
            p_GeometryOut->setCurrentObject(do_geometries);
        }
    }

    // VECTOR
    else if (geo && coObjectAlgorithms::containsType<const coDoVec3 *>(p_DataIn->getCurrentObject()))
    {
        coDoSet *Geo = dynamic_cast<coDoSet *>(geo);
        if (Geo && geo->getAttribute("TIMESTEP"))
        {
            // open the timesteps and call StaticParts for each of them
            coDoSet *Data = dynamic_cast<coDoSet *>(data);
            if (!Data)
            {
                terminator.silent();
                return;
            }
            ScalarContainer ScalarCont;
            ScalarCont.Initialise(data);
            int no_e, no_d;
            const coDistributedObject *const *geoList = Geo->getAllElements(&no_e);
            const coDistributedObject *const *dataList = Data->getAllElements(&no_d);
            if (no_e != no_d)
            {
                terminator.silent();
                return;
            }
            coDistributedObject **GeoFullList = new coDistributedObject *[no_e + 1];
            coDistributedObject **NormFullList = new coDistributedObject *[no_e + 1];
            coDistributedObject **ColorFullList = new coDistributedObject *[no_e + 1];
            GeoFullList[no_e] = NULL;
            NormFullList[no_e] = NULL;
            ColorFullList[no_e] = NULL;
            int i;
            for (i = 0; i < no_e; ++i)
            {
                GeoFullList[i] = NULL;
                NormFullList[i] = NULL;
                ColorFullList[i] = NULL;
                string tstepName = p_GeometryOut->getObjName();
                tstepName += "_TStep_";
                char buf[16];
                sprintf(buf, "%d", i);
                tstepName += buf;
                StaticParts(&GeoFullList[i], &NormFullList[i], &ColorFullList[i],
                            geoList[i], dataList[i], tstepName, false, &ScalarCont);
            }
            string GeoAllStepsName = p_GeometryOut->getObjName();
            GeoAllStepsName += "_Geo_AllTS";
            coDoSet *GeoAllSteps = new coDoSet(GeoAllStepsName.c_str(),
                                               GeoFullList);
            string ColorAllStepsName = p_GeometryOut->getObjName();
            ColorAllStepsName += "_Color_AllTS";
            coDoSet *ColorAllSteps = new coDoSet(ColorAllStepsName.c_str(),
                                                 ColorFullList);
            string NormAllStepsName = p_GeometryOut->getObjName();
            NormAllStepsName += "_Norm_AllTS";
            coDoSet *NormAllSteps = new coDoSet(NormAllStepsName.c_str(),
                                                NormFullList);
            GeoAllSteps->copyAllAttributes(geo);
            if (GeoAllSteps->getAttribute("OBJECTNAME") == NULL)
            {
                GeoAllSteps->addAttribute("OBJECTNAME", getTitle());
            }
            ColorAllSteps->copyAllAttributes(data);
            // add COLORMAP to ColorAllSteps...
            coColors theColors(data, (coDoColormap *)(p_ColorMapIn->getCurrentObject()), false);
            theColors.addColormapAttrib(ColorAllSteps->getName(), ColorAllSteps);
            string geo_name = p_GeometryOut->getObjName();
            if (p_SampleGeom_->isConnected())
            {
                geo_name += "_Cut";
            }
            coDoGeometry *GeoOutput = new coDoGeometry(geo_name.c_str(), GeoAllSteps);
            GeoOutput->setNormals(PER_VERTEX, NormAllSteps);
            GeoOutput->setColors(PER_VERTEX, ColorAllSteps);
            addFeedbackParams(GeoAllSteps);
            if (!p_SampleGeom_->isConnected())
            {
                p_GeometryOut->setCurrentObject(GeoOutput);
            }
            else
            {
                coDistributedObject **setList = new coDistributedObject *[3];
                setList[0] = GeoOutput;
                string creatorModuleName = get_module();
                creatorModuleName += '_';
                creatorModuleName += get_instance();
                GeoOutput->addAttribute("CREATOR_MODULE_NAME", creatorModuleName.c_str());
                setList[1] = SampleToGeometry(p_SampleGeom_->getCurrentObject(), p_SampleData_->getCurrentObject());
                setList[2] = NULL;
                coDoSet *do_geometries = new coDoSet(p_GeometryOut->getObjName(), setList);
                delete GeoOutput;
                delete setList[1];
                delete[] setList;
                p_GeometryOut->setCurrentObject(do_geometries);
            }

            for (i = 0; i < no_e; ++i)
            {
                delete GeoFullList[i];
                if (NormFullList[i] != p_NormalsOut->getCurrentObject())
                    delete NormFullList[i];
                delete ColorFullList[i];
            }
            delete[] GeoFullList;
            delete[] NormFullList;
            delete[] ColorFullList;
        }
        else
        {
            coDistributedObject *geopart = NULL;
            coDistributedObject *normpart = NULL;
            coDistributedObject *colorpart = NULL;
            StaticParts(&geopart, &normpart, &colorpart, geo, data,
                        p_GeometryOut->getObjName());
            string geo_name = p_GeometryOut->getObjName();
            if (p_SampleGeom_->isConnected())
            {
                geo_name += "_Cut";
            }
            if (geopart->getAttribute("OBJECTNAME") == NULL)
            {
                geopart->addAttribute("OBJECTNAME", getTitle());
            }
            coDoGeometry *do_geom = new coDoGeometry(geo_name.c_str(), geopart);
            addFeedbackParams(geopart);
            do_geom->setNormals(PER_VERTEX, normpart);
            do_geom->setColors(PER_VERTEX, colorpart);
            if (!p_SampleGeom_->isConnected())
            {
                p_GeometryOut->setCurrentObject(do_geom);
            }
            else
            {
                coDistributedObject **setList = new coDistributedObject *[3];
                setList[0] = do_geom;
                string creatorModuleName = get_module();
                creatorModuleName += '_';
                creatorModuleName += get_instance();
                do_geom->addAttribute("CREATOR_MODULE_NAME", creatorModuleName.c_str());
                setList[1] = SampleToGeometry(p_SampleGeom_->getCurrentObject(), p_SampleData_->getCurrentObject());
                setList[2] = NULL;
                coDoSet *do_geometries = new coDoSet(p_GeometryOut->getObjName(), setList);
                delete do_geom;
                delete setList[1];
                delete[] setList;
                p_GeometryOut->setCurrentObject(do_geometries);
            }
        }
    }
    else
    {
        terminator.silent();
        Covise::sendWarning("Could not determine whether the input data field is scalar or vector");
    }
#endif
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
int CuttingSurfaceModule::compute(const char *)
{
    isDummy_ = false;
    if (p_option->getValue() == 0) // check that p_vertex is not null for a plane
    {
        if (p_vertex->getValue(0) == 0.0 && p_vertex->getValue(1) == 0.0 && p_vertex->getValue(2) == 0.0)
        {
            sendError("If the cutting surface is a plane, the 'vertex' parameter should not be null");
            return STOP_PIPELINE;
        }
    }
    // get output data object	names
    const char *GridOut = p_MeshOut->getObjName();
    const char *NormalsOut = p_NormalsOut->getObjName();
    const char *DataOut = p_DataOut->getObjName();

    //	get parameter

    float planei, planej, planek;
    float startx, starty, startz;
    float myDistance;
    float radius;
    int gennormals, genstrips;

    if (p_skew->getValue())
    {
        // make them slightly incorrect to prevent cutting exactly: avoid artefacts

        planei = param_vertex[0] * 1.00002f;
        planej = param_vertex[1] * 0.99999f;
        planek = param_vertex[2] * 1.00001f;

        startx = param_point[0] * 1.00001f;
        starty = param_point[1] * 0.99999f;
        startz = param_point[2] * 1.00001f;

        myDistance = param_scalar * 1.00001f;
    }
    else
    {
        planei = param_vertex[0];
        planej = param_vertex[1];
        planek = param_vertex[2];

        startx = param_point[0];
        starty = param_point[1];
        startz = param_point[2];

        myDistance = param_scalar;
    }
    gennormals = p_gennormals->getValue();
    genstrips = p_genstrips->getValue();

    radius = myDistance;
    if (param_option == 0)
    {
        float len;
        len = sqrt(planei * planei + planej * planej + planek * planek);
        planei /= len;
        planej /= len;
        planek /= len;
    }
    if (param_option == 1)
        radius = fabs(myDistance);
    if (param_option == 2)
    {
        float shift_planej = planej - starty;
        float shift_planek = planek - startz;
        if (shift_planej * shift_planej + shift_planek * shift_planek == 0.0)
            radius = 1.0;
        else
            radius = sqrt(shift_planej * shift_planej + shift_planek * shift_planek);
    }
    else if (param_option == 3)
    {
        float shift_planei = planei - startx;
        float shift_planek = planek - startz;
        if (shift_planei * shift_planei + shift_planek * shift_planek == 0.0)
            radius = 1.0;
        else
            radius = sqrt(shift_planei * shift_planei + shift_planek * shift_planek);
    }
    else if (param_option == 4)
    {
        float shift_planej = planej - starty;
        float shift_planei = planei - startx;
        if (shift_planej * shift_planej + shift_planei * shift_planei == 0.0)
            radius = 1.0;
        else
            radius = sqrt(shift_planej * shift_planej + shift_planei * shift_planei);
    }

    //	retrieve data object from shared memeory

    const coDistributedObject *grid_object;
    const coDistributedObject *data_object;
    const coDistributedObject *iblank_object;

    int numelem = 0, numconn, numcoord = 0, data_anz = 0, idata_anz = 0;

    float x_min, x_max, y_min, y_max, z_min, z_max;
    Plane *plane = NULL;
    STR_Plane *splane = NULL;
    RECT_Plane *rplane = NULL;

    const coDoUnstructuredGrid *grid_in = NULL;
    const coDoStructuredGrid *sgrid_in = NULL;
    const coDoUniformGrid *ugrid_in = NULL;
    const coDoRectilinearGrid *rgrid_in = NULL;
    int *el = NULL, *cl = NULL, *tl = NULL;
    char *iblank = NULL;
    float *x_in = NULL, *y_in = NULL, *z_in = NULL;
    float *s_in = NULL, *i_in = NULL;
    float *u_in = NULL, *v_in = NULL, *w_in = NULL;
    unsigned char *bs_in = NULL;

    int x_size, y_size, z_size;

    //  Shared memory data
    const coDoVec3 *uv_data_in = NULL;
    const coDoFloat *us_data_in = NULL;
    const coDoByte *ub_data_in = NULL;
    const coDoFloat *ui_data_in = NULL;
    const coDoFloat *minmax_data_in = NULL;

    grid_object = p_MeshIn->getCurrentObject();
    data_object = p_DataIn->getCurrentObject();
    iblank_object = p_IBlankIn->getCurrentObject();

    AttributeContainer gridAttrs(grid_object);
    AttributeContainer dataAttrs(data_object);

    if (iblank_object)
    {
        const coDoText *iblank_text = dynamic_cast<const coDoText *>(iblank_object);
        if (iblank_text)
            iblank_text->getAddress(&iblank);
    }
    if (data_object)
    {
        us_data_in = dynamic_cast<const coDoFloat *>(data_object);
        uv_data_in = dynamic_cast<const coDoVec3 *>(data_object);
        ub_data_in = dynamic_cast<const coDoByte *>(data_object);
        if (ub_data_in)
        {
            data_anz = ub_data_in->getNumPoints();
            ub_data_in->getAddress(&bs_in);
            DataType = 1;
        }
        else if (us_data_in)
        {
            data_anz = us_data_in->getNumPoints();
            us_data_in->getAddress(&s_in);
            DataType = 1;
        }
        else if (uv_data_in)
        {
            data_anz = uv_data_in->getNumPoints();
            uv_data_in->getAddresses(&u_in, &v_in, &w_in);
            DataType = 0;
        }
        else
        {
            sendError("Received illegal type at port '%s'", p_DataIn->getName());
            DoPostHandle = false;
            return -1;
        }
        if (data_anz == 0)
        {
            sendWarning("Data object '%s' is empty", p_DataIn->getName());
        }
    }
    else
    {
        sendError("Did not receive object at port '%s'", p_DataIn->getName());
        DoPostHandle = false;
        return -1;
    }

    if (grid_object)
    {
        grid_in = dynamic_cast<const coDoUnstructuredGrid *>(grid_object);
        ugrid_in = dynamic_cast<const coDoUniformGrid *>(grid_object);
        rgrid_in = dynamic_cast<const coDoRectilinearGrid *>(grid_object);
        sgrid_in = dynamic_cast<const coDoStructuredGrid *>(grid_object);
        if (grid_in)
        {
            grid_in->getGridSize(&numelem, &numconn, &numcoord);
            grid_in->getAddresses(&el, &cl, &x_in, &y_in, &z_in);
            grid_in->getTypeList(&tl);
            comp_borders(numcoord, x_in, y_in, z_in);
        }
        else if (ugrid_in)
        {
            ugrid_in->getGridSize(&x_size, &y_size, &z_size);
            ugrid_in->getMinMax(&x_min, &x_max, &y_min, &y_max, &z_min, &z_max);
            numcoord = x_size * y_size * z_size;
            numelem = ((x_size - 1) * (y_size - 1) * (z_size - 1));
            if (x_max > x_maxb)
                x_maxb = x_max;
            if (y_max > y_maxb)
                y_maxb = y_max;
            if (z_max > z_maxb)
                z_maxb = z_max;

            if (x_min < x_minb)
                x_minb = x_min;
            if (y_min < y_minb)
                y_minb = y_min;
            if (z_min < z_minb)
                z_minb = z_min;
        }
        else if (rgrid_in)
        {
            rgrid_in->getGridSize(&x_size, &y_size, &z_size);
            rgrid_in->getAddresses(&x_in, &y_in, &z_in);
            numcoord = x_size * y_size * z_size;
            numelem = ((x_size - 1) * (y_size - 1) * (z_size - 1));
            comp_borders(x_size, x_in, NULL, NULL);
            comp_borders(y_size, NULL, y_in, NULL);
            comp_borders(z_size, NULL, NULL, z_in);
        }
        else if (sgrid_in)
        {
            sgrid_in->getGridSize(&x_size, &y_size, &z_size);
            sgrid_in->getAddresses(&x_in, &y_in, &z_in);
            numcoord = x_size * y_size * z_size;
            numelem = ((x_size - 1) * (y_size - 1) * (z_size - 1));
            comp_borders(numcoord, x_in, y_in, z_in);
        }
        else
        {
            sendWarning("Received illegal type '%s' at port '%s'", grid_object->getType(), p_MeshIn->getName());

            p_MeshOut->setCurrentObject(0);
            p_DataOut->setCurrentObject(0);
            if (gennormals)
                p_NormalsOut->setCurrentObject(0);
            return CONTINUE_PIPELINE;
        }
    }
    else
    {
        sendError("Did not receive oject at port '%s'", p_MeshIn->getName());
        return -1;
    }

    //	retrieve Iso data object from shared memeory
    i_in = NULL; // if no isodata given use normal data

    if (data_anz != numcoord)
    {
        Covise::sendError("ERROR: Dataobject's dimension doesn't match Grid ones");
        DoPostHandle = false;
        return -1;
    }

    if (grid_in)
    {
        if ((Polyhedra) && (param_option == 0)) // polyhedra currently only supported for planar cuts
        {
            genstrips = false;
            plane = new POLYHEDRON_Plane(numelem, numcoord, DataType, el, cl, tl,
                                         x_in, y_in, z_in,
                                         s_in, bs_in, i_in,
                                         u_in, v_in, w_in,
                                         sgrid_in, grid_in, maxPolyPerVertex, planei, planej, planek, startx, starty, startz, myDistance,
                                         radius, gennormals, param_option, genstrips, iblank);
        }
        else
        {
            plane = new Plane(numelem, numcoord, DataType, el, cl, tl,
                              x_in, y_in, z_in,
                              s_in, bs_in, i_in,
                              u_in, v_in, w_in,
                              sgrid_in, grid_in, vertexAllocRatio, maxPolyPerVertex, planei, planej, planek, startx, starty, startz, myDistance,
                              radius, gennormals, param_option, genstrips, iblank);
        }

        // plane->set_min_max(x_minb, y_minb, z_minb, x_maxb, y_maxb, z_maxb);

        // if we couldn't do it correctly - re-run
        if (!plane->createPlane())
        {
            vertexAllocRatio += 1.0; //increase by 100%
            p_vertexratio->setValue(vertexAllocRatio);

            Covise::sendInfo("Increased VERTEX_RATIO to %f and re-exec", vertexAllocRatio);
            // send execute message after 0.2 sec for network latency
            setExecGracePeriod(0.2f);
            selfExec();

            delete plane;
            return -1;
        }
    }

    if (ugrid_in) // handle as rect. grids
    {
        rplane = new RECT_Plane(numelem, numcoord, DataType,
                                el, cl, tl, x_in, y_in, z_in, s_in, bs_in, i_in, u_in, v_in, w_in,
                                ugrid_in, x_size, y_size, z_size, maxPolyPerVertex,
                                planei, planej, planek, startx, starty, startz, myDistance, radius,
                                gennormals, param_option, genstrips, iblank);
        // rplane->set_min_max(x_minb, y_minb, z_minb, x_maxb, y_maxb, z_maxb);
        rplane->createPlane();
        plane = rplane;
    }
    if (rgrid_in)
    {
        rplane = new RECT_Plane(numelem, numcoord, DataType, el, cl, tl, x_in, y_in, z_in,
                                s_in, bs_in, i_in, u_in, v_in, w_in, rgrid_in, x_size, y_size, z_size,
                                maxPolyPerVertex,
                                planei, planej, planek, startx, starty, startz,
                                myDistance, radius, gennormals, param_option, genstrips, iblank);
        // rplane->set_min_max(x_minb, y_minb, z_minb, x_maxb, y_maxb, z_maxb);
        rplane->createPlane();
        plane = rplane;
    }
    if (sgrid_in)
    {
        splane = new STR_Plane(numelem, numcoord, DataType, el, cl, tl, x_in, y_in, z_in,
                               s_in, bs_in, i_in, u_in, v_in, w_in, sgrid_in, grid_in, x_size, y_size, z_size,
                               maxPolyPerVertex, planei, planej, planek, startx, starty, startz,
                               myDistance, radius, gennormals, param_option, genstrips, iblank);
        // splane->set_min_max(x_minb, y_minb, z_minb, x_maxb, y_maxb, z_maxb);
        splane->createPlane();
        plane = splane;
    }

    if (DataType)
        plane->createcoDistributedObjects(DataOut, NULL, NormalsOut, GridOut, gridAttrs, dataAttrs);
    else
        plane->createcoDistributedObjects(NULL, DataOut, NormalsOut, GridOut, gridAttrs, dataAttrs);

    if (genstrips)
        p_MeshOut->setCurrentObject(plane->get_obj_strips());
    else
        p_MeshOut->setCurrentObject(plane->get_obj_pol());
    if (gennormals)
        p_NormalsOut->setCurrentObject(plane->get_obj_normal());
    if (DataType)
        p_DataOut->setCurrentObject(plane->get_obj_scalar());
    else
        p_DataOut->setCurrentObject(plane->get_obj_vector());

    if (grid_in)
    {
        delete plane;
    }
    if (ugrid_in)
    {
        delete rplane;
    }
    if (rgrid_in)
    {
        delete rplane;
    }
    if (sgrid_in)
    {
        delete splane;
    }

    return CONTINUE_PIPELINE;
}

#ifdef _COMPLEX_MODULE_

void
CuttingSurfaceModule::StaticParts(coDistributedObject **geopart,
                                  coDistributedObject **normpart,
                                  coDistributedObject **colorpart,
                                  const coDistributedObject *geo, // FIXME
                                  const coDistributedObject *data, // FIXME
                                  string geometryOutName,
                                  bool ColorMapAttrib,
                                  const ScalarContainer *SCont)
{
    // coDistributedObject *geo = p_MeshOut->getCurrentObject();
    int vectOption = p_vector->getValue();
    if (isDummy_)
        vectOption = 1;
    if (geo != NULL && vectOption == 0)
    {
        geo->incRefCount();
    }
    string nameArrows = geometryOutName; //p_GeometryOut->getObjName();
    if (vectOption == 2)
    {
        nameArrows += "_Geom";
    }
    else if (vectOption == 0)
    {
        nameArrows += "_Arrows";
    }
    else
    {
        nameArrows = "";
    }
    float factor = 1.0;

    string color_name = geometryOutName; //p_GeometryOut->getObjName();
    color_name += "_Color";
    coDistributedObject *colorLines = NULL;
    coDistributedObject *colorSurface = NULL;

    coDistributedObject *arrows = ComplexModules::MakeArrows(nameArrows.c_str(),
                                                             geo, data,
                                                             color_name.c_str(),
                                                             &colorSurface, &colorLines, factor, p_ColorMapIn->getCurrentObject(),
                                                             ColorMapAttrib, SCont, p_scale->getValue(), p_length->getValue() + 1,
                                                             p_num_sectors->getValue(), p_arrow_head_factor->getValue(), p_arrow_head_angle->getValue(), p_project_lines->getValue(), vectOption);

    if (vectOption == 0)
    {
        const coDistributedObject **setList = new const coDistributedObject *[3];
        setList[2] = NULL;
        setList[0] = geo;
        setList[1] = arrows;
        string nameGeom = geometryOutName; // p_GeometryOut->getObjName();
        nameGeom += "_Geom";
        *geopart = new coDoSet(nameGeom.c_str(), setList);
        // delete geo;
        delete arrows;
        delete[] setList;
    }
    else if (vectOption == 1)
    {
        string nameGeom = geometryOutName;
        nameGeom += "_Geom";
        *geopart = geo->clone(nameGeom);
    }
    else
    {
        *geopart = arrows;
    }
    // coDoGeometry *do_geom = new coDoGeometry(p_GeometryOut->getObjName(), geopart);
    //delete geopart;

    // now we have to create a set of normals...
    coDistributedObject *norm = p_NormalsOut->getCurrentObject();
    if (norm && vectOption != 2)
    {
        norm->incRefCount();
        if (vectOption == 0)
        {
            string normArrowsName = norm->getName();
            normArrowsName += "_Arrows";
            // dummy normals for lines
            coDistributedObject *normArrows = new coDoVec3(normArrowsName.c_str(), 0);

            coDistributedObject **setList = new coDistributedObject *[3];
            setList[2] = NULL;
            setList[0] = norm;
            setList[1] = normArrows;
            string nameNorm = geometryOutName; // p_GeometryOut->getObjName();
            nameNorm += "_Norm";
            *normpart = new coDoSet(nameNorm.c_str(), setList);
            // delete norm;
            delete normArrows;
            delete[] setList;
        }
        else // 2
        {
            *normpart = norm;
        }
        // do_geom->setNormal(PER_VERTEX,normpart);
    }
    // now we have to create a set of colors...
    /*
      color_name += "_Dummy";
      coDistributedObject *colorDummy = new coDoRGBA(color_name.c_str(),0);
   */
    if (vectOption == 0)
    {
        coDistributedObject **setList = new coDistributedObject *[3];
        setList[2] = NULL;
        setList[0] = colorSurface;
        setList[1] = colorLines;
        string nameColor = geometryOutName; //p_GeometryOut->getObjName();
        nameColor += "_AllColor";
        *colorpart = new coDoSet(nameColor.c_str(), setList);
        delete colorLines;
        delete[] setList;
    }
    else if (vectOption == 1)
    {
        *colorpart = colorSurface;
    }
    else if (vectOption == 2)
    {
        *colorpart = colorLines;
    }
    //do_geom->setColors(PER_VERTEX,colorpart);
}
#endif

#ifdef _COMPLEX_MODULE_
void
CuttingSurfaceModule::getMinMax(const coDistributedObject *obj, float &min, float &max)
{
    min = FLT_MAX;
    max = -FLT_MAX;

    if (obj == NULL)
        return;

    if (obj->isType("SETELE"))
    {
        int nb = 0;
        float local_min, local_max;
        const coDistributedObject *const *set = ((coDoSet *)obj)->getAllElements(&nb);
        for (int i = 0; i < nb; i++)
        {
            getMinMax(set[i], local_min, local_max);
            if (local_min < min)
                min = local_min;
            if (local_max > max)
                max = local_max;
        }
    }
    else if (const coDoByte *bobj = dynamic_cast<const coDoByte *>(obj))
    {
        unsigned char bmin = 0xff, bmax = 0x00;
        const unsigned char *b = bobj->getAddress();
        int num_points = bobj->getNumPoints();
        for (int i = 0; i < num_points; i++)
        {
            if (b[i] < bmin)
                bmin = b[i];
            if (b[i] > bmax)
                bmax = b[i];
        }
        if (bmin / 255.f < min)
            min = bmin / 255.f;
        if (bmax / 255.f > max)
            max = bmax / 255.f;
    }
    else if (obj->isType("USTSDT"))
    {
        float *s;
        ((coDoFloat *)obj)->getAddress(&s);
        int num_points = ((coDoFloat *)obj)->getNumPoints();
        for (int i = 0; i < num_points; i++)
        {
            if (s[i] < min)
                min = s[i];
            if (s[i] > max)
                max = s[i];
        }
    }
    else if (obj->isType("USTVDT"))
    {
        float *u, *v, *w, len;
        ((coDoVec3 *)obj)->getAddresses(&u, &v, &w);
        int num_points = ((coDoVec3 *)obj)->getNumPoints();
        for (int i = 0; i < num_points; i++)
        {
            len = sqrt(u[i] * u[i] + v[i] * v[i] + w[i] * w[i]);
            if (len < min)
                min = len;
            if (len > max)
                max = len;
        }
    }
}
#endif

MODULE_MAIN(Filter, CuttingSurfaceModule)
