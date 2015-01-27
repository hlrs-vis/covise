/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description:  COVISE CropUsg application module                        **
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
 ** Date:  27.01.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "CropUsg.h"
enum
{
    NUMSCALAR = 5
};

int CropUsg::CheckDimensions(const coDistributedObject *grid, const coDoVec3 *vobj, const coDoFloat **sobjs)
{
    // the module accepts only data per vertex...
    int numelem, numconn, numcoord;
    if (const coDoUnstructuredGrid *ugrid = dynamic_cast<const coDoUnstructuredGrid *>(grid))
        ugrid->getGridSize(&numelem, &numconn, &numcoord);
    else if (const coDoPolygons *poly = dynamic_cast<const coDoPolygons *>(grid))
        numcoord = poly->getNumPoints();
    else if (const coDoLines *lines = dynamic_cast<const coDoLines *>(grid))
        numcoord = lines->getNumPoints();
    else if (const coDoPoints *points = dynamic_cast<const coDoPoints *>(grid))
        numcoord = points->getNumPoints();
    else
    {
        Covise::sendError("Only unstr. grids, polygons, lines and points are accepted");
        return -1;
    }

    int dataLength = 0;
    if (vobj)
        dataLength = vobj->getNumPoints();
    if (vobj && dataLength && dataLength != numcoord)
    {
        Covise::sendError("Vector data size has to match the number of nodes");
        return -1;
    }
    for (int i = 0; i < NUMSCALAR; ++i)
    {
        if (sobjs[i])
        {
            dataLength = sobjs[i]->getNumPoints();
            if (dataLength && dataLength != numcoord)
            {
                //Covise::sendError("Scalar data size has to match the number of nodes");
                //return -1;
                sDataPerElem[i] = true;
            }
            else

                sDataPerElem[i] = false;
        }
    }
    return 0;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
CropUsg::CropUsg(int argc, char *argv[])
    : coSimpleModule(argc, argv, "reduce the extents of an unstructured grid, polygonal mesh, lines or points")
{
    // Parameters

    // create the parameters
    const char *method_labels[] = { "GeoCrop", "DataCrop" };
    p_method = addChoiceParam("method", "crop mesh based on data or based on geometry");
    p_method->setValue(2, method_labels, 0);

    p_type_ = addBooleanParam("boundingBox", "use min / max values for each coordinate direction? Otherwise use plane which is defined by a point and a normal vector");
    p_type_->setValue(1);

    p_Xmin_ = addFloatParam("xMin", "X Minimum");
    p_Xmax_ = addFloatParam("xMax", "X Maximum");
    p_Ymin_ = addFloatParam("yMin", "Y Minimum");
    p_Ymax_ = addFloatParam("yMax", "Y Maximum");
    p_Zmin_ = addFloatParam("zMin", "Z Minimum");
    p_Zmax_ = addFloatParam("zMax", "Z Maximum");
    p_invert_crop_ = addBooleanParam("invert_crop", "Invert Crop Area");

    p_normal_ = addFloatVectorParam("normal", "normal vector of plane to cut mesh with");
    p_normal_->setValue(0., 0., 1.);

    p_point_ = addFloatVectorParam("basepoint", "point on plane to cut mesh with");
    p_point_->setValue(0., 0., 0.);

    p_data_min = addFloatParam("data_min", "smallest data value, polygons with smaller values will be removed");
    p_data_max = addFloatParam("data_max", "biggest data value, polygons with bigger values will be removed");
    p_strict_removal = addBooleanParam("strict_selection", "one vertex out of bounds is enough to erase polygon");

    // set the dafault values
    const float defaultMin = -2000.0;
    const float defaultMax = 2000.0;

    p_Xmin_->setValue(defaultMin);
    p_Xmax_->setValue(defaultMax);
    p_Ymin_->setValue(defaultMin);
    p_Ymax_->setValue(defaultMax);
    p_Zmin_->setValue(defaultMin);
    p_Zmax_->setValue(defaultMax);
    p_invert_crop_->setValue(0);

    p_strict_removal->setValue(true);
    p_data_min->setValue(0.0);
    p_data_max->setValue(1.0);

    // Ports

    p_GridInPort_ = addInputPort("GridIn0", "UnstructuredGrid|Polygons|Lines|Points", "Unstructured Grid, Polygons, Lines or Points");
    p_GridOutPort_ = addOutputPort("GridOut0", "UnstructuredGrid|Polygons|Lines|Points", "reduced mesh");

    p_vDataOutPort_ = addOutputPort("DataOut0", "Vec3", "reduced Vertex Data");
    p_vDataInPort_ = addInputPort("DataIn0", "Vec3", "Vertex Data");

    p_sData1OutPort_ = addOutputPort("DataOut1", "Float", "reduced Scalar Data 1");
    p_sData1InPort_ = addInputPort("DataIn1", "Float", "Scalar Data 1");

    p_sData2OutPort_ = addOutputPort("DataOut2", "Float", "reduced Scalar Data 2");
    p_sData2InPort_ = addInputPort("DataIn2", "Float", "Scalar Data 2");

    p_sData3OutPort_ = addOutputPort("DataOut3", "Float", "reduced Scalar Data 3");
    p_sData3InPort_ = addInputPort("DataIn3", "Float", "Scalar Data 3");

    p_sData4OutPort_ = addOutputPort("DataOut4", "Float", "reduced Scalar Data 4");
    p_sData4InPort_ = addInputPort("DataIn4", "Float", "Scalar Data 4");

    p_sData5OutPort_ = addOutputPort("DataOut5", "Float", "reduced Scalar Data 5");
    p_sData5InPort_ = addInputPort("DataIn5", "Float", "Scalar Data 5");

    p_paramOutPort_ = addOutputPort("DataOut6", "Points", "current parameters");
    p_paramInPort_ = addInputPort("DataIn6", "Points", "adjust parameters");

    // port attributes
    const int notRequired = 0;
    p_vDataInPort_->setRequired(notRequired);
    p_sData1InPort_->setRequired(notRequired);
    p_sData2InPort_->setRequired(notRequired);
    p_sData3InPort_->setRequired(notRequired);
    p_sData4InPort_->setRequired(notRequired);
    p_sData5InPort_->setRequired(notRequired);
    p_paramInPort_->setRequired(notRequired);

    p_vDataOutPort_->setDependencyPort(p_vDataInPort_);
    p_sData1OutPort_->setDependencyPort(p_sData1InPort_);
    p_sData2OutPort_->setDependencyPort(p_sData2InPort_);
    p_sData3OutPort_->setDependencyPort(p_sData3InPort_);
    p_sData4OutPort_->setDependencyPort(p_sData4InPort_);
    p_sData5OutPort_->setDependencyPort(p_sData5InPort_);

    setComputeTimesteps(0);
    setComputeMultiblock(0);
    // and the API should take care of attributes
    setCopyAttributes(1);
}

void
CropUsg::postInst()
{
    p_data_min->disable();
    p_data_max->disable();

    p_normal_->disable();
    p_point_->disable();
}

void
CropUsg::param(const char *paramname, bool /*in_map_loading*/)
{
    if (strcmp(paramname, p_method->getName()) == 0)
    {
        switch (p_method->getValue())
        {
        case 0:
            p_Xmin_->enable();
            p_Xmax_->enable();
            p_Ymin_->enable();
            p_Ymax_->enable();
            p_Zmin_->enable();
            p_Zmax_->enable();
            p_data_min->disable();
            p_data_max->disable();
            p_normal_->disable();
            p_point_->disable();
            break;

        case 1:
            p_Xmin_->disable();
            p_Xmax_->disable();
            p_Ymin_->disable();
            p_Ymax_->disable();
            p_Zmin_->disable();
            p_Zmax_->disable();
            p_data_min->enable();
            p_data_max->enable();
            p_normal_->enable();
            p_point_->enable();
            break;
        }
    }

    if (strcmp(paramname, p_type_->getName()) == 0)
    {
        switch ((int)p_type_->getValue())
        {
        case 1:
            p_Xmin_->enable();
            p_Xmax_->enable();
            p_Ymin_->enable();
            p_Ymax_->enable();
            p_Zmin_->enable();
            p_Zmax_->enable();
            p_invert_crop_->enable();
            p_normal_->disable();
            p_point_->disable();
            break;

        case 0:
            p_Xmin_->disable();
            p_Xmax_->disable();
            p_Ymin_->disable();
            p_Ymax_->disable();
            p_Zmin_->disable();
            p_Zmax_->disable();
            p_invert_crop_->disable();
            p_normal_->enable();
            p_point_->enable();
            break;
        }
    }
}

void
CropUsg::preHandleObjects(coInputPort **inports)
{

    if (inports[NUMSCALAR + 2]->getCurrentObject())
    {
        if (const coDoPoints *box = dynamic_cast<const coDoPoints *>(inports[NUMSCALAR + 2]->getCurrentObject()))
        {
            if (box->getNumPoints() == 2)
            {
                float *x, *y, *z;
                box->getAddresses(&x, &y, &z);
                p_Xmin_->setValue(x[0]);
                p_Xmax_->setValue(x[1]);
                p_Ymin_->setValue(y[0]);
                p_Ymax_->setValue(y[1]);
                p_Zmin_->setValue(z[0]);
                p_Zmax_->setValue(z[1]);
            }
            else
            {
                sendWarning("Number of Points at param_in bigger than two. Please correct. Parameters not changed.");
            }
        }
        else
        {
            sendWarning("Parameter input not of type coDoPoints. Parameters not changed.");
        }
    }

    // put params out

    float x[2], y[2], z[2];
    x[0] = p_Xmin_->getValue();
    x[1] = p_Xmax_->getValue();
    y[0] = p_Ymin_->getValue();
    y[1] = p_Ymax_->getValue();
    z[0] = p_Zmin_->getValue();
    z[1] = p_Zmax_->getValue();

    coDoPoints *params_out = new coDoPoints(p_paramOutPort_->getObjName(), 2, x, y, z);
    if (params_out && params_out->objectOk())
    {
        p_paramOutPort_->setCurrentObject(params_out);
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  compute() is called once for every EXECUTE message
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
int
CropUsg::compute(const char *)
{

    const coDistributedObject *gridInObj = p_GridInPort_->getCurrentObject();

    // we should have an object
    if (!gridInObj)
    {
        sendError("Did not receive object at port '%s'", p_GridInPort_->getName());
        return FAIL;
    }
    // it should be the correct type
    if (!dynamic_cast<const coDoUnstructuredGrid *>(gridInObj)
        && !dynamic_cast<const coDoPolygons *>(gridInObj)
        && !dynamic_cast<const coDoLines *>(gridInObj)
        && !dynamic_cast<const coDoPoints *>(gridInObj))
    {
        sendError("Received illegal type at port '%s'", p_GridInPort_->getName());
        return FAIL;
    }

    // optional objects are retrieved

    // vector data
    const coDistributedObject *vDataObj = p_vDataInPort_->getCurrentObject();
    if (vDataObj != NULL)
    {
        if (!dynamic_cast<const coDoVec3 *>(vDataObj))
        {
            sendError("The object at port vdata has wrong type");
            return STOP_PIPELINE;
        }
    }

    // arrange scalar data so that reduce and reduce_poly can work on it
    const coDoFloat **sDataInArr = new const coDoFloat *[NUMSCALAR];
    sDataInArr[0] = (const coDoFloat *)p_sData1InPort_->getCurrentObject();
    sDataInArr[1] = (const coDoFloat *)p_sData2InPort_->getCurrentObject();
    sDataInArr[2] = (const coDoFloat *)p_sData3InPort_->getCurrentObject();
    sDataInArr[3] = (const coDoFloat *)p_sData4InPort_->getCurrentObject();
    sDataInArr[4] = (const coDoFloat *)p_sData5InPort_->getCurrentObject();
    sDataPerElem = new bool[NUMSCALAR];

    coDoFloat **sDataOut;
    sDataOut = new coDoFloat *[NUMSCALAR];

    numSclData_ = 0;
    for (int i = 0; i < NUMSCALAR; ++i)
    {
        if (sDataInArr[i] != NULL)
        {
            if (dynamic_cast<const coDoFloat *>(sDataInArr[i]))
                ++numSclData_;
            else
            {
                sendError("The object at port sdata%d has wrong type", i + 1);
                return STOP_PIPELINE;
            }
        }
        else if ((i == 0) && (p_method->getValue() == 1))
        {
            sendError("data object at port sdata%d required for method DataCrop", i + 1);
            return STOP_PIPELINE;
        }
        // else  sl: sDataOut[i] should be initialised to 0 in any case
        //           (as it is done with vDataOut below),
        //           otherwise the modul crashes if the cropped grid is null
        sDataOut[i] = NULL;
    }

    if (CheckDimensions(gridInObj,
                        (coDoVec3 *)(vDataObj),
                        sDataInArr) < 0)
    {
        return STOP_PIPELINE;
    }

    // prepare names of scalar data
    sNames = new const char *[NUMSCALAR];
    sNames[0] = p_sData1OutPort_->getObjName();
    sNames[1] = p_sData2OutPort_->getObjName();
    sNames[2] = p_sData3OutPort_->getObjName();
    sNames[3] = p_sData4OutPort_->getObjName();
    sNames[4] = p_sData5OutPort_->getObjName();

    float switch_p;
    if (p_method->getValue() == 0)
    {
        Xmin_ = p_Xmin_->getValue();
        Xmax_ = p_Xmax_->getValue();
    }
    else
    {
        Xmin_ = p_data_min->getValue();
        Xmax_ = p_data_max->getValue();
    }
    if (Xmin_ > Xmax_) // sk: added checks 08.02.2001
    {
        switch_p = Xmin_;
        Xmin_ = Xmax_;
        Xmax_ = switch_p;
    }
    Ymin_ = p_Ymin_->getValue();
    Ymax_ = p_Ymax_->getValue();
    if (Ymin_ > Ymax_)
    {
        switch_p = Ymin_;
        Ymin_ = Ymax_;
        Ymax_ = switch_p;
    }
    Zmin_ = p_Zmin_->getValue();
    Zmax_ = p_Zmax_->getValue();
    if (Zmin_ > Zmax_)
    {
        switch_p = Zmin_;
        Zmin_ = Zmax_;
        Zmax_ = switch_p;
    }

    if (const coDoUnstructuredGrid *inGrid = dynamic_cast<const coDoUnstructuredGrid *>(gridInObj))
    {
        // fprintf(stderr, "CropUsg::compute(const char *) working on USGR\n");
        // we have an unstructured grid
        coDoVec3 *vData = (coDoVec3 *)vDataObj;
        coDoUnstructuredGrid *unsGridOut = NULL;
        coDoVec3 *vDataOut = NULL;

        //fprintf(stderr, "CropUsg::compute(const char *) reduce\n");
        // makes the computation
        reduce(inGrid, vData, sDataInArr,
               &unsGridOut, &vDataOut, &sDataOut,
               p_GridOutPort_->getObjName(),
               p_vDataOutPort_->getObjName(),
               sNames);

        // assign grid output data
        if (unsGridOut == 0)
        {
            unsGridOut = new coDoUnstructuredGrid(p_GridOutPort_->getObjName(), 0, 0, 0, 1);
        }
        // sl: already done by coSimpleModule
        unsGridOut->copyAllAttributes(inGrid);
        p_GridOutPort_->setCurrentObject(unsGridOut);

        // assign vector and scalar output data
        common_assign(vData, vDataOut, sDataInArr, sDataOut);
    }
    else if (const coDoPolygons *inPolygons = dynamic_cast<const coDoPolygons *>(gridInObj))
    {
        // we have an unstructured polygons
        coDoVec3 *vdata = (coDoVec3 *)vDataObj;
        coDoPolygons *outPolygons = NULL;
        coDoVec3 *vDataOut = NULL;

        // makes the computation
        reduce_poly(inPolygons, vdata, sDataInArr,
                    &outPolygons, &vDataOut, &sDataOut,
                    (char *)p_GridOutPort_->getObjName(),
                    (char *)p_vDataOutPort_->getObjName(),
                    sNames);

        // assign grid output data
        if (outPolygons == 0)
        {
            outPolygons = new coDoPolygons(p_GridOutPort_->getObjName(), 0, 0, 0);
        }
        // sl: already done by coSimpleModule
        outPolygons->copyAllAttributes(inPolygons);
        p_GridOutPort_->setCurrentObject(outPolygons);

        // assign vector and scalar output data
        common_assign(vdata, vDataOut, sDataInArr, sDataOut);
    }
    else if (const coDoLines *inLines = dynamic_cast<const coDoLines *>(gridInObj))
    {
        // we have an unstructured lines object
        coDoVec3 *vdata = (coDoVec3 *)vDataObj;
        coDoLines *outLines = NULL;
        coDoVec3 *vDataOut = NULL;

        // makes the computation
        reduce_poly(inLines, vdata, sDataInArr,
                    &outLines, &vDataOut, &sDataOut,
                    (char *)p_GridOutPort_->getObjName(),
                    (char *)p_vDataOutPort_->getObjName(),
                    sNames);

        // assign grid output data
        if (outLines == 0)
        {
            outLines = new coDoLines(p_GridOutPort_->getObjName(), 0, 0, 0);
        }
        // sl: already done by coSimpleModule
        outLines->copyAllAttributes(inLines);
        p_GridOutPort_->setCurrentObject(outLines);

        // assign vector and scalar output data
        common_assign(vdata, vDataOut, sDataInArr, sDataOut);
    }
    if (gridInObj->isType("POINTS"))
    {

        // we have an unstructured polygons
        coDoPoints *inPoints = (coDoPoints *)gridInObj;
        coDoVec3 *vdata = (coDoVec3 *)vDataObj;
        coDoPoints *outPoints = NULL;
        coDoVec3 *vDataOut = NULL;

        // makes the computation
        reduce_poly(inPoints, vdata, sDataInArr,
                    &outPoints, &vDataOut, &sDataOut,
                    (char *)p_GridOutPort_->getObjName(),
                    (char *)p_vDataOutPort_->getObjName(),
                    sNames);

        // assign grid output data
        if (outPoints == 0)
        {
            outPoints = new coDoPoints(p_GridOutPort_->getObjName(), 0);
        }
        // sl: already done by coSimpleModule
        outPoints->copyAllAttributes(inPoints);
        p_GridOutPort_->setCurrentObject(outPoints);

        // assign vector and scalar output data
        common_assign(vdata, vDataOut, sDataInArr, sDataOut);
    }

    // clean up
    delete[] sNames;

    return SUCCESS;
}

// assign vector and scalar objects
void CropUsg::common_assign(const coDoVec3 *vdata,
                            coDoVec3 *vDataOut,
                            const coDoFloat **sDataInArr,
                            coDoFloat **sDataOut)
{
    if (vdata)
    {
        if (vDataOut == 0)
        {
            vDataOut = new coDoVec3(p_vDataOutPort_->getObjName(), 0);
        }
        p_vDataOutPort_->setCurrentObject(vDataOut);
    }
    if (sDataInArr[0])
    {
        if (sDataOut[0] == 0)
        {
            sDataOut[0] = new coDoFloat(sNames[0], 0);
        }
        p_sData1OutPort_->setCurrentObject(sDataOut[0]);
    }
    if (sDataInArr[1])
    {
        if (sDataOut[1] == 0)
        {
            sDataOut[1] = new coDoFloat(sNames[1], 0);
        }
        p_sData2OutPort_->setCurrentObject(sDataOut[1]);
    }
    if (sDataInArr[2])
    {
        if (sDataOut[2] == 0)
        {
            sDataOut[2] = new coDoFloat(sNames[2], 0);
        }
        p_sData3OutPort_->setCurrentObject(sDataOut[2]);
    }
    if (sDataInArr[3])
    {
        if (sDataOut[3] == 0)
        {
            sDataOut[3] = new coDoFloat(sNames[3], 0);
        }
        p_sData4OutPort_->setCurrentObject(sDataOut[3]);
    }
    if (sDataInArr[4])
    {
        if (sDataOut[4] == 0)
        {
            sDataOut[4] = new coDoFloat(sNames[4], 0);
        }
        p_sData5OutPort_->setCurrentObject(sDataOut[4]);
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  reduce(coDoUnstructuredGrid *grid_in,
// ++++		coDoVec3 *vdata,
// ++++		coDoFloat **sdata,
// ++++		coDoUnstructuredGrid **grid_out,
// ++++		coDoVec3 **vodata,
// ++++		coDoFloat ***sodata,
// ++++		char *Gname, char *Vname, char **Snames)
// ++++
// ++++  takes grid data and crops the grid
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void
CropUsg::reduce(const coDoUnstructuredGrid *grid_in,
                const coDoVec3 *vdata,
                const coDoFloat **sdata,
                coDoUnstructuredGrid **grid_out,
                coDoVec3 **vodata,
                coDoFloat ***sodata,
                const char *Gname, const char *Vname, const char **Snames)
{
    float *x_in, *y_in, *z_in, *x_out, *y_out, *z_out,
        *v1_out, *v2_out, *v3_out,
        *v1_in, *v2_in, *v3_in,
        **sin = NULL, **sout = NULL;
    int *new_el, *new_vl, *new_tl,
        *el, *tl, *vl, numelem, numconn, numcoord, i, n,
        new_numelem = 0, new_numconn = 0, new_numcoord = 0;
    int *used;
    int *isused;
    int *new_used;

    grid_in->getGridSize(&numelem, &numconn, &numcoord);
    grid_in->getAddresses(&el, &vl, &x_in, &y_in, &z_in);
    grid_in->getTypeList(&tl);
    isused = new int[numelem];
    memset(isused, 1, numelem * sizeof(int));

    new_el = new int[numelem];
    new_tl = new int[numelem];
    new_vl = new int[numconn];
    used = new int[numcoord];
    new_used = new int[numcoord];

    // retrieve parameters
    int invert = p_invert_crop_->getValue();
    int numdata = 0;
    //int celldata=0;

    // int offset=0;
    // int lastnr=0;
    int outsiders = 0;
    bool strict_removal = p_strict_removal->getValue();

    if (p_method->getValue() == 1)
    {
        numdata = sdata[0]->getNumPoints(); // data can be defined on vertices or on cells
        if (numdata == numcoord)
        {
            //celldata=0;
        }
        else if (numdata == numelem)
        {
            //celldata=1;
        }
        else
        {
            Covise::sendError("data dimension does not fit with number of vertices nor with number of elements");
        }
    }

    if (numSclData_)
    {
        sin = new float *[numSclData_];
        sout = new float *[numSclData_];
    }
    float *data = NULL;
    if (p_method->getValue() == 0) // "GeoCrop"
    {
        if (p_type_->getValue()) // BoundingBox
        {
            if (invert == 0)
            {
                for (i = 0; i < numcoord; i++)
                {
                    if (outside(x_in[i], y_in[i], z_in[i]))
                    {
                        used[i] = -1;
                        new_used[i] = -1;
                    }
                    else
                    {
                        used[i] = new_numcoord;
                        new_used[i] = new_numcoord;
                        new_numcoord++;
                    }
                }
            }
            else // invert == 1
            {
                for (i = 0; i < numcoord; i++)
                {
                    if (outside(x_in[i], y_in[i], z_in[i]))
                    {
                        used[i] = new_numcoord;
                        new_used[i] = new_numcoord;
                        new_numcoord++;
                    }
                    else
                    {
                        used[i] = -1;
                        new_used[i] = -1;
                    }
                }
            }
        }
        else // plane
        {
            nn[0] = p_normal_->getValue(0);
            nn[1] = p_normal_->getValue(1);
            nn[2] = p_normal_->getValue(2);

            pp[0] = p_point_->getValue(0);
            pp[1] = p_point_->getValue(1);
            pp[2] = p_point_->getValue(2);

            // calc d for HNF
            float d = pp[0] * nn[0] + pp[1] * nn[1] + pp[2] * nn[2];

            for (i = 0; i < numcoord; i++)
            {
                if (dist_positive(x_in[i], y_in[i], z_in[i], d))
                {
                    used[i] = new_numcoord;
                    new_numcoord++;
                }
                else
                {
                    used[i] = -1;
                }
            }
        }
    }
    else // "DataCrop"
    {
        sdata[0]->getAddress(&data);
        if (!sDataPerElem[0])
        {
            if (invert == 0)
            {
                for (i = 0; i < numdata; i++)
                {
                    if (outside(data[i]))
                    {
                        used[i] = -1;
                        new_used[i] = -1;
                    }
                    else
                    {
                        used[i] = new_numcoord;
                        new_used[i] = new_numcoord;
                        new_numcoord++;
                    }
                }
            }
            else // invert == 1
            {
                for (i = 0; i < numdata; i++)
                {
                    if (outside(data[i]))
                    {
                        used[i] = new_numcoord;
                        new_used[i] = new_numcoord;
                        new_numcoord++;
                    }
                    else
                    {
                        used[i] = -1;
                        new_used[i] = -1;
                    }
                }
            }
        }
    }
    if (data != NULL && sDataPerElem[0])
    {
        for (i = 0; i < numcoord; i++)
            used[i] = -1;
        for (i = 0; i < numelem; i++)
        {
            if (outside(data[i]))
                isused[i] = 0;
            else
            {
                int num_p;
                if (i == (numelem - 1))
                    num_p = numconn - el[i];
                else
                    num_p = el[i + 1] - el[i];
                for (n = 0; n < num_p; n++)
                {
                    if (used[vl[el[i] + n]] < 0)
                    {
                        used[vl[el[i] + n]] = new_numcoord++;
                    }
                }
            }
        }
    }
    else
    {

        for (i = 0; i < numelem; i++)
        {

            int numConnOfElem = grid_in->getNumConnOfElement(i);
            outsiders = 0;
            if (strict_removal) // one "outside" vertex is enough to delete element
            {
                for (n = 0; n < numConnOfElem; n++)
                    if (used[vl[el[i] + n]] < 0)
                    {
                        isused[i] = 0;
                        break;
                    }
            }
            else // all vertices must be "outside" to delete element
            {
                for (n = 0; n < numConnOfElem; n++)
                {
                    if (used[vl[el[i] + n]] < 0)
                    {
                        outsiders++;
                    }
                }
                if (outsiders == numConnOfElem)
                {
                    isused[i] = 0;
                }
                else
                {
                    if ((outsiders > 0) && (outsiders < numConnOfElem))
                    {
                        for (n = 0; n < numConnOfElem; n++)
                        {
                            // used[vl[el[i]+n]]=-2;
                            if (used[vl[el[i] + n]] < 0 && new_used[vl[el[i] + n]] < 0)
                            {
                                new_used[vl[el[i] + n]] = new_numcoord;
                                new_numcoord++;
                            }
                        }
                    }
                }
            }
        }
    }

    //    if (!strict_removal)
    //    {
    //       for(i=0;i<numcoord;i++)
    //       {
    //         if (used[i]==-2)
    //         {
    //            offset++;
    //            used[i]=lastnr+offset;
    //         }
    //         else if (used[i]!=-1)
    //
    //         {
    //            lastnr=used[i];
    //            used[i]+=offset;
    //         }
    //       }
    //       new_numcoord+=offset;
    //    }

    for (i = 0; i < numelem; i++)
    {
        if (isused[i])
        {
            new_el[new_numelem] = new_numconn;
            new_tl[new_numelem] = tl[i];
            new_numelem++;

            int numConnOfElem = grid_in->getNumConnOfElement(i);
            for (n = 0; n < numConnOfElem; n++)
                // new_vl[new_numconn++]=used[vl[el[i]+n]];
                new_vl[new_numconn++] = new_used[vl[el[i] + n]];
        }
    }

    if (new_numelem)
    {
        if (vdata)
        {
            vdata->getAddresses(&v1_in, &v2_in, &v3_in);
            if (vdata->getNumPoints())
            {
                (*vodata) = new coDoVec3(Vname, new_numcoord);
                (*vodata)->getAddresses(&v1_out, &v2_out, &v3_out);
                for (i = 0; i < numcoord; i++)
                {
                    // if(used[i]>=0)
                    if (new_used[i] >= 0)
                    {
                        // v1_out[used[i]]=v1_in[i];
                        // v2_out[used[i]]=v2_in[i];
                        // v3_out[used[i]]=v3_in[i];

                        v1_out[new_used[i]] = v1_in[i];
                        v2_out[new_used[i]] = v2_in[i];
                        v3_out[new_used[i]] = v3_in[i];
                    }
                }
            }
            else
            {
                (*vodata) = new coDoVec3(Vname, 0);
            }
        }
        n = 0;
        if (numSclData_)
        {
            for (i = 0; i < NUMSCALAR; i++)
            {
                if (sdata[i] != NULL)
                {
                    if (sDataPerElem[i])
                    {
                        if (sdata[i]->getNumPoints())
                        {
                            sdata[i]->getAddress(sin + n);
                            (*sodata)[i] = new coDoFloat(Snames[i], new_numelem);
                            (*sodata)[i]->getAddress(sout + n);
                            int ne = 0;
                            for (i = 0; i < numelem; i++)
                            {
                                if (isused[i])
                                {
                                    sout[n][ne] = sin[n][i];
                                    ne++;
                                }
                            }
                            sout[n] = 0;
                        }
                        else
                        {
                            (*sodata)[i] = new coDoFloat(Snames[i], 0);
                            sout[n] = 0;
                        }
                    }
                    else
                    {
                        if (sdata[i]->getNumPoints())
                        {
                            sdata[i]->getAddress(sin + n);

                            (*sodata)[i] = new coDoFloat(Snames[i], new_numcoord);
                            (*sodata)[i]->getAddress(sout + n);
                        }
                        else
                        {
                            (*sodata)[i] = new coDoFloat(Snames[i], 0);
                            sout[n] = 0;
                        }
                    }
                    n++;
                }
            }
        }
        (*grid_out) = new coDoUnstructuredGrid(Gname, new_numelem, new_numconn, new_numcoord, 1);

        (*grid_out)->getAddresses(&el, &vl, &x_out, &y_out, &z_out);
        (*grid_out)->getTypeList(&tl);

        memcpy(el, new_el, new_numelem * sizeof(int));
        memcpy(tl, new_tl, new_numelem * sizeof(int));
        memcpy(vl, new_vl, new_numconn * sizeof(int));

        for (i = 0; i < numcoord; i++)
        {
            // if(used[i]>=0)
            if (new_used[i] >= 0)
            {
                // x_out[used[i]]=x_in[i];
                // y_out[used[i]]=y_in[i];
                // z_out[used[i]]=z_in[i];

                x_out[new_used[i]] = x_in[i];
                y_out[new_used[i]] = y_in[i];
                z_out[new_used[i]] = z_in[i];

                for (n = 0; n < numSclData_; n++)
                    if (sout[n])
                        // sout[n][used[i]]=sin[n][i];
                        sout[n][new_used[i]] = sin[n][i];
            }
        }
        //  		  if(cname!=NULL)
        //  			(*grid_out)->addAttribute("COLOR",cname);
        //  		  if(vertname!=NULL)
        //  			(*grid_out)->addAttribute("vertexOrder",vertname);
    }
    delete[] new_el;
    delete[] new_tl;
    delete[] new_vl;
    delete[] used;
    delete[] isused;
    delete[] new_used;

    if (numSclData_)
    {
        delete[] sin;
        delete[] sout;
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  reduce_poly(coDoPolygons *grid_in,
// ++++	    coDoVec3 *vdata,
// ++++	    coDoFloat **sdata,
// ++++	    coDoPolygons **grid_out,
// ++++	    coDoVec3 **vodata,
// ++++	    coDoFloat ***sodata,
// ++++	    char *Gname,char *Vname,char **Snames)
// ++++
// ++++
// ++++  takes polygons and data and crops it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void
CropUsg::reduce_poly(const coDoPolygons *grid_in,
                     const coDoVec3 *vdata,
                     const coDoFloat **sdata,
                     coDoPolygons **grid_out,
                     coDoVec3 **vodata,
                     coDoFloat ***sodata,
                     const char *Gname, const char *Vname, const char **Snames)
{
    float *x_in, *y_in, *z_in, *x_out, *y_out, *z_out,
        *v1_out, *v2_out, *v3_out,
        *v1_in, *v2_in, *v3_in,
        **sin = NULL, **sout = NULL;
    int *new_el, *new_vl,
        *el, *vl, numelem, numconn, numcoord, i, n,
        new_numelem = 0, new_numconn = 0, new_numcoord = 0;
    int *used;
    int *isused;

    numcoord = grid_in->getNumPoints();
    numconn = grid_in->getNumVertices();
    numelem = grid_in->getNumPolygons();

    isused = new int[numelem];
    memset(isused, 1, numelem * sizeof(int));

    grid_in->getAddresses(&x_in, &y_in, &z_in, &vl, &el);

    new_el = new int[numelem];
    new_vl = new int[numconn];

    used = new int[numcoord];

    // retrieve parameters
    int invert = p_invert_crop_->getValue();
    int numdata = 0;
    //int celldata=0;

    int offset = 0;
    int lastnr = 0;
    int outsiders = 0;
    bool strict_removal = p_strict_removal->getValue();

    float *data = NULL;
    if (p_method->getValue() == 1)
    {
        numdata = sdata[0]->getNumPoints(); // data can be defined on vertices or on cells
        if (numdata == numcoord)
        {
            //celldata=0;
        }
        else if (numdata == numelem)
        {
            //celldata=1;
        }
        else
        {
            Covise::sendError("data dimension does not fit with number of vertices nor with number of elements");
        }
    }

    if (numSclData_)
    {
        sin = new float *[numSclData_];
        sout = new float *[numSclData_];
    }

    if (p_method->getValue() == 0) // "GeoCrop"
    {
        for (i = 0; i < numcoord; i++)
        {
            if (invert == 0)
            {
                if (outside(x_in[i], y_in[i], z_in[i]))
                {
                    used[i] = -1;
                }
                else
                {
                    used[i] = new_numcoord;
                    new_numcoord++;
                }
            }
            else // invert == 1
            {
                if (outside(x_in[i], y_in[i], z_in[i]))
                {
                    used[i] = new_numcoord;
                    new_numcoord++;
                }
                else
                {
                    used[i] = -1;
                }
            }
        }
    }
    else // "DataCrop"
    {
        sdata[0]->getAddress(&data);
        if (!sDataPerElem[0])
        {
            for (i = 0; i < numdata; i++)
            {
                if (invert == 0)
                {
                    if (outside(data[i]))
                    {
                        used[i] = -1;
                    }
                    else
                    {
                        used[i] = new_numcoord;
                        new_numcoord++;
                    }
                }
                else // invert == 1
                {
                    if (outside(data[i]))
                    {
                        used[i] = new_numcoord;
                        new_numcoord++;
                    }
                    else
                    {
                        used[i] = -1;
                    }
                }
            }
        }
    }

    if (data != NULL && sDataPerElem[0])
    {
        for (i = 0; i < numcoord; i++)
            used[i] = -1;
        for (i = 0; i < numelem; i++)
        {
            if (outside(data[i]))
                isused[i] = 0;
            else
            {
                int num_p;
                if (i == (numelem - 1))
                    num_p = numconn - el[i];
                else
                    num_p = el[i + 1] - el[i];
                for (n = 0; n < num_p; n++)
                {
                    if (used[vl[el[i] + n]] < 0)
                    {
                        used[vl[el[i] + n]] = new_numcoord++;
                    }
                }
            }
        }
    }
    else
    {
        for (i = 0; i < numelem; i++)
        {
            outsiders = 0;
            int num_p;
            if (i == (numelem - 1))
                num_p = numconn - el[i];
            else
                num_p = el[i + 1] - el[i];
            if (strict_removal) // one "outside" vertex is enough to delete element
            {
                for (n = 0; n < num_p; n++)
                {
                    if (used[vl[el[i] + n]] < 0)
                    {
                        isused[i] = 0;
                        break;
                    }
                }
            }
            else // all vertices must be "outside" to delete element
            {
                for (n = 0; n < num_p; n++)
                {
                    if (used[vl[el[i] + n]] < 0)
                    {
                        outsiders++;
                    }
                }
                if (outsiders == num_p)
                {
                    isused[i] = 0;
                }
                else
                {
                    if ((outsiders > 0) && (outsiders < num_p))
                    {
                        for (n = 0; n < num_p; n++)
                        {
                            used[vl[el[i] + n]] = -2;
                        }
                    }
                }
            }
        }
    }

    if (!strict_removal)
    {
        for (i = 0; i < numcoord; i++)
        {
            if (used[i] == -2)
            {
                offset++;
                used[i] = lastnr + offset;
            }
            else if (used[i] != -1)
            {
                lastnr = used[i];
                used[i] += offset;
            }
        }
        new_numcoord += offset;
    }

    for (i = 0; i < numelem; i++)
    {
        if (isused[i])
        {
            int num_p;
            if (i == (numelem - 1))
                num_p = numconn - el[i];
            else
                num_p = el[i + 1] - el[i];
            new_el[new_numelem] = new_numconn;
            new_numelem++;
            for (n = 0; n < num_p; n++)
                new_vl[new_numconn++] = used[vl[el[i] + n]];
        }
    }

    if (new_numelem)
    {
        if (vdata != NULL)
        {
            vdata->getAddresses(&v1_in, &v2_in, &v3_in);
            if (vdata->getNumPoints())
            {
                (*vodata) = new coDoVec3(Vname, new_numcoord);
                (*vodata)->getAddresses(&v1_out, &v2_out, &v3_out);
                for (i = 0; i < numcoord; i++)
                {
                    if (used[i] >= 0)
                    {
                        v1_out[used[i]] = v1_in[i];
                        v2_out[used[i]] = v2_in[i];
                        v3_out[used[i]] = v3_in[i];
                    }
                }
            }
            else
            {
                (*vodata) = new coDoVec3(Vname, 0);
            }
        }
        n = 0;
        if (numSclData_)
        {
            for (i = 0; i < NUMSCALAR; i++)
            {
                if (sdata[i] != 0L)
                {
                    if (sDataPerElem[i])
                    {
                        if (sdata[i]->getNumPoints())
                        {
                            sdata[i]->getAddress(sin + n);
                            (*sodata)[i] = new coDoFloat(Snames[i], new_numelem);
                            (*sodata)[i]->getAddress(sout + n);
                            int ne = 0;
                            for (int ni = 0; ni < numelem; ni++)
                            {
                                if (isused[ni])
                                {
                                    sout[n][ne] = sin[n][ni];
                                    ne++;
                                }
                            }
                            sout[n] = 0;
                        }
                        else
                        {
                            (*sodata)[i] = new coDoFloat(Snames[i], 0);
                            sout[n] = 0;
                        }
                    }
                    else
                    {
                        if (sdata[i]->getNumPoints())
                        {
                            sdata[i]->getAddress(sin + n);
                            (*sodata)[i] = new coDoFloat(Snames[i], new_numcoord);
                            (*sodata)[i]->getAddress(sout + n);
                        }
                        else
                        {
                            (*sodata)[i] = new coDoFloat(Snames[i], 0);
                            sout[n] = 0;
                        }
                    }
                    n++;
                }
            }
        }
        (*grid_out) = new coDoPolygons(Gname, new_numcoord, new_numconn, new_numelem);
        (*grid_out)->getAddresses(&x_out, &y_out, &z_out, &vl, &el);
        memcpy(el, new_el, new_numelem * sizeof(int));
        memcpy(vl, new_vl, new_numconn * sizeof(int));
        for (i = 0; i < numcoord; i++)
        {
            if (used[i] >= 0)
            {
                x_out[used[i]] = x_in[i];
                y_out[used[i]] = y_in[i];
                z_out[used[i]] = z_in[i];
                for (n = 0; n < numSclData_; n++)
                    if (sout[n])
                        sout[n][used[i]] = sin[n][i];
            }
        }
        //  		  if(cname!=NULL)
        //  			(*grid_out)->addAttribute("COLOR",cname);
        //  		  if(vertname!=NULL)
        //  			(*grid_out)->addAttribute("vertexOrder",vertname);
    }
    delete[] new_el;
    delete[] new_vl;
    delete[] used;
    delete[] isused;
    if (numSclData_)
    {
        delete[] sin;
        delete[] sout;
    }
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  reduce_poly(coDoPolygons *grid_in,
// ++++	    coDoVec3 *vdata,
// ++++	    coDoFloat **sdata,
// ++++	    coDoPolygons **grid_out,
// ++++	    coDoVec3 **vodata,
// ++++	    coDoFloat ***sodata,
// ++++	    char *Gname,char *Vname,char **Snames)
// ++++
// ++++
// ++++  takes polygons and data and crops it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void
CropUsg::reduce_poly(const coDoLines *grid_in,
                     const coDoVec3 *vdata,
                     const coDoFloat **sdata,
                     coDoLines **grid_out,
                     coDoVec3 **vodata,
                     coDoFloat ***sodata,
                     const char *Gname, const char *Vname, const char **Snames)
{
    float *x_in, *y_in, *z_in, *x_out, *y_out, *z_out,
        *v1_out, *v2_out, *v3_out,
        *v1_in, *v2_in, *v3_in,
        **sin = NULL, **sout = NULL;
    int *new_el, *new_vl,
        *el, *vl, numelem, numconn, numcoord, i, n,
        new_numelem = 0, new_numconn = 0, new_numcoord = 0;
    int *used;
    int *isused;

    numcoord = grid_in->getNumPoints();
    numconn = grid_in->getNumVertices();
    numelem = grid_in->getNumLines();

    isused = new int[numelem];
    memset(isused, 1, numelem * sizeof(int));

    grid_in->getAddresses(&x_in, &y_in, &z_in, &vl, &el);

    new_el = new int[numelem];
    new_vl = new int[numconn];

    used = new int[numcoord];

    // retrieve parameters
    int invert = p_invert_crop_->getValue();
    int numdata = 0;
    //int celldata=0;

    int offset = 0;
    int lastnr = 0;
    int outsiders = 0;
    bool strict_removal = p_strict_removal->getValue();

    float *data = NULL;
    if (p_method->getValue() == 1)
    {
        numdata = sdata[0]->getNumPoints(); // data can be defined on vertices or on cells
        if (numdata == numcoord)
        {
            //celldata=0;
        }
        else if (numdata == numelem)
        {
            //celldata=1;
        }
        else
        {
            Covise::sendError("data dimension does not fit with number of vertices nor with number of elements");
        }
    }

    if (numSclData_)
    {
        sin = new float *[numSclData_];
        sout = new float *[numSclData_];
    }

    if (p_method->getValue() == 0) // "GeoCrop"
    {
        for (i = 0; i < numcoord; i++)
        {
            if (invert == 0)
            {
                if (outside(x_in[i], y_in[i], z_in[i]))
                {
                    used[i] = -1;
                }
                else
                {
                    used[i] = new_numcoord;
                    new_numcoord++;
                }
            }
            else // invert == 1
            {
                if (outside(x_in[i], y_in[i], z_in[i]))
                {
                    used[i] = new_numcoord;
                    new_numcoord++;
                }
                else
                {
                    used[i] = -1;
                }
            }
        }
    }
    else // "DataCrop"
    {
        sdata[0]->getAddress(&data);
        if (!sDataPerElem[0])
        {
            for (i = 0; i < numdata; i++)
            {
                if (invert == 0)
                {
                    if (outside(data[i]))
                    {
                        used[i] = -1;
                    }
                    else
                    {
                        used[i] = new_numcoord;
                        new_numcoord++;
                    }
                }
                else // invert == 1
                {
                    if (outside(data[i]))
                    {
                        used[i] = new_numcoord;
                        new_numcoord++;
                    }
                    else
                    {
                        used[i] = -1;
                    }
                }
            }
        }
    }

    if (data != NULL && sDataPerElem[0])
    {
        for (i = 0; i < numcoord; i++)
            used[i] = -1;
        for (i = 0; i < numelem; i++)
        {
            if (outside(data[i]))
                isused[i] = 0;
            else
            {
                int num_p;
                if (i == (numelem - 1))
                    num_p = numconn - el[i];
                else
                    num_p = el[i + 1] - el[i];
                for (n = 0; n < num_p; n++)
                {
                    if (used[vl[el[i] + n]] < 0)
                    {
                        used[vl[el[i] + n]] = new_numcoord++;
                    }
                }
            }
        }
    }
    else
    {
        for (i = 0; i < numelem; i++)
        {
            outsiders = 0;
            int num_p;
            if (i == (numelem - 1))
                num_p = numconn - el[i];
            else
                num_p = el[i + 1] - el[i];
            if (strict_removal) // one "outside" vertex is enough to delete element
            {
                for (n = 0; n < num_p; n++)
                {
                    if (used[vl[el[i] + n]] < 0)
                    {
                        isused[i] = 0;
                        break;
                    }
                }
            }
            else // all vertices must be "outside" to delete element
            {
                for (n = 0; n < num_p; n++)
                {
                    if (used[vl[el[i] + n]] < 0)
                    {
                        outsiders++;
                    }
                }
                if (outsiders == num_p)
                {
                    isused[i] = 0;
                }
                else
                {
                    if ((outsiders > 0) && (outsiders < num_p))
                    {
                        for (n = 0; n < num_p; n++)
                        {
                            used[vl[el[i] + n]] = -2;
                        }
                    }
                }
            }
        }
    }

    if (!strict_removal)
    {
        for (i = 0; i < numcoord; i++)
        {
            if (used[i] == -2)
            {
                offset++;
                used[i] = lastnr + offset;
            }
            else if (used[i] != -1)
            {
                lastnr = used[i];
                used[i] += offset;
            }
        }
        new_numcoord += offset;
    }

    for (i = 0; i < numelem; i++)
    {
        if (isused[i])
        {
            int num_p;
            if (i == (numelem - 1))
                num_p = numconn - el[i];
            else
                num_p = el[i + 1] - el[i];
            new_el[new_numelem] = new_numconn;
            new_numelem++;
            for (n = 0; n < num_p; n++)
                new_vl[new_numconn++] = used[vl[el[i] + n]];
        }
    }

    if (new_numelem)
    {
        if (vdata != NULL)
        {
            vdata->getAddresses(&v1_in, &v2_in, &v3_in);
            if (vdata->getNumPoints())
            {
                (*vodata) = new coDoVec3(Vname, new_numcoord);
                (*vodata)->getAddresses(&v1_out, &v2_out, &v3_out);
                for (i = 0; i < numcoord; i++)
                {
                    if (used[i] >= 0)
                    {
                        v1_out[used[i]] = v1_in[i];
                        v2_out[used[i]] = v2_in[i];
                        v3_out[used[i]] = v3_in[i];
                    }
                }
            }
            else
            {
                (*vodata) = new coDoVec3(Vname, 0);
            }
        }
        n = 0;
        if (numSclData_)
        {
            for (i = 0; i < NUMSCALAR; i++)
            {
                if (sdata[i] != 0L)
                {
                    if (sDataPerElem[i])
                    {
                        if (sdata[i]->getNumPoints())
                        {
                            sdata[i]->getAddress(sin + n);
                            (*sodata)[i] = new coDoFloat(Snames[i], new_numelem);
                            (*sodata)[i]->getAddress(sout + n);
                            int ne = 0;
                            for (i = 0; i < numelem; i++)
                            {
                                if (isused[i])
                                {
                                    sout[n][ne] = sin[n][i];
                                    ne++;
                                }
                            }
                            sout[n] = 0;
                        }
                        else
                        {
                            (*sodata)[i] = new coDoFloat(Snames[i], 0);
                            sout[n] = 0;
                        }
                    }
                    else
                    {
                        if (sdata[i]->getNumPoints())
                        {
                            sdata[i]->getAddress(sin + n);
                            (*sodata)[i] = new coDoFloat(Snames[i], new_numcoord);
                            (*sodata)[i]->getAddress(sout + n);
                        }
                        else
                        {
                            (*sodata)[i] = new coDoFloat(Snames[i], 0);
                            sout[n] = 0;
                        }
                    }
                    n++;
                }
            }
        }
        (*grid_out) = new coDoLines(Gname, new_numcoord, new_numconn, new_numelem);
        (*grid_out)->getAddresses(&x_out, &y_out, &z_out, &vl, &el);
        memcpy(el, new_el, new_numelem * sizeof(int));
        memcpy(vl, new_vl, new_numconn * sizeof(int));
        for (i = 0; i < numcoord; i++)
        {
            if (used[i] >= 0)
            {
                x_out[used[i]] = x_in[i];
                y_out[used[i]] = y_in[i];
                z_out[used[i]] = z_in[i];
                for (n = 0; n < numSclData_; n++)
                    if (sout[n])
                        sout[n][used[i]] = sin[n][i];
            }
        }
        //  		  if(cname!=NULL)
        //  			(*grid_out)->addAttribute("COLOR",cname);
        //  		  if(vertname!=NULL)
        //  			(*grid_out)->addAttribute("vertexOrder",vertname);
    }
    delete[] new_el;
    delete[] new_vl;
    delete[] used;
    delete[] isused;
    if (numSclData_)
    {
        delete[] sin;
        delete[] sout;
    }
}

MODULE_MAIN(Filter, CropUsg)
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  reduce_poly(coDoPoints *grid_in,
// ++++	    coDoVec3 *vdata,
// ++++	    coDoFloat **sdata,
// ++++	    coDoPoints **grid_out,
// ++++	    coDoVec3 **vodata,
// ++++	    coDoFloat ***sodata,
// ++++	    char *Gname,char *Vname,char **Snames)
// ++++
// ++++
// ++++  takes points and data and crops it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void
CropUsg::reduce_poly(const coDoPoints *grid_in,
                     const coDoVec3 *vdata,
                     const coDoFloat **sdata,
                     coDoPoints **grid_out,
                     coDoVec3 **vodata,
                     coDoFloat ***sodata,
                     const char *Gname, const char *Vname, const char **Snames)
{
    float *x_in, *y_in, *z_in, *x_out, *y_out, *z_out,
        *v1_out, *v2_out, *v3_out,
        *v1_in, *v2_in, *v3_in,
        **sin = NULL, **sout = NULL;
    int numcoord, i, n, new_numcoord = 0;
    int *used;

    numcoord = grid_in->getNumPoints();

    grid_in->getAddresses(&x_in, &y_in, &z_in);

    used = new int[numcoord];

    // retrieve parameters
    int invert = p_invert_crop_->getValue();

    float *data = NULL;
    if (numSclData_)
    {
        sin = new float *[numSclData_];
        sout = new float *[numSclData_];
    }

    if (p_method->getValue() == 0) // "GeoCrop"
    {
        for (i = 0; i < numcoord; i++)
        {
            if (invert == 0)
            {
                if (outside(x_in[i], y_in[i], z_in[i]))
                {
                    used[i] = -1;
                }
                else
                {
                    used[i] = new_numcoord;
                    new_numcoord++;
                }
            }
            else // invert == 1
            {
                if (outside(x_in[i], y_in[i], z_in[i]))
                {
                    used[i] = new_numcoord;
                    new_numcoord++;
                }
                else
                {
                    used[i] = -1;
                }
            }
        }
    }
    else // "DataCrop"
    {
        sdata[0]->getAddress(&data);

        if (!sDataPerElem[0])
        {
            for (i = 0; i < numcoord; i++)
            {
                if (invert == 0)
                {
                    if (outside(data[i]))
                    {
                        used[i] = -1;
                    }
                    else
                    {
                        used[i] = new_numcoord;
                        new_numcoord++;
                    }
                }
                else // invert == 1
                {
                    if (outside(data[i]))
                    {
                        used[i] = new_numcoord;
                        new_numcoord++;
                    }
                    else
                    {
                        used[i] = -1;
                    }
                }
            }
        }
    }

    if (new_numcoord)
    {
        if (vdata != NULL)
        {
            vdata->getAddresses(&v1_in, &v2_in, &v3_in);
            if (vdata->getNumPoints())
            {
                (*vodata) = new coDoVec3(Vname, new_numcoord);
                (*vodata)->getAddresses(&v1_out, &v2_out, &v3_out);
                for (i = 0; i < numcoord; i++)
                {
                    if (used[i] >= 0)
                    {
                        v1_out[used[i]] = v1_in[i];
                        v2_out[used[i]] = v2_in[i];
                        v3_out[used[i]] = v3_in[i];
                    }
                }
            }
            else
            {
                (*vodata) = new coDoVec3(Vname, 0);
            }
        }
        n = 0;
        if (numSclData_)
        {
            for (i = 0; i < NUMSCALAR; i++)
            {
                if (sdata[i] != 0L)
                {
                    if (sdata[i]->getNumPoints())
                    {
                        sdata[i]->getAddress(sin + n);
                        (*sodata)[i] = new coDoFloat(Snames[i], new_numcoord);
                        (*sodata)[i]->getAddress(sout + n);
                    }
                    else
                    {
                        (*sodata)[i] = new coDoFloat(Snames[i], 0);
                        sout[n] = 0;
                    }
                    n++;
                }
            }
        }
        (*grid_out) = new coDoPoints(Gname, new_numcoord);
        (*grid_out)->getAddresses(&x_out, &y_out, &z_out);
        for (i = 0; i < numcoord; i++)
        {
            if (used[i] >= 0)
            {
                x_out[used[i]] = x_in[i];
                y_out[used[i]] = y_in[i];
                z_out[used[i]] = z_in[i];
                for (n = 0; n < numSclData_; n++)
                    if (sout[n])
                        sout[n][used[i]] = sin[n][i];
            }
        }
    }
    delete[] used;
    if (numSclData_)
    {
        delete[] sin;
        delete[] sout;
    }
}

int CropUsg::dist_positive(float x, float y, float z, float d)
{
    // calc distance to plane defined by our parameters

    // n[0-2] ... normal vector of plane
    // p[0-2] ... base point of plane
    // d      ... distance to origin (not normalized)

    if ((x * nn[0] + y * nn[1] + z * nn[2] - d) >= 0)
        return 1;
    else
        return 0;
}
