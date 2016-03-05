/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description:  COVISE Isosurface application module                     **
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
 ** Date:  23.07.96  V1.0                                                  **
\**************************************************************************/

#include "IsoSurface.h"
#include <alg/coIsoSurface.h>
#include <alg/coColors.h>
#include <do/covise_gridmethods.h>
#include <util/coviseCompat.h>
#include <config/CoviseConfig.h>
#include <util/covise_version.h>
#include <api/coFeedback.h>
#include <do/coDoText.h>
#include <do/coDoData.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoGeometry.h>
#include <util/coWristWatch.h>

#ifdef _COMPLEX_MODULE_
#include <alg/coComplexModules.h>
#include <covise/covise_objalg.h>
#endif

coWristWatch ww_;

static inline double sqr(float x)
{
    return double(x) * double(x);
}

int numiso, set_num_elem = 0, cur_elem, cur_line_elem;
/*
int cuc_count;
int *cuc = NULL, *cuc_pos = NULL;
*/
float startpt;
int num_coord;
char buf[1000];
const char *dtype, *gtype;
const char *GridIn, *DataIn, *IsoDataIn;

//  Shared memory data
const coDoFloat *s_data_in = NULL;
const coDoVec3 *v_data_in = NULL;
const coDoFloat *i_data_in = NULL;
const coDoUnstructuredGrid *grid_in = NULL;
const coDoStructuredGrid *sgrid_in = NULL;
const coDoUniformGrid *ugrid_in = NULL;
const coDoRectilinearGrid *rgrid_in = NULL;
coDoSet *polygons_set_out, *normals_set_out,
    *data_set_out;

//.....................................
/*
int *el,*cl,*tl;
float *x_in;
float *y_in;
float *z_in;
*/

float *s_in;
float *i_in;
float *u_in;
float *v_in;
float *w_in;
//.....................................

int find_startcell_fast(const float *p,
                        const float *x_in, const float *y_in, const float *z_in,
                        int numelem, int numconn,
                        const int *el, const int *cl, const int *tl,
                        int cuc_count, const int *cuc, const int *cuc_pos);
void find_uni_value(float point[3], float &isowert);
void find_rct_value(float point[3], float &isowert);
void find_str_value(float point[3], float &isowert);

void NullInputData(coOutputPort *p_GridOut, coOutputPort *p_NormalsOut, coOutputPort *p_DataOut, int DataType, int gennormals, int genstrips, const char *colorn);

// sl: With this function you may apply the previous
//     functions to the case of a single element of any type
//     The output emulates a structured grid with a single
//     element obtained from an unstructured grid with a single
//     element.
static void fill_lists(float isoscalar[8],
                       float x_list[8], float y_list[8], float z_list[8],
                       float *x_coord, float *y_coord, float *z_coord,
                       int code[8], int ini_cell, float *data, int *cl)
{
    for (int i = 0; i < 8; ++i)
    {
        isoscalar[i] = data[cl[ini_cell + code[i]]];
        x_list[i] = x_coord[cl[ini_cell + code[i]]];
        y_list[i] = y_coord[cl[ini_cell + code[i]]];
        z_list[i] = z_coord[cl[ini_cell + code[i]]];
    }
}

void
IsoSurface::UpdateIsoValue()
{
    if (!p_IsoDataIn->getCurrentObject())
    {
        return;
    }
    _scalarName = p_IsoDataIn->getCurrentObject()->getName();
    ScalarContainer scalarField;
    scalarField.Initialise(p_IsoDataIn->getCurrentObject());
    scalarField.MinMax(_min, _max);
    if (_min <= _max)
    {
        float old_val = p_isovalue->getValue();
        if (_max <= old_val)
            old_val = _max;
        if (_min >= old_val)
            old_val = _min;
        p_isovalue->setValue(_min, _max, old_val);
    }
}

void IsoSurface::preHandleObjects(coInputPort **InPorts)
{
    ww_.reset();
    // Automatically adapt our Module's title to the species
    if (autoTitle)
    {
        const coDistributedObject *obj = p_IsoDataIn->getCurrentObject();
        if (obj)
        {
            const char *species = obj->getAttribute("SPECIES");

            int len = 0;
            if (species)
                len += (int)strlen(species) + 3;
            char *buf = new char[len + 64];
            if (species)
                sprintf(buf, "Iso-%s:%s", get_instance(), species);
            else
                sprintf(buf, "Iso-%s", get_instance());
            setTitle(buf);
            delete[] buf;
        }
    }
    inSelfExec = false; // we just start this execution

    if (p_IsoDataIn->getCurrentObject() == NULL)
    {
        _scalarName = "";
    }
    else if ((_scalarName != p_IsoDataIn->getCurrentObject()->getName()
              && p_autominmax_->getValue())
             || (_scalarName == p_IsoDataIn->getCurrentObject()->getName()
                 && !_autominmax && p_autominmax_->getValue()))
    {
        UpdateIsoValue();
    }
    else if (p_autominmax_->getValue())
    {
        p_isovalue->setMin(_min);
        p_isovalue->setMax(_max);
    }

    _autominmax = p_autominmax_->getValue();

    if (p_pointOrValue->getValue() == POINT)
    {
        fillThePoint();
        setUpIsoList(InPorts);
    }
    return;
}

void
IsoSurface::copyAttributesToOutObj(coInputPort **input_ports,
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
        return;
    }
    switch (port - shiftOut)
    {
    case 0:
        // FIXME
        if (input_ports[0] && output_ports[port] && output_ports[port]->getCurrentObject() && input_ports[0]->getCurrentObject())
        {
            copyAttributes(output_ports[port]->getCurrentObject(),
                           input_ports[0]->getCurrentObject());
        }
        break;
    case 1:
    {
        int inDataPort = 1;
        if (p_DataIn->getCurrentObject())
        {
            inDataPort = 2;
        }
        if (input_ports[inDataPort] && output_ports[port] && output_ports[port]->getCurrentObject() && input_ports[inDataPort]->getCurrentObject())
        {
            copyAttributes(output_ports[port]->getCurrentObject(),
                           input_ports[inDataPort]->getCurrentObject());
        }
    }
    break;
    default:
        break;
    }
    return;
}

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
            Covise::sendInfo("complete run: %6.3f s\n", ww_.elapsed());
    }
    void silent()
    {
        silent_ = true;
    }
};

void IsoSurface::postHandleObjects(coOutputPort **OutPorts)
{

    lookUp = 0;

    Terminator terminator;

    //sl:  FEEDBACK for the object of the first port
    if (OutPorts == NULL || OutPorts[shiftOut]->getCurrentObject() == NULL)
    {
        if (!inSelfExec)
        {
            terminator.silent();
            sendWarning("The output could not be correctly created");
        }
        return;
    }

#ifndef _COMPLEX_MODULE_ // in the complex case, only for scalar data
    addFeedbackParams(OutPorts[shiftOut]->getCurrentObject());
#endif

    objLabVal.clean();
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
    coInputPort *p_mappedData = p_DataIn->isConnected() ? p_DataIn : p_IsoDataIn;
    // SCALAR
    if (coObjectAlgorithms::containsType<const coDoFloat *>(p_mappedData->getCurrentObject()))
    {
        // this is the easiest case...
        coDistributedObject *geo = p_GridOut->getCurrentObject();
        addFeedbackParams(geo);
        if (geo)
        {
            geo->incRefCount();
        }
        coDoGeometry *do_geom = new coDoGeometry(p_GeometryOut->getObjName(), geo);
        coDistributedObject *norm = p_NormalsOut->getCurrentObject();
        if (norm)
        {
            norm->incRefCount();
            do_geom->setNormals(PER_VERTEX, norm);
        }
        if (p_color_or_texture->getValue()) // colors
        {
            string color_name = p_GeometryOut->getObjName();
            color_name += "_Color";
            coDistributedObject *color = ComplexModules::DataTexture(color_name,
                                                                     p_DataOut->getCurrentObject(),
                                                                     p_ColorMapIn->getCurrentObject(), false);
            if (color)
            {
                do_geom->setColors(PER_VERTEX, color);
            }
        }
        else // texture
        {
            string texture_name = p_GeometryOut->getObjName();
            texture_name += "_Texture";
            coDistributedObject *texture = ComplexModules::DataTexture(texture_name,
                                                                       p_DataOut->getCurrentObject(),
                                                                       p_ColorMapIn->getCurrentObject(), true);
            if (texture)
            {
                do_geom->setTexture(0, texture);
            }
        }
        p_GeometryOut->setCurrentObject(do_geom);
    }
    // VECTOR
    else if (coObjectAlgorithms::containsType<const coDoVec3 *>(p_mappedData->getCurrentObject()))
    {
        coDistributedObject *geo = p_GridOut->getCurrentObject();
        coDistributedObject *data = p_DataOut->getCurrentObject();
        if (geo->isType("SETELE") && geo->getAttribute("TIMESTEP"))
        {
            // open the timesteps and call StaticParts for each of them
            if (!data->isType("SETELE"))
            {
                return;
            }
            ScalarContainer ScalarCont;
            ScalarCont.Initialise(data);
            coDoSet *Geo = (coDoSet *)geo;
            coDoSet *Data = (coDoSet *)data;
            int no_e, no_d;
            const coDistributedObject *const *geoList = Geo->getAllElements(&no_e);
            const coDistributedObject *const *dataList = Data->getAllElements(&no_d);
            if (no_e != no_d)
            {
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
            ColorAllSteps->copyAllAttributes(data);
            // add COLORMAP to ColorAllSteps...
            coColors theColors(data, (coDoColormap *)(p_ColorMapIn->getCurrentObject()), false);
            theColors.addColormapAttrib(ColorAllSteps->getName(), ColorAllSteps);
            coDoGeometry *GeoOutput = new coDoGeometry(p_GeometryOut->getObjName(),
                                                       GeoAllSteps);
            GeoOutput->setNormals(PER_VERTEX, NormAllSteps);
            GeoOutput->setColors(PER_VERTEX, ColorAllSteps);
            addFeedbackParams(GeoAllSteps);
            p_GeometryOut->setCurrentObject(GeoOutput);

            for (i = 0; i < no_e; ++i)
            {
                delete GeoFullList[i];
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
            coDoGeometry *do_geom = new coDoGeometry(p_GeometryOut->getObjName(), geopart);
            addFeedbackParams(geopart);
            p_GeometryOut->setCurrentObject(do_geom);
            do_geom->setNormals(PER_VERTEX, normpart);
            do_geom->setColors(PER_VERTEX, colorpart);
        }
    }
    else
    {
        terminator.silent();
        Covise::sendWarning("Could not determine whether the input data field is scalar or vector");
    }
#endif
    return;
}

void IsoSurface::setUpIsoList(coInputPort **inPorts)
{

    const char *dataType;

    // check for error
    if (NULL == inPorts || NULL == inPorts[0]->getCurrentObject()
        || NULL == inPorts[1]->getCurrentObject())
        return;

    // begin with the first object
    dataType = (inPorts[0]->getCurrentObject())->getType();
    if (0 == strcmp(dataType, "SETELE") && (inPorts[0]->getCurrentObject())->getAttribute("TIMESTEP"))
    {
        // This is a set with time steps
        myPair result = find_isovalueT(inPorts[0]->getCurrentObject(), inPorts[1]->getCurrentObject());
        if (result.Tag == TRIUMPH)
        {
            p_isovalue->setValue(result.Value);
        }
    }
    else
    {
        // This is a set without time steps or a "normal" object
        myPair result = find_isovalueSG(inPorts[0]->getCurrentObject(), inPorts[1]->getCurrentObject());
        if (result.Tag == TRIUMPH)
        {
            p_isovalue->setValue(result.Value);
        }
    }
}

myPair IsoSurface::find_isovalueT(const coDistributedObject *inObj,
                                  const coDistributedObject *idata)
{
    return find_isovalueSGoT(inObj, idata);
}

myPair IsoSurface::find_isovalueSG(const coDistributedObject *inObj,
                                   const coDistributedObject *idata)
{
    myPair resultSons;

    level++;
    resultSons = find_isovalueSGoT(inObj, idata);
    if (level == 1)
    {
        if (inObj && idata)
            resultSons = find_isovalueSGoT(inObj, idata);
        // cout << "Added to list: "<< resultSons.Value<< endl;
        objLabVal.addToList(new Elem(inObj, resultSons.Tag, resultSons.Value));
    }
    level--;
    return resultSons;
}

myPair IsoSurface::find_isovalueSGoT(const coDistributedObject *inObj,
                                     const coDistributedObject *idata)
{
    myPair resultSons;
    int numSetElem, t;
    const coDoSet *p_Grid;
    const coDoSet *p_IsoData;
    int setIsT = 0;
    const coDistributedObject *const *gridList;
    const coDistributedObject *const *isoDataList;

    if (strcmp(inObj->getType(), "SETELE") == 0) // a set
    {
        p_Grid = (const coDoSet *)inObj;
        if (strcmp(idata->getType(), "SETELE") != 0)
        {
            sendError("Data element type do not match");
        }
        if (inObj->getAttribute("TIMESTEP"))
            setIsT = 1;
        p_IsoData = (const coDoSet *)idata;
        gridList = p_Grid->getAllElements(&numSetElem);
        isoDataList = p_IsoData->getAllElements(&t);
        if (numSetElem != t)
            sendError("Number of elements in matching sets do not coincide");
        for (int i = 0; i < numSetElem; ++i)
        {
            if (gridList[i] == 0)
                continue;
            // Proceed only for non-set sons
            // or set-sons that are not time steps
            if (strcmp(gridList[i]->getType(), "SETELE") != 0 || !(gridList[i]->getAttribute("TIMESTEP")))
            {
                resultSons = find_isovalueSG(gridList[i], isoDataList[i]);
                if (resultSons.Tag == TRIUMPH && setIsT == 0)
                    break;
            }
            else
            {
                sendError(" do not accept a set of time steps, which is an element of another set");
            }
        }
    } // This is a normal element
    else
    {
        if (strcmp(inObj->getType(), "UNIGRD") == 0 && strcmp(idata->getType(), "USTSDT") == 0)
        {
            resultSons = find_isovalueU((coDoUniformGrid *)inObj,
                                        (coDoFloat *)idata);
        }
        else if (strcmp(inObj->getType(), "RCTGRD") == 0 && strcmp(idata->getType(), "USTSDT") == 0)
        {
            resultSons = find_isovalueR((coDoRectilinearGrid *)inObj,
                                        (coDoFloat *)idata);
        }
        else if (strcmp(inObj->getType(), "STRGRD") == 0 && strcmp(idata->getType(), "USTSDT") == 0)
        {
            resultSons = find_isovalueS((coDoStructuredGrid *)inObj,
                                        (coDoFloat *)idata);
        }
        else if (strcmp(inObj->getType(), "UNSGRD") == 0 && strcmp(idata->getType(), "USTSDT") == 0)
        {
            resultSons = find_isovalueUU((coDoUnstructuredGrid *)inObj,
                                         (coDoFloat *)idata);
        }
        else if (strcmp(inObj->getType(), "UNSGRD") == 0 && strcmp(idata->getType(), "USTSDT") == 0)
        {
            resultSons = find_isovalueUS((coDoUnstructuredGrid *)inObj,
                                         (coDoFloat *)idata);
        }
        else
        {
            sendError(" cannot process this kind of data");
        }
    }
    return resultSons;
}

myPair IsoSurface::find_isovalueU(const coDoUniformGrid *i_mesh,
                                  const coDoFloat *isodata)
{
    myPair result;

    ugrid_in = i_mesh;
    i_data_in = isodata;
    if (isodata->getNumPoints() == 0)
    {
        return result;
    }
    find_uni_value(startp, result.Value);
    if (result.Value == FLT_MAX)
        result.Tag = FAILURE;
    else
        result.Tag = TRIUMPH;
    return result;
}

myPair IsoSurface::find_isovalueR(const coDoRectilinearGrid *i_mesh,
                                  const coDoFloat *isodata)
{
    myPair result;

    rgrid_in = i_mesh;
    i_data_in = isodata;
    if (isodata->getNumPoints() == 0)
    {
        return result;
    }
    find_rct_value(startp, result.Value);
    if (result.Value == FLT_MAX)
        result.Tag = FAILURE;
    else
        result.Tag = TRIUMPH;
    return result;
}

myPair IsoSurface::find_isovalueS(const coDoStructuredGrid *i_mesh,
                                  const coDoFloat *isodata)
{
    myPair result;

    sgrid_in = i_mesh;
    i_data_in = isodata;
    if (isodata->getNumPoints() == 0)
    {
        return result;
    }
    find_str_value(startp, result.Value);
    if (result.Value == FLT_MAX)
        result.Tag = FAILURE;
    else
        result.Tag = TRIUMPH;
    return result;
}

myPair IsoSurface::find_isovalueUU(const coDoUnstructuredGrid *i_mesh,
                                   const coDoFloat *isodata)
{
    int *el, *cl, *tl;
    float *x_in;
    float *y_in;
    float *z_in;
    int cuc_count;
    int *cuc = NULL, *cuc_pos = NULL;
    int numelem, numconn, numcoord;
    int data_anz;
    int cell = -1;
    myPair result;

    grid_in = i_mesh;
    grid_in->getGridSize(&numelem, &numconn, &numcoord);
    num_coord = numcoord;

    i_data_in = isodata;
    data_anz = i_data_in->getNumPoints();
    if (data_anz == 0)
        return result;

    grid_in->getAddresses(&el, &cl, &x_in, &y_in, &z_in);
    grid_in->getTypeList(&tl);
    grid_in->getNeighborList(&cuc_count, &cuc, &cuc_pos);

    if (data_anz != numcoord)
    {
        sendError("ERROR: Dataobject's dimension doesn't match Grid ones");
    }

    i_data_in->getAddress(&i_in);

    cell = find_startcell_fast(startp, x_in, y_in, z_in, numelem, numconn, el, cl, tl,
                               cuc_count, cuc, cuc_pos);
    if (cell >= 0)
    {
        // sl: For the interpolation we make first the lists
        //     needed for an structured grid of one element
        //     and 8 nodes.
        //     Then we may use cell3 and intp3 as above for
        //     structured grids.
        int i_dim = 2;
        int j_dim = 2;
        int k_dim = 2;
        int cellx = 1, celly = 1, cellz = 1;
        int status;
        float x_list[8], y_list[8], z_list[8], isoscalar[8];
        float a, b, g;
        float amat[3][3], bmat[3][3];
        // These arrays encode the node transformation
        // from unstructured to structured
        int hexa_nodes[8] = { 5, 1, 6, 2, 4, 0, 7, 3 };
        int pyra_nodes[8] = { 4, 1, 4, 2, 4, 0, 4, 3 };
        int pris_nodes[8] = { 4, 1, 5, 2, 3, 0, 5, 2 };
        int tetr_nodes[8] = { 3, 1, 3, 2, 3, 0, 3, 2 };
        switch (tl[cell])
        {
        // sl: These cases would produce results according
        //     to the same method of multilinear interpolation,
        //     used for structured grids
        //     instead of the interpolation with
        //     the weights calculated from the distances to the nodes
        //     (see default case below). Hopefully this latter method
        //     is good enough. If not, you may suppress the
        //     commentaries for the "cases" below and activate them.
        //     They have
        //     been tested, and they work... but the tests
        //     were not exhaustive enough to risk a possible bug.

        case TYPE_HEXAGON:
            fill_lists(isoscalar, x_list, y_list, z_list,
                       x_in, y_in, z_in,
                       hexa_nodes,
                       el[cell], i_in, cl);
            grid_methods::cell3(i_dim, j_dim, k_dim, x_list, y_list, z_list,
                                &cellx, &celly, &cellz, &a, &b, &g,
                                startp, amat, bmat, &status);
            // cout << "Status " << status << endl;
            if (status == 0)
                grid_methods::intp3(i_dim, j_dim, k_dim, isoscalar, isoscalar, isoscalar,
                                    cellx, celly, cellz, a, b, g, &isovalue);
            break;
        case TYPE_PYRAMID:
            fill_lists(isoscalar, x_list, y_list, z_list,
                       x_in, y_in, z_in,
                       pyra_nodes,
                       el[cell], i_in, cl);
            grid_methods::cell3(i_dim, j_dim, k_dim, x_list, y_list, z_list,
                                &cellx, &celly, &cellz, &a, &b, &g,
                                startp, amat, bmat, &status);
            if (status == 0)
                grid_methods::intp3(i_dim, j_dim, k_dim, isoscalar, isoscalar, isoscalar,
                                    cellx, celly, cellz, a, b, g, &isovalue);
            break;
        case TYPE_PRISM:
            fill_lists(isoscalar, x_list, y_list, z_list,
                       x_in, y_in, z_in,
                       pris_nodes,
                       el[cell], i_in, cl);
            grid_methods::cell3(i_dim, j_dim, k_dim, x_list, y_list, z_list,
                                &cellx, &celly, &cellz, &a, &b, &g,
                                startp, amat, bmat, &status);
            if (status == 0)
                grid_methods::intp3(i_dim, j_dim, k_dim, isoscalar, isoscalar, isoscalar,
                                    cellx, celly, cellz, a, b, g, &isovalue);
            break;
        case TYPE_TETRAHEDER:
            fill_lists(isoscalar, x_list, y_list, z_list,
                       x_in, y_in, z_in,
                       tetr_nodes,
                       el[cell], i_in, cl);
            grid_methods::cell3(i_dim, j_dim, k_dim, x_list, y_list, z_list,
                                &cellx, &celly, &cellz, &a, &b, &g,
                                startp, amat, bmat, &status);
            if (status == 0)
                grid_methods::intp3(i_dim, j_dim, k_dim, isoscalar, isoscalar, isoscalar,
                                    cellx, celly, cellz, a, b, g, &isovalue);
            break;
        case TYPE_POLYHEDRON:
        {
            bool start_vertex_set;

            int i;
            //int current_cell;
            int next_elem_index;
            int start_vertex;

            // int *vertex_list;

            vector<int> temp_elem_in;
            vector<int> temp_conn_in;
            vector<int> temp_vertex_list;

            /***********************************************/
            /* Calculation of the number of nodes in the cell */
            /***********************************************/

            start_vertex_set = false;
            //current_cell = cell;
            next_elem_index = (cell < numelem) ? el[cell + 1] : numconn;

            /* Construct DO_Polygons Element and Connectivity Lists */
            for (i = el[cell]; i < next_elem_index; i++)
            {
                if (i == el[cell] && start_vertex_set == false)
                {
                    start_vertex = cl[el[cell]];
                    temp_elem_in.push_back((int)temp_conn_in.size());
                    temp_conn_in.push_back(start_vertex);
                    start_vertex_set = true;
                }

                if (i > el[cell] && start_vertex_set == true)
                {
                    if (cl[i] != start_vertex)
                    {
                        temp_conn_in.push_back(cl[i]);
                    }

                    else
                    {
                        start_vertex_set = false;
                        continue;
                    }
                }

                if (i > el[cell] && start_vertex_set == false)
                {
                    start_vertex = cl[i];
                    temp_elem_in.push_back((int)temp_conn_in.size());
                    temp_conn_in.push_back(start_vertex);
                    start_vertex_set = true;
                }
            }

            /* Construct Vertex List */
            for (i = 0; i < temp_conn_in.size(); i++)
            {
                if (temp_vertex_list.size() == 0)
                {
                    temp_vertex_list.push_back(temp_conn_in[i]);
                }

                else
                {
                    if (find(temp_vertex_list.begin(), temp_vertex_list.end(), temp_conn_in[i]) == temp_vertex_list.end())
                    {
                        temp_vertex_list.push_back(temp_conn_in[i]);
                    }
                }
            }

            sort(temp_vertex_list.begin(), temp_vertex_list.end());

            /*********************************************************/
            /* Isovalue Interpolation                                                     */
            /*                                                                                       */
            /* The implementation of the Shepard Method used in the */
            /* default case is also implemented for polyhedral cells      */
            /* although other interpolation methods could be applied  */
            /* as well.  The default case now becomes obsolete.          */
            /*********************************************************/

            double weight;
            double weigth_sum = 0;
            isovalue = 0.0;

            for (i = 0; i < temp_vertex_list.size(); i++)
            {
                weight = sqr(startp[0] - x_in[temp_vertex_list[i]]) + sqr(startp[1] - y_in[temp_vertex_list[i]]) + sqr(startp[2] - z_in[temp_vertex_list[i]]);
                weight = (weight == 0.0) ? 1e20 : (1.0 / weight);

                weigth_sum += weight;
                isovalue += (float)(weight * i_in[temp_vertex_list[i]]);
            }
            status = 0;
            isovalue = (float)(isovalue / weigth_sum);
        }
        break;
        default:
        {
            double weigth_sum = 0;
            int i;
            isovalue = 0.0;
            for (i = 0; i < UnstructuredGrid_Num_Nodes[tl[cell]]; i++)
            {
                // sl: the weighting with a sqrt will produce
                //     results which are more similar to those
                //     obtained by multilinear interpolation
                double weight = sqr(startp[0] - x_in[cl[el[cell] + i]]) + sqr(startp[1] - y_in[cl[el[cell] + i]]) + sqr(startp[2] - z_in[cl[el[cell] + i]]);
                weight = (weight == 0.0) ? 1e20 : (1.0 / weight);

                weigth_sum += weight;
                isovalue += (float)weight * i_in[cl[el[cell] + i]];
                //fprintf(stderr,"%f\n%f\n",i_in[cl[el[cell]+i]],weight);
            }
            status = 0;
            isovalue /= (float)weigth_sum;
        }
        break;
        }
        if (status == 0)
        {
            result.Tag = TRIUMPH;
            result.Value = isovalue;
        }
    }
    grid_in->freeNeighborList();
    return result;
}

myPair IsoSurface::find_isovalueUS(const coDoUnstructuredGrid *i_mesh,
                                   const coDoFloat *isodata)
{
    int *el, *cl, *tl;
    float *x_in;
    float *y_in;
    float *z_in;
    int cuc_count;
    int *cuc = NULL, *cuc_pos = NULL;
    int numelem, numconn, numcoord;
    int data_anz;
    int cell = -1;
    myPair result;

    grid_in = i_mesh;
    grid_in->getGridSize(&numelem, &numconn, &numcoord);
    num_coord = numcoord;

    i_data_in = isodata;
    data_anz = i_data_in->getNumPoints();
    if (data_anz == 0)
        return result;
    grid_in->getAddresses(&el, &cl, &x_in, &y_in, &z_in);
    grid_in->getTypeList(&tl);
    grid_in->getNeighborList(&cuc_count, &cuc, &cuc_pos);
    i_data_in->getAddress(&i_in);
    if (data_anz != numcoord)
    {
        sendError("ERROR: Dataobject's dimension doesn't match Grid ones");
    }

    cell = find_startcell_fast(startp, x_in, y_in, z_in, numelem, numconn, el, cl, tl,
                               cuc_count, cuc, cuc_pos);

    if (cell < 0)
    {
        result.Tag = FAILURE;
        result.Value = FLT_MAX;
    }
    else
    {
        // sl: For the interpolation we make first the lists
        //     needed for an structured grid of one element
        //     and 8 nodes.
        //     Then we may use cell3 and intp3 as above for
        //     structured grids.
        int i_dim = 2;
        int j_dim = 2;
        int k_dim = 2;
        int cellx = 1, celly = 1, cellz = 1;
        int status;
        float x_list[8], y_list[8], z_list[8], isoscalar[8];
        float a, b, g;
        float amat[3][3], bmat[3][3];
        // These arrays encode the node transformation
        // from unstructured to structured
        int hexa_nodes[8] = { 5, 1, 6, 2, 4, 0, 7, 3 };
        int pyra_nodes[8] = { 4, 1, 4, 2, 4, 0, 4, 3 };
        int pris_nodes[8] = { 4, 1, 5, 2, 3, 0, 5, 2 };
        int tetr_nodes[8] = { 3, 1, 3, 2, 3, 0, 3, 2 };
        switch (tl[cell])
        {
        // sl: These cases would produce results according
        //     to the same method of multilinear interpolation,
        //     used for structured grids
        //     instead of the interpolation with
        //     the weights calculated from the distances to the nodes
        //     (see default case below). Hopefully this latter method
        //     is good enough. If not, you may suppress the
        //     commentaries for the "cases" below and activate them.
        //     They have
        //     been tested, and they work... but the tests
        //     were not exhaustive enough to risk a possible bug.

        case TYPE_HEXAGON:
            fill_lists(isoscalar, x_list, y_list, z_list,
                       x_in, y_in, z_in,
                       hexa_nodes,
                       el[cell], i_in, cl);
            grid_methods::cell3(i_dim, j_dim, k_dim, x_list, y_list, z_list,
                                &cellx, &celly, &cellz, &a, &b, &g,
                                startp, amat, bmat, &status);
            grid_methods::intp3(i_dim, j_dim, k_dim, isoscalar, isoscalar, isoscalar,
                                cellx, celly, cellz, a, b, g, &isovalue);
            break;
        case TYPE_PYRAMID:
            fill_lists(isoscalar, x_list, y_list, z_list,
                       x_in, y_in, z_in,
                       pyra_nodes,
                       el[cell], i_in, cl);
            grid_methods::cell3(i_dim, j_dim, k_dim, x_list, y_list, z_list,
                                &cellx, &celly, &cellz, &a, &b, &g,
                                startp, amat, bmat, &status);
            grid_methods::intp3(i_dim, j_dim, k_dim, isoscalar, isoscalar, isoscalar,
                                cellx, celly, cellz, a, b, g, &isovalue);
            break;
        case TYPE_PRISM:
            fill_lists(isoscalar, x_list, y_list, z_list,
                       x_in, y_in, z_in,
                       pris_nodes,
                       el[cell], i_in, cl);
            grid_methods::cell3(i_dim, j_dim, k_dim, x_list, y_list, z_list,
                                &cellx, &celly, &cellz, &a, &b, &g,
                                startp, amat, bmat, &status);
            grid_methods::intp3(i_dim, j_dim, k_dim, isoscalar, isoscalar, isoscalar,
                                cellx, celly, cellz, a, b, g, &isovalue);
            break;
        case TYPE_TETRAHEDER:
            fill_lists(isoscalar, x_list, y_list, z_list,
                       x_in, y_in, z_in,
                       tetr_nodes,
                       el[cell], i_in, cl);
            grid_methods::cell3(i_dim, j_dim, k_dim, x_list, y_list, z_list,
                                &cellx, &celly, &cellz, &a, &b, &g,
                                startp, amat, bmat, &status);
            grid_methods::intp3(i_dim, j_dim, k_dim, isoscalar, isoscalar, isoscalar,
                                cellx, celly, cellz, a, b, g, &isovalue);
            break;
        case TYPE_POLYHEDRON:
            sendError("ERROR:  Polyhedral meshes require unstructured data objects");
            result.Tag = FAILURE;
            result.Value = FLT_MAX;
            break;
        default:
        {
            double weigth_sum = 0;
            int i;
            isovalue = 0.0;
            for (i = 0; i < UnstructuredGrid_Num_Nodes[tl[cell]]; i++)
            {
                // sl: the weighting with a sqrt will produce
                //     results which are more similar to those
                //     obtained by multilinear interpolation
                double weight = sqr(startp[0] - x_in[cl[el[cell] + i]]) + sqr(startp[1] - y_in[cl[el[cell] + i]]) + sqr(startp[2] - z_in[cl[el[cell] + i]]);
                weight = (weight == 0.0) ? 1e20 : (1.0 / weight);

                weigth_sum += weight;
                isovalue += (float)weight * i_in[cl[el[cell] + i]];
                //fprintf(stderr,"%f\n%f\n",i_in[cl[el[cell]+i]],weight);
            }

            isovalue /= (float)weigth_sum;
        }
        break;
        }
        if (tl[cell] != TYPE_POLYHEDRON)
        {
            result.Tag = TRIUMPH;
            result.Value = isovalue;
        }
    }
    grid_in->freeNeighborList();
    return result;
}

// sl: Function to find isovalue for uni- grids given a point...
void find_uni_value(float point[3], float &isowert)
{
    int x_size, y_size, z_size;
    ugrid_in->getGridSize(&x_size, &y_size, &z_size);
    //   if(x_size<2 || y_size<2 || z_size<2) return;
    if (x_size <= 0 || y_size <= 0 || z_size <= 0)
    {
        isowert = FLT_MAX;
        return;
    }

    float x_min, y_min, z_min;
    float x_max, y_max, z_max;
    ugrid_in->getMinMax(&x_min, &x_max,
                        &y_min, &y_max,
                        &z_min, &z_max);

    int cellx = 0, celly = 0, cellz = 0;
    float dx, dy, dz;
    ugrid_in->getDelta(&dx, &dy, &dz);

    if (x_min <= x_max)
    {
        if (point[0] < x_min || point[0] > x_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }
    else
    {
        if (point[0] > x_min || point[0] < x_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }

    if (y_min <= y_max)
    {
        if (point[1] < y_min || point[1] > y_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }
    else
    {
        if (point[1] > y_min || point[1] < y_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }

    if (z_min <= z_max)
    {
        if (point[2] < z_min || point[2] > z_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }
    else
    {
        if (point[2] > z_min || point[2] < z_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }

    if (x_size == 1 || dx == 0.0)
    {
        cellx = 0;
    }
    else
    {
#ifdef __sgi
        cellx = int(ffloor(fabsf((point[0] - x_min) / dx)));
#else
        cellx = int(floor(fabs((point[0] - x_min) / dx)));
#endif
    }

    if (y_size == 1 || dy == 0.0)
    {
        celly = 0;
    }
    else
    {
#ifdef __sgi
        celly = int(ffloor(fabsf((point[1] - y_min) / dy)));
#else
        celly = int(floor(fabs((point[1] - y_min) / dy)));
#endif
    }

    if (z_size == 1 || dz == 0.0)
    {
        cellz = 0;
    }
    else
    {
#ifdef __sgi
        cellz = int(ffloor(fabsf((point[2] - z_min) / dz)));
#else
        cellz = int(floor(fabs((point[2] - z_min) / dz)));
#endif
    }

    // Now we have to read in the 8 scalar values at the corners of the cell
    // Read from i_data_in
    float valmmm;
    float valmmp;
    float valmpm;
    float valmpp;
    float valpmm;
    float valpmp;
    float valppm;
    float valppp;
    int to_next_x = 1;
    int to_next_y = 1;
    int to_next_z = 1;
    if (x_size == 1)
        to_next_x = 0;
    if (y_size == 1)
        to_next_y = 0;
    if (z_size == 1)
        to_next_z = 0;
    int dims[3] = { x_size, y_size, z_size };
    i_data_in->getPointValue(coIndex(cellx, celly, cellz, dims), &valmmm);
    i_data_in->getPointValue(coIndex(cellx, celly, cellz + to_next_z, dims), &valmmp);
    i_data_in->getPointValue(coIndex(cellx, celly + to_next_y, cellz, dims), &valmpm);
    i_data_in->getPointValue(coIndex(cellx, celly + to_next_y, cellz + to_next_z, dims), &valmpp);
    i_data_in->getPointValue(coIndex(cellx + to_next_x, celly, cellz, dims), &valpmm);
    i_data_in->getPointValue(coIndex(cellx + to_next_x, celly, cellz + to_next_z, dims), &valpmp);
    i_data_in->getPointValue(coIndex(cellx + to_next_x, celly + to_next_y, cellz, dims), &valppm);
    i_data_in->getPointValue(coIndex(cellx + to_next_x, celly + to_next_y, cellz + to_next_z, dims), &valppp);

    // Now we calculate the values of the form functions at our point

    float x, y, z; // these are non-dimensional coordinates in the cell!!!
    // from -1. to +1.
    if (dx == 0.0)
        x = 0.0;
    else
        x = 2.0f * (point[0] - x_min - (cellx + 0.5f) * dx) / dx;

    if (dy == 0.0)
        y = 0.0;
    else
        y = 2.0f * (point[1] - y_min - (celly + 0.5f) * dy) / dy;

    if (dz == 0.0)
        z = 0.0;
    else
        z = 2.0f * (point[2] - z_min - (cellz + 0.5f) * dz) / dz;

    float formmmm;
    float formmmp;
    float formmpm;
    float formmpp;
    float formpmm;
    float formpmp;
    float formppm;
    float formppp;
    formmmm = (1.0f - x) * (1.0f - y) * (1.0f - z) / 8.0f;
    formmmp = (1.0f - x) * (1.0f - y) * (1.0f + z) / 8.0f;
    formmpm = (1.0f - x) * (1.0f + y) * (1.0f - z) / 8.0f;
    formmpp = (1.0f - x) * (1.0f + y) * (1.0f + z) / 8.0f;
    formpmm = (1.0f + x) * (1.0f - y) * (1.0f - z) / 8.0f;
    formpmp = (1.0f + x) * (1.0f - y) * (1.0f + z) / 8.0f;
    formppm = (1.0f + x) * (1.0f + y) * (1.0f - z) / 8.0f;
    formppp = (1.0f + x) * (1.0f + y) * (1.0f + z) / 8.0f;

    // Interpolate
    isowert = valmmm * formmmm;
    isowert += valmmp * formmmp;
    isowert += valmpm * formmpm;
    isowert += valmpp * formmpp;
    isowert += valpmm * formpmm;
    isowert += valpmp * formpmp;
    isowert += valppm * formppm;
    isowert += valppp * formppp;
}

// sl: Function to find isovalue for rct grids given a point...
void find_rct_value(float point[3], float &isowert)
{
    int x_size, y_size, z_size;
    rgrid_in->getGridSize(&x_size, &y_size, &z_size);
    //   if(x_size<2 || y_size<2 || z_size<2) return;
    if (x_size <= 0 || y_size <= 0 || z_size <= 0)
    {
        isowert = FLT_MAX;
        return;
    }

    float x_min, y_min, z_min;
    float x_max, y_max, z_max;
    rgrid_in->getPointCoordinates(0, &x_min, 0, &y_min, 0, &z_min);
    rgrid_in->getPointCoordinates(x_size - 1, &x_max, y_size - 1, &y_max, z_size - 1, &z_max);

    if (x_min <= x_max)
    {
        if (point[0] < x_min || point[0] > x_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }
    else
    {
        if (point[0] > x_min || point[0] < x_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }

    if (y_min <= y_max)
    {
        if (point[1] < y_min || point[1] > y_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }
    else
    {
        if (point[1] > y_min || point[1] < y_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }

    if (z_min <= z_max)
    {
        if (point[2] < z_min || point[2] > z_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }
    else
    {
        if (point[2] > z_min || point[2] < z_max)
        {
            isowert = FLT_MAX;
            return;
        }
    }

    int cellx = 0, celly = 0, cellz = 0;
    float *x_list;
    float *y_list;
    float *z_list;
    rgrid_in->getAddresses(&x_list, &y_list, &z_list);

    float dies = x_list[0];
    float next;
    int i;
    if (x_size == 1)
    {
        cellx = 0;
    }
    else
    {
        for (i = 0; i < x_size - 1; ++i)
        {
            next = x_list[i + 1];
            if (dies <= point[0] && point[0] < next)
                break;
            if (dies >= point[0] && point[0] > next)
                break;
            dies = next;
        }
        if (i == x_size - 1)
#ifdef __sgi
            cellx = (fabsf(x_min - point[0]) <= fabsf(x_max - point[0]) ? 0 : x_size - 2);
#else
            cellx = (fabs(x_min - point[0]) <= fabs(x_max - point[0]) ? 0 : x_size - 2);
#endif
        else
            cellx = i;
    }

    dies = y_list[0];
    int j;
    if (y_size == 1)
    {
        celly = 0;
    }
    else
    {
        for (j = 0; j < y_size - 1; ++j)
        {
            next = y_list[j + 1];
            if (dies <= point[1] && point[1] < next)
                break;
            if (dies >= point[1] && point[1] > next)
                break;
            dies = next;
        }
        if (j == y_size - 1)
#ifdef __sgi
            celly = (fabsf(y_min - point[1]) <= fabsf(y_max - point[1]) ? 0 : y_size - 2);
#else
            celly = (fabs(y_min - point[1]) <= fabs(y_max - point[1]) ? 0 : y_size - 2);
#endif
        else
            celly = j;
    }

    dies = z_list[0];
    int k;
    if (z_size == 1)
    {
        cellz = 0;
    }
    else
    {
        for (k = 0; k < z_size - 1; ++k)
        {
            next = z_list[k + 1];
            if (dies <= point[2] && point[2] < next)
                break;
            if (dies >= point[2] && point[2] > next)
                break;
            dies = next;
        }
        if (k == z_size - 1)
#ifdef __sgi
            cellz = (fabsf(z_min - point[2]) <= fabsf(z_max - point[2]) ? 0 : z_size - 2);
#else
            cellz = (fabs(z_min - point[2]) <= fabs(z_max - point[2]) ? 0 : z_size - 2);
#endif
        else
            cellz = k;
    }

    // Now we have to read in the 8 scalar values at the corners of the cell
    // from i_data_in
    float valmmm;
    float valmmp;
    float valmpm;
    float valmpp;
    float valpmm;
    float valpmp;
    float valppm;
    float valppp;
    int to_next_x = 1;
    int to_next_y = 1;
    int to_next_z = 1;
    if (x_size == 1)
        to_next_x = 0;
    if (y_size == 1)
        to_next_y = 0;
    if (z_size == 1)
        to_next_z = 0;
    int dims[3] = { x_size, y_size, z_size };
    i_data_in->getPointValue(coIndex(cellx, celly, cellz, dims), &valmmm);
    i_data_in->getPointValue(coIndex(cellx, celly, cellz + to_next_z, dims), &valmmp);
    i_data_in->getPointValue(coIndex(cellx, celly + to_next_y, cellz, dims), &valmpm);
    i_data_in->getPointValue(coIndex(cellx, celly + to_next_y, cellz + to_next_z, dims), &valmpp);
    i_data_in->getPointValue(coIndex(cellx + to_next_x, celly, cellz, dims), &valpmm);
    i_data_in->getPointValue(coIndex(cellx + to_next_x, celly, cellz + to_next_z, dims), &valpmp);
    i_data_in->getPointValue(coIndex(cellx + to_next_x, celly + to_next_y, cellz, dims), &valppm);
    i_data_in->getPointValue(coIndex(cellx + to_next_x, celly + to_next_y, cellz + to_next_z, dims), &valppp);

    // Now we calculate the values of the form functions at our point

    float x, y, z; // these are non-dimensional coordinates in the cell!!!
    // from -1. to +1.
    if (x_list[cellx] != x_list[cellx + to_next_x])
        x = 2.0f * (point[0] - 0.5f * (x_list[cellx] + x_list[cellx + to_next_x])) / (-x_list[cellx] + x_list[cellx + to_next_x]);
    else
        x = 0.0;
    if (y_list[celly] != y_list[celly + to_next_y])
        y = 2.0f * (point[1] - 0.5f * (y_list[celly] + y_list[celly + to_next_y])) / (-y_list[celly] + y_list[celly + to_next_y]);
    else
        y = 0.0;
    if (z_list[cellz] != z_list[cellz + to_next_z])
        z = 2.0f * (point[2] - 0.5f * (z_list[cellz] + z_list[cellz + to_next_z])) / (-z_list[cellz] + z_list[cellz + to_next_z]);
    else
        z = 0.0;

    float formmmm;
    float formmmp;
    float formmpm;
    float formmpp;
    float formpmm;
    float formpmp;
    float formppm;
    float formppp;
    formmmm = (1.0f - x) * (1.0f - y) * (1.0f - z) / 8.0f;
    formmmp = (1.0f - x) * (1.0f - y) * (1.0f + z) / 8.0f;
    formmpm = (1.0f - x) * (1.0f + y) * (1.0f - z) / 8.0f;
    formmpp = (1.0f - x) * (1.0f + y) * (1.0f + z) / 8.0f;
    formpmm = (1.0f + x) * (1.0f - y) * (1.0f - z) / 8.0f;
    formpmp = (1.0f + x) * (1.0f - y) * (1.0f + z) / 8.0f;
    formppm = (1.0f + x) * (1.0f + y) * (1.0f - z) / 8.0f;
    formppp = (1.0f + x) * (1.0f + y) * (1.0f + z) / 8.0f;

    // Interpolate
    isowert = valmmm * formmmm;
    isowert += valmmp * formmmp;
    isowert += valmpm * formmpm;
    isowert += valmpp * formmpp;
    isowert += valpmm * formpmm;
    isowert += valpmp * formpmp;
    isowert += valppm * formppm;
    isowert += valppp * formppp;
}

void find_str_value(float point[3], float &isowert)
{
    int idim, jdim, kdim;
    sgrid_in->getGridSize(&idim, &jdim, &kdim);
    float *x_list, *y_list, *z_list;
    sgrid_in->getAddresses(&x_list, &y_list, &z_list);
    float a = 0.5, b = 0.5, g = 0.5;
    float amat[3][3], bmat[3][3];
    int status;
    int cellx = 1, celly = 1, cellz = 1;
    // sl: I am not sure that these calls to metr3 are necessary,
    //     but in case they are, here they are.
    /*
      metr3(idim,jdim,kdim,x_list,y_list,z_list,cellx,celly,cellz,
                    a,b,g,amat,bmat,&idegen,&status);
   */
    grid_methods::cell3(idim, jdim, kdim, x_list, y_list, z_list, &cellx, &celly, &cellz, &a, &b, &g,
                        point, amat, bmat, &status);
    if (status == 1)
    {
        isowert = FLT_MAX;
        return;
    }
    float *isoscalar;
    float isointerp[3];
    i_data_in->getAddress(&isoscalar);
    // sl: intp3 is for vector interpolation, not for scalar; but
    //     it is handy now...
    grid_methods::intp3(idim, jdim, kdim, isoscalar, isoscalar, isoscalar,
                        cellx, celly, cellz, a, b, g, isointerp);
    isowert = *isointerp;
}

void
IsoSurface::param(const char *paramName, bool inMapLoading)
{
    if (inMapLoading)
        return;

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

    string pName(paramName);

    if (pName == p_pointOrValue->getName())
    {

        if (p_pointOrValue->getValue() == POINT)
        {
            p_isovalue->disable();
            p_isopoint->enable();
        }
        else if (p_pointOrValue->getValue() == VALUE)
        {
            p_isovalue->enable();
            p_isopoint->disable();
        }
    }

    else if (pName == p_autominmax_->getName() && !inMapLoading)
    {
        if (p_autominmax_->getValue())
        {
            _scalarName = "";
            selfExec();
        }
        //_autominmax = p_autominmax_->getValue();
    }
    else if (pName == p_isopoint->getName())
    {
        p_isovalue->disable();
        p_isopoint->enable();
        p_pointOrValue->setValue(POINT);
    }
    else if (pName == p_isovalue->getName())
    {
        p_isovalue->enable();
        p_isopoint->disable();
        p_pointOrValue->setValue(VALUE);
    }
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
int IsoSurface::compute(const char *)
{
    char colorn[20];

    int *el = NULL, *cl = NULL, *tl = NULL;
    float *x_in;
    float *y_in;
    float *z_in;
    int cuc_count;
    int *cuc = NULL, *cuc_pos = NULL;
    int x_size, y_size, z_size;
    float x_min = FLT_MAX, x_max = -FLT_MAX;
    float y_min = FLT_MAX, y_max = -FLT_MAX;
    float z_min = FLT_MAX, z_max = -FLT_MAX;
    const coDistributedObject *data_obj;
    // sl: coDistributedObject	**grid_objs,**data_objs,**idata_objs;
    int numelem = 0, numconn = 0, numcoord = 0, data_anz = 0, i_data_anz = 0, DataType = 0;
    ;
    int d_num_elem = 0; // NLR cell= -1;
    IsoPlane *plane;
    STR_IsoPlane *splane;
    UNI_IsoPlane *uplane;
    RECT_IsoPlane *rplane;
    POLYHEDRON_IsoPlane *pplane;

    //const char *DataOut    =  p_DataOut->getObjName();
    const char *NormalsOut = p_NormalsOut->getObjName();
    const char *GridOut = p_GridOut->getObjName();

    //	get parameter values
    int gennormals = p_gennormals->getValue();
    int genstrips = p_genstrips->getValue();
    set_num_elem = 0;
    s_data_in = NULL;
    v_data_in = NULL;
    i_data_in = NULL;
    grid_in = NULL;

    x_in = NULL;
    y_in = NULL;
    z_in = NULL;
    char *iblank = NULL;

    isovalue = getValue(lookUp);

    const coDistributedObject *iblank_obj;
    iblank_obj = p_IBlankIn->getCurrentObject();

    if (iblank_obj != NULL && iblank_obj->objectOk())
    {
        const coDoText *iblank_text = dynamic_cast<const coDoText *>(iblank_obj);
        if (iblank_text)
            iblank_text->getAddress(&iblank);
    }

    // GRID
    data_obj = p_GridIn->getCurrentObject();
    if (data_obj != 0L && data_obj->objectOk())
    {
        gtype = data_obj->getType();
        if (strcmp(gtype, "UNSGRD") == 0)
        {
            grid_in = (coDoUnstructuredGrid *)data_obj;
            grid_in->getGridSize(&numelem, &numconn, &numcoord);
            num_coord = numcoord;
            grid_in->getAddresses(&el, &cl, &x_in, &y_in, &z_in);
            grid_in->getTypeList(&tl);
            grid_in->getNeighborList(&cuc_count, &cuc, &cuc_pos);
            // sl NLR cell=find_startcell_fast(startp);
            if (grid_in->getAttribute("COLOR") == NULL)
            {
                // colorn=new char[20];
                strcpy(colorn, "green");
            }
            else
                colorn[0] = '\0';
        }
        else if (strcmp(gtype, "UNIGRD") == 0)
        {
            ugrid_in = (coDoUniformGrid *)data_obj;
            ugrid_in->getGridSize(&x_size, &y_size, &z_size);
            ugrid_in->getMinMax(&x_min, &x_max, &y_min, &y_max, &z_min, &z_max);
            numcoord = x_size * y_size * z_size;
            numelem = ((x_size - 1) * (y_size - 1) * (z_size - 1));
            if (ugrid_in->getAttribute("COLOR") == NULL)
            {
                // colorn=new char[20];
                strcpy(colorn, "white");
            }
            else
                colorn[0] = '\0';
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            rgrid_in = (coDoRectilinearGrid *)data_obj;
            rgrid_in->getGridSize(&x_size, &y_size, &z_size);
            rgrid_in->getAddresses(&x_in, &y_in, &z_in);
            numcoord = x_size * y_size * z_size;
            numelem = ((x_size - 1) * (y_size - 1) * (z_size - 1));
            if (rgrid_in->getAttribute("COLOR") == NULL)
            {
                // colorn=new char[20];
                strcpy(colorn, "white");
            }
            else
                colorn[0] = '\0';
        }
        else if (strcmp(gtype, "STRGRD") == 0)
        {
            sgrid_in = (coDoStructuredGrid *)data_obj;
            sgrid_in->getGridSize(&x_size, &y_size, &z_size);
            sgrid_in->getAddresses(&x_in, &y_in, &z_in);
            numcoord = x_size * y_size * z_size;
            numelem = ((x_size - 1) * (y_size - 1) * (z_size - 1));
            if (sgrid_in->getAttribute("COLOR") == NULL)
            {
                // colorn=new char[20];
                strcpy(colorn, "white");
            }
            else
                colorn[0] = '\0';
        }

        else
        {
            sendWarning("Received illegal type '%s' at port '%s'", data_obj->getType(), p_GridIn->getName());
            p_GridOut->setCurrentObject(0);
            p_NormalsOut->setCurrentObject(0);
            p_DataOut->setCurrentObject(0);
            return CONTINUE_PIPELINE;
        }
    }
    else
    {
#ifndef TOLERANT
        sendError("ERROR: Data object 'meshIn' can't be accessed in shared memory");
#endif
        return STOP_PIPELINE;
    }
    if (numelem == 0
        || (numconn == 0 && strcmp(gtype, "UNSGRD") == 0)
        || numcoord == 0)
    {
        sendWarning("WARNING: Data object 'meshIn' is empty");
    }

    // MAPPED DATA
    //	retrieve data object from shared memeory

    coInputPort *p_mappedData = p_DataIn->isConnected() ? p_DataIn : p_IsoDataIn;
    data_obj = p_mappedData->getCurrentObject();
    if (data_obj != 0L && data_obj->objectOk())
    {
        dtype = data_obj->getType();
        if (strcmp(dtype, "USTSDT") == 0)
        {
            s_data_in = (coDoFloat *)data_obj;
            data_anz = s_data_in->getNumPoints();
            s_data_in->getAddress(&s_in);
            DataType = 1;
        }
        else if (strcmp(dtype, "USTVDT") == 0)
        {
            v_data_in = (coDoVec3 *)data_obj;
            data_anz = v_data_in->getNumPoints();
            v_data_in->getAddresses(&u_in, &v_in, &w_in);
            DataType = 0;
        }
        else
        {
            sendError("ERROR: Data object 'dataIn' has wrong data type");
            return STOP_PIPELINE;
        }
    }
    else
    {
#ifndef TOLERANT
        sendError("ERROR: Data object 'dataIn' can't be accessed in shared memory");
#endif
        return STOP_PIPELINE;
    }

    // check dimensions
    if ((set_num_elem == 0) && (data_anz != numcoord) && data_anz != 0)
    {
        sendError("ERROR: Dataobject's dimension doesn't match Grid ones");
        return STOP_PIPELINE;
    }
    if (set_num_elem != d_num_elem)
    {
        sendError("ERROR: number of elements do not match");
        return STOP_PIPELINE;
    }

    // ISO DATA
    //	retrieve data object from shared memeory
    data_obj = p_IsoDataIn->getCurrentObject();
    if (data_obj != 0L && data_obj->objectOk())
    {
        dtype = data_obj->getType();
        if (strcmp(dtype, "USTSDT") == 0)
        {
            i_data_in = (coDoFloat *)data_obj;
            i_data_anz = i_data_in->getNumPoints();
            i_data_in->getAddress(&i_in);
        }
        else
        {
            sendError("ERROR: Data object 'isoDataIn' has wrong data type");
            return STOP_PIPELINE;
        }
    }
    else
    {
#ifndef TOLERANT
        sendError("ERROR: Data object 'isoDataIn' can't be accessed in shared memory");
#endif
        return STOP_PIPELINE;
    }

    // check dimensions
    if ((set_num_elem == 0) && (i_data_anz != numcoord) && i_data_anz != 0)
    {
        sendError("ERROR: IsoDataobject's dimension doesn't match Grid ones");
        return STOP_PIPELINE;
    }
    if (set_num_elem != d_num_elem)
    {
        sendError("ERROR: number of elements do not match");
        return STOP_PIPELINE;
    }

    if (GridOut == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'meshOut'");
        return STOP_PIPELINE;
    }
    if (NormalsOut == NULL)
    {
        Covise::sendError("ERROR: Object name not correct for 'normalsOut'");
        return STOP_PIPELINE;
    }

    //======================================================================
    // create the iso surface
    //======================================================================
    if (set_num_elem == 0)
    {
        if (data_anz == 0 || i_data_anz == 0)
        {
            NullInputData(p_GridOut, p_NormalsOut, p_DataOut, DataType, gennormals, genstrips, colorn);
        }
        else if (strcmp(gtype, "UNSGRD") == 0)
        {
            // Support for polyhedral cells
            if (Polyhedra)
            {
                pplane = new POLYHEDRON_IsoPlane(numelem, numconn, numcoord, DataType, /*vertexRatio,*/
                                                 el, cl, tl,
                                                 x_in, y_in, z_in, s_in, i_in, u_in, v_in, w_in, isovalue,
                                                 (p_DataIn->isConnected() != 0), iblank);
                if (!pplane->createIsoPlane())
                {
                    delete pplane;
                    sendError("The isosurface could not be created");
                    return STOP_PIPELINE;
                }
                pplane->createcoDistributedObjects(p_GridOut, p_DataOut);
                delete pplane;
            }
            else
            {
                plane = new IsoPlane(numelem, numcoord, DataType, vertexRatio,
                                     el, cl, tl,
                                     x_in, y_in, z_in, s_in, i_in, u_in, v_in, w_in, isovalue,
                                     (p_DataIn->isConnected() != 0), iblank);
                if (!plane->createIsoPlane())
                {
                    delete plane;

                    if (plane->polyhedral_cells_found)
                    {
                        sendInfo("The dataset contains apparently only polyhedral cells");
                        sendInfo("Please verify that the polyhedral cell support option has been selected");
                    }
                    else
                    {
                        // increase VERTEX_RATIO and start over
                        vertexRatio += 5.;
                        sendInfo("Increased VERTEX_RATIO to %.0f%%", vertexRatio);
                        setExecGracePeriod(0.2f); // wait only 0.2 sec for network latency
                        inSelfExec = true;
                        selfExec();
                    }
                    return STOP_PIPELINE;
                }
                plane->createcoDistributedObjects(p_GridOut, p_NormalsOut, p_DataOut, gennormals, genstrips, colorn);
                delete plane;
            }
        }
        else if (strcmp(gtype, "UNIGRD") == 0)
        {
            uplane = new UNI_IsoPlane(numelem, numcoord, DataType,
                                      x_min, x_max, y_min, y_max, z_min, z_max,
                                      x_size, y_size, z_size,
                                      s_in, i_in, u_in, v_in, w_in, isovalue,
                                      (p_DataIn->isConnected() != 0), iblank);
            uplane->createIsoPlane();
            uplane->createcoDistributedObjects(p_GridOut, p_NormalsOut, p_DataOut, gennormals, genstrips, colorn);
            delete uplane;
        }
        else if (strcmp(gtype, "RCTGRD") == 0)
        {
            rplane = new RECT_IsoPlane(numelem, numcoord, DataType,
                                       x_size, y_size, z_size,
                                       x_in, y_in, z_in,
                                       s_in, i_in, u_in, v_in, w_in, isovalue,
                                       (p_DataIn->isConnected() != 0), iblank);
            rplane->createIsoPlane();
            rplane->createcoDistributedObjects(p_GridOut, p_NormalsOut, p_DataOut, gennormals, genstrips, colorn);
            delete rplane;
        }
        else if (strcmp(gtype, "STRGRD") == 0)
        {
            splane = new STR_IsoPlane(numelem, numcoord, DataType,
                                      x_size, y_size, z_size,
                                      x_in, y_in, z_in, s_in, i_in, u_in, v_in, w_in, isovalue,
                                      (p_DataIn->isConnected() != 0), iblank);
            splane->createIsoPlane();
            splane->createcoDistributedObjects(p_GridOut, p_NormalsOut, p_DataOut, gennormals, genstrips, colorn);
            delete splane;
        }
    }
    if (grid_in)
    {
        grid_in->freeNeighborList();
    }
    return CONTINUE_PIPELINE;
}

void NullInputData(coOutputPort *p_GridOut, coOutputPort *p_NormalsOut, coOutputPort *p_DataOut,
                   int Datatype, int gennormals, int genstrips, const char *colorn)
{
    (void)colorn;
    coDoFloat *s_data_out;
    coDoVec3 *v_data_out;
    coDoPolygons *polygons_out;
    coDoTriangleStrips *strips_out;
    coDoVec3 *normals_out;
    const char *DataOut = p_DataOut->getObjName();
    const char *NormalsOut = p_NormalsOut->getObjName();
    const char *GridOut = p_GridOut->getObjName();
    // data
    if (Datatype) // (Scalar Data)
    {
        s_data_out = new coDoFloat(DataOut, 0);
        p_DataOut->setCurrentObject(s_data_out);
    }
    else
    {
        v_data_out = new coDoVec3(DataOut, 0);
        p_DataOut->setCurrentObject(v_data_out);
    }
    // normals
    if (gennormals)
    {
        normals_out = new coDoVec3(NormalsOut, 0);
        p_NormalsOut->setCurrentObject(normals_out);
    }
    // geometry
    if (genstrips)
    {
        strips_out = new coDoTriangleStrips(GridOut, 0, 0, 0);
        if (strips_out->objectOk())
        {
            strips_out->addAttribute("vertexOrder", "2");
            /*
                  strips_out->addAttribute("COLOR",colorn);
                  sprintf(buf,"I%s\n%s\n%s\n",Covise::get_module(),Covise::get_instance(),Covise::get_host());
                  strips_out->addAttribute("FEEDBACK", buf);
         */
        }
        p_GridOut->setCurrentObject(strips_out);
    }
    else
    {
        polygons_out = new coDoPolygons(GridOut, 0, 0, 0);
        if (polygons_out->objectOk())
        {
            //sprintf(buf,"I%s\n%s\n%s\n",Covise::get_module(),Covise::get_instance(),Covise::get_host());
            if (gennormals)
                polygons_out->addAttribute("vertexOrder", "1");
            else
                polygons_out->addAttribute("vertexOrder", "2");
        }
        else
        {
            Covise::sendError("ERROR: creation of dummy data object 'dataOut' failed");
            return;
        }
        // delete polygons_out;
        p_GridOut->setCurrentObject(polygons_out);
    }
}

//=====================================================================
//return the minimum of an float array
//=====================================================================
float Min_of(float array[], int length)
{
    int i;
    float val = array[0];

    for (i = 1; i < length; i++)
    {
        val = (val <= array[i]) ? val : array[i];
    }
    return val;
}

//=====================================================================
//return the maximum of an float array
//=====================================================================
float Max_of(float array[], int length)
{
    int i;
    float val = array[0];

    for (i = 1; i < length; i++)
    {
        val = (val >= array[i]) ? val : array[i];
    }
    return val;
}

//=====================================================================
// compare function for qsort library function
//=====================================================================
int compare(const void *v1, const void *v2)
{

    if (*((int *)v1) < *((int *)v2))
        return -1;
    else if (*((int *)v1) == *((int *)v2))
        return 0;
    else
        return 1;
}

//=====================================================================
// function to convert hexahedron to tetrahedron
//=====================================================================
void hex2tet(int ind, int i, int *tel, int *tcl, const int *el, const int *cl)
{
    int j;

    // fill the tel list
    for (j = 0; j < 5; j++)
    {
        tel[j] = j * 4;
    }

    // fill the tcl list
    if (ind > 0)
    {
        //positive decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 4];
        tcl[2] = cl[el[i] + 5];
        tcl[3] = cl[el[i] + 7];

        tcl[4] = cl[el[i]];
        tcl[5] = cl[el[i] + 5];
        tcl[6] = cl[el[i] + 2];
        tcl[7] = cl[el[i] + 7];

        tcl[8] = cl[el[i]];
        tcl[9] = cl[el[i] + 1];
        tcl[10] = cl[el[i] + 2];
        tcl[11] = cl[el[i] + 5];

        tcl[12] = cl[el[i]];
        tcl[13] = cl[el[i] + 2];
        tcl[14] = cl[el[i] + 3];
        tcl[15] = cl[el[i] + 7];

        tcl[16] = cl[el[i] + 2];
        tcl[17] = cl[el[i] + 5];
        tcl[18] = cl[el[i] + 6];
        tcl[19] = cl[el[i] + 7];
    }
    else if (ind < 0)
    {
        //negative decomposition
        tcl[0] = cl[el[i] + 3];
        tcl[1] = cl[el[i] + 4];
        tcl[2] = cl[el[i] + 6];
        tcl[3] = cl[el[i] + 7];

        tcl[4] = cl[el[i] + 1];
        tcl[5] = cl[el[i] + 3];
        tcl[6] = cl[el[i] + 4];
        tcl[7] = cl[el[i] + 6];

        tcl[8] = cl[el[i] + 1];
        tcl[9] = cl[el[i] + 2];
        tcl[10] = cl[el[i] + 3];
        tcl[11] = cl[el[i] + 6];

        tcl[12] = cl[el[i]];
        tcl[13] = cl[el[i] + 1];
        tcl[14] = cl[el[i] + 3];
        tcl[15] = cl[el[i] + 4];

        tcl[16] = cl[el[i] + 1];
        tcl[17] = cl[el[i] + 4];
        tcl[18] = cl[el[i] + 5];
        tcl[19] = cl[el[i] + 6];
    }

    return;
}

//=====================================================================
// function to convert pyramid to tetrahedron
//=====================================================================
void pyra2tet(int ind, int i, int *tel, int *tcl, const int *el, const int *cl)
{
    // fill the tel list
    tel[0] = 0;
    tel[1] = 4;

    // fill the tcl list
    if (ind > 0)
    {
        //positive decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 1];
        tcl[2] = cl[el[i] + 3];
        tcl[3] = cl[el[i] + 4];

        tcl[4] = cl[el[i] + 1];
        tcl[5] = cl[el[i] + 2];
        tcl[6] = cl[el[i] + 3];
        tcl[7] = cl[el[i] + 4];
    }
    else if (ind < 0)
    {
        //negative decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 1];
        tcl[2] = cl[el[i] + 2];
        tcl[3] = cl[el[i] + 4];

        tcl[4] = cl[el[i]];
        tcl[5] = cl[el[i] + 2];
        tcl[6] = cl[el[i] + 3];
        tcl[7] = cl[el[i] + 4];
    }

    return;
}

//=====================================================================
// function to convert prism to tetrahedron
//=====================================================================
void prism2tet(int ind, int i, int *tel, int *tcl, const int *el, const int *cl)
{
    int j;

    // fill the tel list
    for (j = 0; j < 3; j++)
    {
        tel[j] = j * 4;
    }

    // fill the tcl list
    if (ind > 0)
    {
        //positive decomposition
        tcl[0] = cl[el[i]];
        tcl[1] = cl[el[i] + 1];
        tcl[2] = cl[el[i] + 4];
        tcl[3] = cl[el[i] + 5];

        tcl[4] = cl[el[i]];
        tcl[5] = cl[el[i] + 1];
        tcl[6] = cl[el[i] + 3];
        tcl[7] = cl[el[i] + 5];

        tcl[8] = cl[el[i] + 1];
        tcl[9] = cl[el[i] + 2];
        tcl[10] = cl[el[i] + 3];
        tcl[11] = cl[el[i] + 5];
    }
    else if (ind < 0)
    {
        //negative decomposition
        tcl[0] = cl[el[i] + 2];
        tcl[1] = cl[el[i] + 3];
        tcl[2] = cl[el[i] + 4];
        tcl[3] = cl[el[i] + 5];

        tcl[4] = cl[el[i]];
        tcl[5] = cl[el[i] + 1];
        tcl[6] = cl[el[i] + 2];
        tcl[7] = cl[el[i] + 4];

        tcl[8] = cl[el[i]];
        tcl[9] = cl[el[i] + 2];
        tcl[10] = cl[el[i] + 3];
        tcl[11] = cl[el[i] + 4];
    }

    return;
}

//=====================================================================
// function to compute the volume of a tetrahedral cell
//=====================================================================
float tetra_vol(const float p0[3], const float p1[3], const float p2[3], const float p3[3])
{
    //returns the volume of the tetrahedral cell
    float vol;

    vol = (((p2[1] - p0[1]) * (p3[2] - p0[2]) - (p3[1] - p0[1]) * (p2[2] - p0[2])) * (p1[0] - p0[0]) + ((p2[2] - p0[2]) * (p3[0] - p0[0]) - (p3[2] - p0[2]) * (p2[0] - p0[0])) * (p1[1] - p0[1]) + ((p2[0] - p0[0]) * (p3[1] - p0[1]) - (p3[0] - p0[0]) * (p2[1] - p0[1])) * (p1[2] - p0[2])) / 6.0f;

    return vol;
}

//=====================================================================
// is the point inside the tetrahedral cell
//=====================================================================
int isin_tetra(const float px[3],
               const float p0[3], const float p1[3], const float p2[3], const float p3[3])
{
    //returns 1 if point px is inside the tetrahedra cell, else 0
    float vg, w0, w1, w2, w3;

    vg = fabs(tetra_vol(p0, p1, p2, p3));

    w0 = fabs(tetra_vol(px, p1, p2, p3));
    w0 /= vg;

    w1 = fabs(tetra_vol(p0, px, p2, p3));
    w1 /= vg;

    w2 = fabs(tetra_vol(p0, p1, px, p3));
    w2 /= vg;

    w3 = fabs(tetra_vol(p0, p1, p2, px));
    w3 /= vg;

    if (w0 + w1 + w2 + w3 <= 1. + 0.000001)
        return 1;

    else
        return 0;
}

void IsoSurface::setIterator(coInputPort **inPorts, int t)
{
    if (p_pointOrValue->getValue() == VALUE)
    {
        return;
    }
    const char *dataType;

    dataType = (inPorts[0]->getCurrentObject())->getType();
    if (strcmp(dataType, "SETELE") == 0 && inPorts[0]->getCurrentObject()->getAttribute("TIMESTEP"))
        lookUp = t;
    return;
}

//=====================================================================
// find the startcell (fast)
//=====================================================================

class NextFactor
{
public:
    enum Result
    {
        TOO_SMALL,
        TOO_BIG
    };
    NextFactor();
    float SuggestFactor(float factor, Result);

private:
    float _inf;
    float _sup;
};

NextFactor::NextFactor()
{
    _inf = -FLT_MAX;
    _sup = FLT_MAX;
}

float
NextFactor::SuggestFactor(float factor, Result result)
{
    switch (result)
    {
    case TOO_SMALL:
        _inf = factor;
        break;
    case TOO_BIG:
        _sup = factor;
    }
    if (_inf == -FLT_MAX)
    {
        return factor * 0.5f;
    }
    else if (_sup == FLT_MAX)
    {
        return (factor + factor);
    }
    else
    {
        return 0.5f * (_inf + _sup);
    }
}

int find_startcell_fast(const float *p,
                        const float *x_in, const float *y_in, const float *z_in,
                        int numelem, int numconn,
                        const int *el, const int *cl, const int *tl,
                        int cuc_count, const int *cuc, const int *cuc_pos)
{
    (void)cuc_count;
    int i, j, k, mark;
    int tmp_el[5], tmp_cl[20];
    int *tmp_inbox, tmp_inbox_count, *inbox, inbox_count;
    int *pib, pib_count;

    float p0[3], p1[3], p2[3], p3[3];
    //   float tp[3];
    float xb[8], yb[8], zb[8];
    float tmp_length[3];
    float x_min, x_max, y_min, y_max, z_min, z_max;

    float factor = 2;

    pib = new int[5000]; // Maximum of 2000 points in a box

    vector<int> temp_elem_in;
    vector<int> temp_conn_in;
    vector<int> temp_vertex_list;

    // create ref box
    switch (tl[0])
    {
    case TYPE_HEXAGON:
    {
        //Covise::sendInfo("INFO: DataSet -> TYPE_HEXAGON");
        for (i = 0; i < 8; i++)
        {
            xb[i] = x_in[cl[el[0] + i]];
            yb[i] = y_in[cl[el[0] + i]];
            zb[i] = z_in[cl[el[0] + i]];
        }

        tmp_length[0] = Max_of(xb, 8) - Min_of(xb, 8);
        tmp_length[1] = Max_of(yb, 8) - Min_of(yb, 8);
        tmp_length[2] = Max_of(zb, 8) - Min_of(zb, 8);
    }
    break;

    case TYPE_PRISM:
    {
        //Covise::sendInfo("INFO: DataSet -> TYPE_PRISM");
        for (i = 0; i < 6; i++)
        {
            xb[i] = x_in[cl[el[0] + i]];
            yb[i] = y_in[cl[el[0] + i]];
            zb[i] = z_in[cl[el[0] + i]];
        }

        tmp_length[0] = Max_of(xb, 6) - Min_of(xb, 6);
        tmp_length[1] = Max_of(yb, 6) - Min_of(yb, 6);
        tmp_length[2] = Max_of(zb, 6) - Min_of(zb, 6);
    }
    break;

    case TYPE_PYRAMID:
    {
        //Covise::sendInfo("INFO: DataSet -> TYPE_PYRAMID");
        for (i = 0; i < 5; i++)
        {
            xb[i] = x_in[cl[el[0] + i]];
            yb[i] = y_in[cl[el[0] + i]];
            zb[i] = z_in[cl[el[0] + i]];
        }

        tmp_length[0] = Max_of(xb, 5) - Min_of(xb, 5);
        tmp_length[1] = Max_of(yb, 5) - Min_of(yb, 5);
        tmp_length[2] = Max_of(zb, 5) - Min_of(zb, 5);
    }
    break;

    case TYPE_TETRAHEDER:
    {
        for (i = 0; i < 4; i++)
        {
            //Covise::sendInfo("INFO: DataSet -> TYPE_TETRAHEDER");
            xb[i] = x_in[cl[el[0] + i]];
            yb[i] = y_in[cl[el[0] + i]];
            zb[i] = z_in[cl[el[0] + i]];
        }

        tmp_length[0] = Max_of(xb, 4) - Min_of(xb, 4);
        tmp_length[1] = Max_of(yb, 4) - Min_of(yb, 4);
        tmp_length[2] = Max_of(zb, 4) - Min_of(zb, 4);
    }
    break;

    case TYPE_POLYHEDRON:
    {
        bool start_vertex_set;

        int i;
        int cell;
        //int current_cell;
        int next_elem_index;
        int start_vertex;

        float *temp_xb;
        float *temp_yb;
        float *temp_zb;

        /***********************************************/
        /* Calculation of the number of nodes in the cell */
        /***********************************************/

        start_vertex_set = false;
        cell = 0;
        //current_cell = cell;
        next_elem_index = (cell < numelem) ? el[cell + 1] : numconn;

        /* Construct DO_Polygons Element and Connectivity Lists */
        for (i = el[cell]; i < next_elem_index; i++)
        {
            if (i == el[cell] && start_vertex_set == false)
            {
                start_vertex = cl[el[cell]];
                temp_elem_in.push_back((int)temp_conn_in.size());
                temp_conn_in.push_back(start_vertex);
                start_vertex_set = true;
            }

            if (i > el[cell] && start_vertex_set == true)
            {
                if (cl[i] != start_vertex)
                {
                    temp_conn_in.push_back(cl[i]);
                }

                else
                {
                    start_vertex_set = false;
                    continue;
                }
            }

            if (i > el[cell] && start_vertex_set == false)
            {
                start_vertex = cl[i];
                temp_elem_in.push_back((int)temp_conn_in.size());
                temp_conn_in.push_back(start_vertex);
                start_vertex_set = true;
            }
        }

        /* Construct Vertex List */
        for (i = 0; i < temp_conn_in.size(); i++)
        {
            if (temp_vertex_list.size() == 0)
            {
                temp_vertex_list.push_back(temp_conn_in[i]);
            }

            else
            {
                if (find(temp_vertex_list.begin(), temp_vertex_list.end(), temp_conn_in[i]) == temp_vertex_list.end())
                {
                    temp_vertex_list.push_back(temp_conn_in[i]);
                }
            }
        }

        sort(temp_vertex_list.begin(), temp_vertex_list.end());

        temp_xb = new float[temp_vertex_list.size()];
        temp_yb = new float[temp_vertex_list.size()];
        temp_zb = new float[temp_vertex_list.size()];

        for (i = 0; i < temp_vertex_list.size(); i++)
        {
            temp_xb[i] = x_in[temp_vertex_list[i]];
            temp_yb[i] = y_in[temp_vertex_list[i]];
            temp_zb[i] = z_in[temp_vertex_list[i]];
        }

        tmp_length[0] = Max_of(temp_xb, (int)temp_vertex_list.size()) - Min_of(temp_xb, (int)temp_vertex_list.size());
        tmp_length[1] = Max_of(temp_yb, (int)temp_vertex_list.size()) - Min_of(temp_yb, (int)temp_vertex_list.size());
        tmp_length[2] = Max_of(temp_zb, (int)temp_vertex_list.size()) - Min_of(temp_zb, (int)temp_vertex_list.size());

        delete[] temp_xb;
        delete[] temp_yb;
        delete[] temp_zb;
    }
    break;

    default:
    {
        return -2;
    }
    }

    pib_count = 0;
    // find all coords inside the box
    NextFactor nextFactor;

    while (1)
    {
        //cerr << "factor: "<< factor <<"\n";
        /*
            if(factor<=0)
               return(-1);
      */
        float box_length = factor * Max_of(tmp_length, 3);
        //int box_flag = 1;

        // compute the absolute position of the box
        x_min = p[0] - box_length * 0.5f;
        x_max = p[0] + box_length * 0.5f;
        y_min = p[1] - box_length * 0.5f;
        y_max = p[1] + box_length * 0.5f;
        z_min = p[2] - box_length * 0.5f;
        z_max = p[2] + box_length * 0.5f;
        pib_count = 0;

        for (i = 0; i < num_coord; i++) //every coord
        {

            if (x_in[i] > x_min && x_in[i] < x_max)
            {
                if (y_in[i] > y_min && y_in[i] < y_max)
                {
                    if (z_in[i] > z_min && z_in[i] < z_max)
                    {
                        if (pib_count >= 5000)
                        {
                            //cerr << "pib_count out of range\n";
                            // factor-=1.5;
                            factor = nextFactor.SuggestFactor(factor, NextFactor::TOO_BIG);
                            break;
                        }
                        pib[pib_count] = i;
                        pib_count++;
                    }
                }
            }
        }
        if (pib_count >= 8 && pib_count < 5000) // 8 is not always a good choice!!!
        {
            break;
        }
        else if (pib_count < 8)
        {
            factor = nextFactor.SuggestFactor(factor, NextFactor::TOO_SMALL);
        }
    }

    // compute the size of inbox
    tmp_inbox_count = 0;
    for (i = 0; i < pib_count; i++)
    {
        tmp_inbox_count += cuc_pos[pib[i] + 1] - cuc_pos[pib[i]];
    }

    tmp_inbox = new int[tmp_inbox_count];

    // filling tmp_inbox list
    k = 0;
    for (i = 0; i < pib_count; i++)
    {
        for (j = cuc_pos[pib[i]]; j < cuc_pos[pib[i] + 1]; j++)
        {
            tmp_inbox[k] = cuc[j];
            k++;
        }
    }

    qsort(tmp_inbox, k, sizeof(int), compare);

    // create the inbox list
    inbox = new int[tmp_inbox_count];
    inbox_count = 0;
    mark = tmp_inbox[0];
    inbox[0] = tmp_inbox[0];
    inbox_count = 1;

    for (i = 1; i < k; i++)
    {
        if (tmp_inbox[i] != mark)
        {
            inbox[inbox_count] = tmp_inbox[i];
            mark = tmp_inbox[i];
            inbox_count++;
        }
    }

    // look for cell inside the box
    for (i = 0; i < inbox_count; i++)
    {
        switch (tl[inbox[i]])
        {

        case TYPE_HEXAGON:
        {
            hex2tet(1, inbox[i], tmp_el, tmp_cl, el, cl);
            for (j = 0; j < 5; j++)
            {
                p0[0] = x_in[tmp_cl[tmp_el[j]]];
                p0[1] = y_in[tmp_cl[tmp_el[j]]];
                p0[2] = z_in[tmp_cl[tmp_el[j]]];

                p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                if (isin_tetra(p, p0, p1, p2, p3) == 1)
                {
                    k = inbox[i];

                    delete[] pib;
                    delete[] tmp_inbox;
                    delete[] inbox;

                    return k;
                }
            }
        }
        break;

        case TYPE_PRISM:
        {
            prism2tet(1, inbox[i], tmp_el, tmp_cl, el, cl);
            for (j = 0; j < 3; j++)
            {
                p0[0] = x_in[tmp_cl[tmp_el[j]]];
                p0[1] = y_in[tmp_cl[tmp_el[j]]];
                p0[2] = z_in[tmp_cl[tmp_el[j]]];

                p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                if (isin_tetra(p, p0, p1, p2, p3) == 1)
                {
                    k = inbox[i];

                    delete[] pib;
                    delete[] tmp_inbox;
                    delete[] inbox;

                    return k;
                }
            }
        }
        break;

        case TYPE_PYRAMID:
        {
            pyra2tet(1, inbox[i], tmp_el, tmp_cl, el, cl);
            for (j = 0; j < 2; j++)
            {
                p0[0] = x_in[tmp_cl[tmp_el[j]]];
                p0[1] = y_in[tmp_cl[tmp_el[j]]];
                p0[2] = z_in[tmp_cl[tmp_el[j]]];

                p1[0] = x_in[tmp_cl[tmp_el[j] + 1]];
                p1[1] = y_in[tmp_cl[tmp_el[j] + 1]];
                p1[2] = z_in[tmp_cl[tmp_el[j] + 1]];

                p2[0] = x_in[tmp_cl[tmp_el[j] + 2]];
                p2[1] = y_in[tmp_cl[tmp_el[j] + 2]];
                p2[2] = z_in[tmp_cl[tmp_el[j] + 2]];

                p3[0] = x_in[tmp_cl[tmp_el[j] + 3]];
                p3[1] = y_in[tmp_cl[tmp_el[j] + 3]];
                p3[2] = z_in[tmp_cl[tmp_el[j] + 3]];

                if (isin_tetra(p, p0, p1, p2, p3) == 1)
                {
                    k = inbox[i];

                    delete[] pib;
                    delete[] tmp_inbox;
                    delete[] inbox;

                    return k;
                }
            }
        }
        break;

        case TYPE_TETRAHEDER:
        {
            p0[0] = x_in[cl[el[inbox[i]]]];
            p0[1] = y_in[cl[el[inbox[i]]]];
            p0[2] = z_in[cl[el[inbox[i]]]];

            p1[0] = x_in[cl[el[inbox[i]] + 1]];
            p1[1] = y_in[cl[el[inbox[i]] + 1]];
            p1[2] = z_in[cl[el[inbox[i]] + 1]];

            p2[0] = x_in[cl[el[inbox[i]] + 2]];
            p2[1] = y_in[cl[el[inbox[i]] + 2]];
            p2[2] = z_in[cl[el[inbox[i]] + 2]];

            p3[0] = x_in[cl[el[inbox[i]] + 3]];
            p3[1] = y_in[cl[el[inbox[i]] + 3]];
            p3[2] = z_in[cl[el[inbox[i]] + 3]];

            if (isin_tetra(p, p0, p1, p2, p3) == 1)
            {
                k = inbox[i];

                delete[] pib;
                delete[] tmp_inbox;
                delete[] inbox;

                return k;
            }
        }
        break;

        case TYPE_POLYHEDRON:
        {
            // Instead of a decomposition into tetrahedra, an in-polyhedron test
            // is applied for arbitrary polyhedra.

            char inclusion_test;

            int l;
            int m;
            int cell_radius;

            int *temp_elem_list;
            int *temp_conn_list;

            float *new_x_coord_in;
            float *new_y_coord_in;
            float *new_z_coord_in;

            vector<int> new_temp_conn_in;

            grid_methods::POINT3D cell_box_min;
            grid_methods::POINT3D cell_box_max;
            grid_methods::POINT3D end_point;
            grid_methods::POINT3D particle_location;
            particle_location.x = 0.0;
            particle_location.y = 0.0;
            particle_location.z = 0.0;

            grid_methods::TESSELATION triangulated_cell;

            /* Construct New Connectivity List */
            for (l = 0; l < temp_conn_in.size(); l++)
            {
                for (m = 0; m < temp_vertex_list.size(); m++)
                {
                    if (temp_conn_in[l] == temp_vertex_list[m])
                    {
                        new_temp_conn_in.push_back(m);
                        break;
                    }
                }
            }

            temp_elem_list = new int[temp_elem_in.size()];
            temp_conn_list = new int[temp_conn_in.size()];
            new_x_coord_in = new float[temp_vertex_list.size()];
            new_y_coord_in = new float[temp_vertex_list.size()];
            new_z_coord_in = new float[temp_vertex_list.size()];

            for (l = 0; l < temp_elem_in.size(); l++)
            {
                temp_elem_list[l] = temp_elem_in[l];
            }

            for (l = 0; l < new_temp_conn_in.size(); l++)
            {
                temp_conn_list[l] = new_temp_conn_in[l];
            }

            /* Construct New Set of Coordinates */
            for (l = 0; l < temp_vertex_list.size(); l++)
            {
                new_x_coord_in[l] = x_in[temp_vertex_list[l]];
                new_y_coord_in[l] = y_in[temp_vertex_list[l]];
                new_z_coord_in[l] = z_in[temp_vertex_list[l]];
            }

            grid_methods::TesselatePolyhedron(triangulated_cell, (int)temp_elem_in.size(), temp_elem_list, (int)new_temp_conn_in.size(), temp_conn_list, new_x_coord_in, new_y_coord_in, new_z_coord_in);

            grid_methods::ComputeBoundingBox((int)temp_vertex_list.size(), new_x_coord_in, new_y_coord_in, new_z_coord_in, cell_box_min, cell_box_max, cell_radius /*, cell_box_vertices*/);

            inclusion_test = grid_methods::InPolyhedron(new_x_coord_in, new_y_coord_in, new_z_coord_in, cell_box_min, cell_box_max, particle_location, end_point, cell_radius, triangulated_cell);

            if (inclusion_test == 'i' || inclusion_test == 'V' || inclusion_test == 'E' || inclusion_test == 'F')
            {
                k = inbox[i];

                delete[] pib;
                delete[] tmp_inbox;
                delete[] inbox;

                delete[] temp_elem_list;
                delete[] temp_conn_list;
                delete[] new_x_coord_in;
                delete[] new_y_coord_in;
                delete[] new_z_coord_in;

                return k;
            }
        }
        break;

        default:
            delete[] pib;
            delete[] tmp_inbox;
            delete[] inbox;

            return -2;
        }
    }

    delete[] pib;
    delete[] tmp_inbox;
    delete[] inbox;

    return -1;
}

IsoSurface::IsoSurface(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Determine an isosurface by a point or value", true)
    , isovalue(isovalueHack[0])
#ifdef _COMPLEX_MODULE_
    , shiftOut(1)
#else
    , shiftOut(0)
#endif
    , _autominmax(false)
    , _min(FLT_MAX)
    , _max(-FLT_MAX)
    , inSelfExec(false)
{
#ifdef _COMPLEX_MODULE_
    autoTitleConfigured = coCoviseConfig::isOn("System.AutoName.IsoSurfaceComp", false);
#else
    autoTitleConfigured = coCoviseConfig::isOn("System.AutoName.IsoSurface", false);
#endif

    Polyhedra = coCoviseConfig::isOn("Module.IsoSurface.SupportPolyhedra", true);

    /// Send old-style or new-style feedback: Default values different HLRS/Vrc
    fbStyle_ = FEED_NEW;
    std::string fbStyleStr = coCoviseConfig::getEntry("System.FeedbackStyle.IsoSurface");
    if (!fbStyleStr.empty())
    {
        if (0 == strncasecmp("NONE", fbStyleStr.c_str(), 4))
            fbStyle_ = FEED_NONE;
        if (0 == strncasecmp("OLD", fbStyleStr.c_str(), 3))
            fbStyle_ = FEED_OLD;
        if (0 == strncasecmp("NEW", fbStyleStr.c_str(), 3))
            fbStyle_ = FEED_NEW;
        if (0 == strncasecmp("BOTH", fbStyleStr.c_str(), 4))
            fbStyle_ = FEED_BOTH;
    }

    // initially we do what is configured in covise.config - but User may override
    // by setting his own title: done in param()
    autoTitle = autoTitleConfigured;

    // input ports
    p_GridIn = addInputPort("GridIn0", "UnstructuredGrid|UniformGrid|StructuredGrid|RectilinearGrid", "Grid");
    p_IsoDataIn = addInputPort("DataIn0", "Float", "Data for isosurface generation");
    p_DataIn = addInputPort("DataIn1", "Float|Vec3", "Data to be mapped onto the isosurface");
    p_DataIn->setRequired(0);
    p_IBlankIn = addInputPort("DataIn2", "Text", "this char Array marks cells to be processed or not");
    p_IBlankIn->setRequired(0);

#ifdef _COMPLEX_MODULE_
    p_GeometryOut = addOutputPort("GeometryOut0", "Geometry", "Colored isoSurface");
#endif
    // output ports
    p_GridOut = addOutputPort("GridOut0", "Polygons|TriangleStrips", "The isosurface");
    p_DataOut = addOutputPort("DataOut0", "Float|Vec3", "interpolated data");
    p_NormalsOut = addOutputPort("DataOut1", "Vec3", "Surface normals");

    // Parameters
    p_gennormals = addBooleanParam("gennormals", "Supply normals");
    p_gennormals->setValue(1);

    p_genstrips = addBooleanParam("genstrips", "Convert triangles to strips");
    p_genstrips->setValue(0);

    p_pointOrValue = addChoiceParam("Interactor", "Point or value working mode");
    const char *Modi[] = { "Point", "Value" };
    p_pointOrValue->setValue(2, Modi, 1);

    p_isopoint = addFloatVectorParam("isopoint", "Point for isosurface");
    p_isopoint->setValue(0.0, 0.0, 0.0);

    p_isovalue = addFloatSliderParam("isovalue", "Value for isosurfaces");
    p_isovalue->setValue(0.0, 1.0, 0.5);

    p_autominmax_ = addBooleanParam("autominmax", "Automatic minmax");
    p_autominmax_->setValue(1);

    p_isovalue->setValue(0.0);
    p_gennormals->setValue(1);
    p_genstrips->setValue(1);
    p_isopoint->setValue(0.0, 0.0, 0.0);

    //
    level = 0;
    lookUp = 0;

    vertexRatio = coCoviseConfig::getFloat("Module.IsoSurface.VertexRatio", 20.);
#ifdef _COMPLEX_MODULE_
    p_ColorMapIn = addInputPort("ColormapIn0", "ColorMap", "color map to create geometry");
    p_ColorMapIn->setRequired(0);
    p_color_or_texture = addBooleanParam("color_or_texture", "colors or texture");
    p_color_or_texture->setValue(1);

    const char *ChoiseVal1[] = { "1*scale", "length*scale" /*,"according_to_data"*/ };
    p_scale = addFloatSliderParam("scale", "Scale factor");
    p_scale->setValue(0.0, 1.0, 1.0);
    p_length = addChoiceParam("length", "Length of vectors");
    p_length->setValue(2, ChoiseVal1, 0);
    p_num_sectors = addInt32Param("num_sectors", "number of lines for line tip");
    p_num_sectors->setValue(0);

    const char *vector_labels[] = { "SurfaceAndLines", "OnlySurface", "OnlyLines" };
    p_vector = addChoiceParam("vector", "SurfaceOrLines");
    p_vector->setValue(3, vector_labels, 0);
#endif
}

void
IsoSurface::addFeedbackParams(coDistributedObject *obj)
{
    if (fbStyle_ == FEED_OLD || fbStyle_ == FEED_BOTH)
    {
        sprintf(buf, "I%s\n%s\n%s\n%f\n%f\n%f\n",
                Covise::get_module(), Covise::get_instance(), Covise::get_host(),
                p_isovalue->getMin(), p_isovalue->getMax(),
                p_isovalue->getValue());
        obj->addAttribute("FEEDBACK", buf);
    }

    if (fbStyle_ == FEED_NEW || fbStyle_ == FEED_BOTH)
    {
        coFeedback feedback("IsoSurface");
        feedback.addPara(p_pointOrValue);
        feedback.addPara(p_isopoint);
        feedback.addPara(p_isovalue);
        feedback.addPara(p_gennormals);
        feedback.addPara(p_genstrips);
#ifdef _COMPLEX_MODULE_
        feedback.addPara(p_color_or_texture);
        feedback.addPara(p_scale);
        feedback.addPara(p_length);
        feedback.addPara(p_num_sectors);
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
        if (strcmp(t, "IsoSurface") != 0)
        {
            feedback.addString(ud);
        }
        delete[] t;
        delete[] ud;
        feedback.apply(obj);
    }
}

#ifdef _COMPLEX_MODULE_

void
IsoSurface::StaticParts(coDistributedObject **geopart,
                        coDistributedObject **normpart,
                        coDistributedObject **colorpart,
                        const coDistributedObject *geo,
                        const coDistributedObject *data,
                        string geometryOutName,
                        bool ColorMapAttrib,
                        const ScalarContainer *SCont)
{
    // coDistributedObject *geo = p_MeshOut->getCurrentObject();
    int vectOption = p_vector->getValue();
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
    coDistributedObject *colorSurf = NULL;
    coDistributedObject *colorLines = NULL;

    coDistributedObject *arrows = ComplexModules::MakeArrows(nameArrows.c_str(),
                                                             geo, data,
                                                             color_name.c_str(),
                                                             &colorSurf, &colorLines, factor, p_ColorMapIn->getCurrentObject(),
                                                             ColorMapAttrib, SCont, p_scale->getValue(), p_length->getValue(),
                                                             p_num_sectors->getValue(), 0, vectOption);

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
        string nameGeom = geometryOutName; // p_GeometryOut->getObjName();
        nameGeom += "_Geom";
        *geopart = geo->clone(nameGeom);
    }
    else
    {
        *geopart = arrows;
    }

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
    }

    if (vectOption == 0)
    {
        coDistributedObject **setList = new coDistributedObject *[3];
        setList[2] = NULL;
        setList[0] = colorSurf;
        setList[1] = colorLines;
        string nameColor = geometryOutName; //p_GeometryOut->getObjName();
        nameColor += "_AllColor";
        *colorpart = new coDoSet(nameColor.c_str(), setList);
        delete colorLines;
        delete[] setList;
    }
    else if (vectOption == 1)
    {
        *colorpart = colorSurf;
    }
    else if (vectOption == 2)
    {
        *colorpart = colorLines;
    }
}
#endif

MODULE_MAIN(Mapper, IsoSurface)
