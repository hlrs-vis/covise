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

#include "ShiftValues.h"
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
ShiftValues::UpdateIsoValue()
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

void ShiftValues::preHandleObjects(coInputPort **InPorts)
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

    fillThePoint();
    setUpIsoList(InPorts);
    return;
}

void
ShiftValues::copyAttributesToOutObj(coInputPort **input_ports,
                                   coOutputPort **output_ports, int port)
{
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

void ShiftValues::postHandleObjects(coOutputPort **OutPorts)
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

    addFeedbackParams(OutPorts[shiftOut]->getCurrentObject());

    objLabVal.clean();
    return;
}

void ShiftValues::setUpIsoList(coInputPort **inPorts)
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

myPair ShiftValues::find_isovalueT(const coDistributedObject *inObj,
                                  const coDistributedObject *idata)
{
    findCurrentTimestep = 0;
    return find_isovalueSGoT(inObj, idata);
}

myPair ShiftValues::find_isovalueSG(const coDistributedObject *inObj,
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

myPair ShiftValues::find_isovalueSGoT(const coDistributedObject *inObj,
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

        if(setIsT == 1)
        {
            referenceValues.resize(numSetElem);
        }
        isoDataList = p_IsoData->getAllElements(&t);
        if (numSetElem != t)
            sendError("Number of elements in matching sets do not coincide");
        for (int i = 0; i < numSetElem; ++i)
        {
            if(setIsT == 1)
            {
                findCurrentTimestep = i;
            }
            if (gridList[i] == 0)
                continue;
            // Proceed only for non-set sons
            // or set-sons that are not time steps
            if (strcmp(gridList[i]->getType(), "SETELE") != 0 || !(gridList[i]->getAttribute("TIMESTEP")))
            {
                resultSons = find_isovalueSG(gridList[i], isoDataList[i]);
                if(resultSons.Tag ==TRIUMPH)
                {
                    referenceValues[findCurrentTimestep] = resultSons.Value;
                }
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
        else
        {
            sendError(" cannot process this kind of data");
        }
    }
    return resultSons;
}

myPair ShiftValues::find_isovalueU(const coDoUniformGrid *i_mesh,
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

myPair ShiftValues::find_isovalueR(const coDoRectilinearGrid *i_mesh,
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

myPair ShiftValues::find_isovalueS(const coDoStructuredGrid *i_mesh,
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

myPair ShiftValues::find_isovalueUU(const coDoUnstructuredGrid *i_mesh,
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
    cell = ((coDoUnstructuredGrid *)i_mesh)->getCell(startp,0.001f);
    //cell = find_startcell_fast(startp, x_in, y_in, z_in, numelem, numconn, el, cl, tl,
    //                           cuc_count, cuc, cuc_pos);
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

myPair ShiftValues::find_isovalueUS(const coDoUnstructuredGrid *i_mesh,
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
ShiftValues::param(const char *paramName, bool inMapLoading)
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

            p_isovalue->disable();
            p_isopoint->enable();

    if (pName == p_autominmax_->getName() && !inMapLoading)
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
    }
    else if (pName == p_isovalue->getName())
    {
        p_isovalue->enable();
        p_isopoint->disable();
    }
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
int ShiftValues::compute(const char *)
{

    //input port object
    const coDistributedObject *obj = p_IsoDataIn->getCurrentObject();
    int num_values;
    float *s_out;
    float *u = NULL;
    int refIndex=12;
coDoFloat *u_scalar_data = NULL;
    if (!obj)
    {
        //no input, no output
        sendError("Did not receive object at port '%s'", p_IsoDataIn->getName());
        return FAIL;
    }


    //coDistributedObject *data = p_DataOut->getCurrentObject();

    const coDoFloat *float_data = (const coDoFloat *)obj;
    num_values = float_data->getNumPoints();
    float_data->getAddress(&u);
        u_scalar_data = new coDoFloat(p_DataOut->getObjName(), num_values); 
	if (!u_scalar_data->objectOk()) 
        { 
            const char *name = NULL; 
            if (p_DataOut->getCurrentObject()) 
                name = p_DataOut->getCurrentObject()->getName(); 
            sendError("Failed to create the object '%s' for the port '%s'", name, p_DataOut->getName()); 
            return FAIL; 
        } 
        fprintf(stderr,"RefValue = %f, T = %d\n",referenceValues[currentTimestep],currentTimestep);
	u_scalar_data->getAddress(&s_out); 

            for (int i = 0; i < num_values; i++)
            {
                *s_out = u[i] -referenceValues[currentTimestep];
                s_out++;
            }
p_DataOut->setCurrentObject(u_scalar_data);

    return SUCCESS;
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
            
                 // strips_out->addAttribute("COLOR",colorn);
                 // sprintf(buf,"I%s\n%s\n%s\n",Covise::get_module(),Covise::get_instance(),Covise::get_host());
                 // strips_out->addAttribute("FEEDBACK", buf);
         
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

void ShiftValues::setIterator(coInputPort **inPorts, int t)
{
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
    int pib_count;
    std::vector<int> pib;
    pib.reserve(5000);

    float p0[3], p1[3], p2[3], p3[3];
    //   float tp[3];
    float xb[8], yb[8], zb[8];
    float tmp_length[3];
    float x_min, x_max, y_min, y_max, z_min, z_max;

    float factor = 2;

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

    bool box_increased = false;
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
        pib.clear();
        for (i = 0; i < num_coord; i++) //every coord
        {

            if (x_in[i] > x_min && x_in[i] < x_max)
            {
                if (y_in[i] > y_min && y_in[i] < y_max)
                {
                    if (z_in[i] > z_min && z_in[i] < z_max)
                    {
                        if (pib_count >= 5000 && !box_increased)
                        {
                            //cerr << "pib_count out of range\n";
                            // factor-=1.5;
                            factor = nextFactor.SuggestFactor(factor, NextFactor::TOO_BIG);
                            break;
                        }
                        pib.push_back(i);
                        pib_count++;
                    }
                }
            }
        }
        fprintf(stderr,"pib_count = %d\n", pib_count);
        if (pib_count >= 8 && pib_count < 5000) // 8 is not always a good choice!!!
        {
            break;
        }
        else if (pib_count < 8)
        {
            factor = nextFactor.SuggestFactor(factor, NextFactor::TOO_SMALL);
            box_increased = true;
        }
        else if (box_increased)
        {
            break;
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
            delete[] tmp_inbox;
            delete[] inbox;

            return -2;
        }
    }

    delete[] tmp_inbox;
    delete[] inbox;

    return -1;
}

ShiftValues::ShiftValues(int argc, char *argv[])
    : coSimpleModule(argc, argv, "Shift values so that selectes point has constant value", true)
    , isovalue(isovalueHack[0])
    , shiftOut(0)
    , _autominmax(false)
    , _min(FLT_MAX)
    , _max(-FLT_MAX)
    , inSelfExec(false)
{
    autoTitleConfigured = coCoviseConfig::isOn("System.AutoName.ShiftValues", false);

    Polyhedra = coCoviseConfig::isOn("Module.ShiftValues.SupportPolyhedra", true);

    /// Send old-style or new-style feedback: Default values different HLRS/Vrc
    fbStyle_ = FEED_NEW;
    std::string fbStyleStr = coCoviseConfig::getEntry("System.FeedbackStyle.ShiftValues");
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

    // output ports
    p_DataOut = addOutputPort("DataOut0", "Float|Vec3", "interpolated data");

    // Parameters

    p_isopoint = addFloatVectorParam("isopoint", "Point for isosurface");
    p_isopoint->setValue(0.0, 0.0, 0.0);

    p_isovalue = addFloatSliderParam("isovalue", "Value for isosurfaces");
    p_isovalue->setValue(0.0, 1.0, 0.5);

    p_autominmax_ = addBooleanParam("autominmax", "Automatic minmax");
    p_autominmax_->setValue(1);

    p_isovalue->setValue(0.0);
    p_isopoint->setValue(0.0, 0.0, 0.0);

    //
    level = 0;
    lookUp = 0;
}

void
ShiftValues::addFeedbackParams(coDistributedObject *obj)
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
        coFeedback feedback("ShiftValues");
        feedback.addPara(p_isopoint);
        feedback.addPara(p_isovalue);

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
        if (strcmp(t, "ShiftValues") != 0)
        {
            feedback.addString(ud);
        }
        delete[] t;
        delete[] ud;
        feedback.apply(obj);
    }
}


MODULE_MAIN(Mapper, ShiftValues)
