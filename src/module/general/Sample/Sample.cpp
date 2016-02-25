/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                (C)1999-2000 VirCinity  **
 **   Sample module                                                        **
 **                                                                        **
 ** Authors: Ralph Bruckschen (Vircinity)                                  **
 **          D. Rainer (RUS)                                               **
 **          S. Leseduarte (Vircinity)                                     **
 **                                                                        **
\**************************************************************************/

#include "Sample.h"
#include "api/coFeedback.h"

const int DefaultSize = 128;

int ExpandSetList(const coDistributedObject *const *objects, int h_many,
                  const char **type, std::vector<const coDistributedObject *> &out_array);

Sample::Sample(int argc, char *argv[])
    : coModule(argc, argv, "Sample data on points, unstructured and uniform grids to a uniform grid")
{

    const char *outsideChoice[] = { "MAX_FLT", "user_defined_fill_value" };
    const char *sizeChoice[] = { "user defined", "8", "16", "32", "64", "128", "256", "512", "1024" };
    const char *algorithmChoices[] = {
        "possible holes",
        "no holes and no expansion",
        "no holes and expansion",
        "accurate and slow"
        /*, "number weights" */
    };
    const char *bounding_boxChoices[] = { "automatic per timestep", "automatic global", "manual" };

    const char *pointSamplingChoices[] = { "linear", "logarithmic", "normalized linear", "normalized logarithmic" };

    // fill value for outside
    outsideChoiceParam = addChoiceParam("outside", "fill value for outside - MAXFLT or number");
    outsideChoiceParam->setValue(2, outsideChoice, 0);

    fillValueParam = addFloatParam("fill_value", "Fill Value if not intersecting");
    fillValueParam->setValue(0.0);

    // choose algorithm
    p_algorithm = addChoiceParam("algorithm", "choose algorithm");
    p_algorithm->setValue(4, algorithmChoices, 2);

    // choose mapping for point sampling
    p_pointSampling = addChoiceParam("point_sampling", "choose mapping for point sampling");
    p_pointSampling->setValue(4, pointSamplingChoices, 2);

    // select dimension in i direction
    iSizeChoiceParam = addChoiceParam("isize", "unigrid size in i direction");
    iSizeChoiceParam->setValue(9, sizeChoice, 0);

    iSizeParam = addInt32Param("user_defined_isize", "user defined i_size");
    iSizeParam->setValue(DefaultSize);

    jSizeChoiceParam = addChoiceParam("jsize", "unigrid size in j direction");
    jSizeChoiceParam->setValue(9, sizeChoice, 0);

    jSizeParam = addInt32Param("user_defined_jsize", "user defined j_size");
    jSizeParam->setValue(DefaultSize);

    kSizeChoiceParam = addChoiceParam("ksize", "unigrid size in k direction");
    kSizeChoiceParam->setValue(9, sizeChoice, 0);

    kSizeParam = addInt32Param("user_defined_ksize", "user defined k_size");
    kSizeParam->setValue(DefaultSize);

    p_bounding_box = addChoiceParam("bounding_box", "bounding box calculation");
    p_bounding_box->setValue(3, bounding_boxChoices, 0); // default to manual

    float boundsIni[3] = { -1.0, -1.0, -1.0 };
    p_P1bound_manual = addFloatVectorParam("P1_bounds", "First point");
    p_P1bound_manual->setValue(3, boundsIni);

    float boundsEnd[3] = { 1.0, 1.0, 1.0 };
    p_P2bound_manual = addFloatVectorParam("P2_bounds", "Second point");
    p_P2bound_manual->setValue(3, boundsEnd);

    // eps to cover numerical problems
    epsParam = addFloatParam("eps", "small value to cover numerical problems");
    epsParam->setValue(0.0);

    // add an input port for 'coDoUnstructuredGrid' objects
    Grid_In_Port = addInputPort("GridIn", "UnstructuredGrid|UniformGrid|RectilinearGrid|StructuredGrid|Points", "Grid input");
    Data_In_Port = addInputPort("DataIn", "Float|Vec3", "Data input");
    Data_In_Port->setRequired(0);
    Reference_Grid_In_Port = addInputPort("ReferenceGridIn", "UniformGrid", "Reference Grid");
    Reference_Grid_In_Port->setRequired(0);

    // add an output port for this type
    Grid_Out_Port = addOutputPort("GridOut", "UniformGrid", "Grid Output Port");
    Data_Out_Port = addOutputPort("DataOut", "Float|Vec3", "Data Output Port");
}

void
Sample::postInst()
{
    //algorithmParam->show();
    outsideChoiceParam->show();
    p_algorithm->show();
    p_pointSampling->show();
    p_bounding_box->show();
    iSizeChoiceParam->show();
    jSizeChoiceParam->show();
    kSizeChoiceParam->show();
    //   epsParam->show();

    param(outsideChoiceParam->getName(), false);
    param(iSizeChoiceParam->getName(), false);
    param(jSizeChoiceParam->getName(), false);
    param(kSizeChoiceParam->getName(), false);
    param(p_bounding_box->getName(), false);
    param(p_pointSampling->getName(), false);
    param(p_algorithm->getName(), false);
}

void
Sample::param(const char *name, bool inMapLoading)
{
    if (strcmp(name, outsideChoiceParam->getName()) == 0)
    {

        int outside = outsideChoiceParam->getValue();

        if (outside == Sample::OUTSIDE_MAX_FLT)
            fillValueParam->hide();
        else if (outside == Sample::OUTSIDE_NUMBER)
            fillValueParam->show();
        else
            fprintf(stderr, "unhandled outside value = %d\n", outside);
    }

    else if (strcmp(name, iSizeChoiceParam->getName()) == 0)
    {
        int s = iSizeChoiceParam->getValue() + 1;
        if (s == 1)
            iSizeParam->show();
        else
        {
            iSizeParam->setValue(2 << s);
            iSizeParam->hide();
        }
    }

    else if (strcmp(name, jSizeChoiceParam->getName()) == 0)
    {
        int s = jSizeChoiceParam->getValue() + 1;
        if (s == 1)
            jSizeParam->show();
        else
        {
            jSizeParam->setValue(2 << s);
            jSizeParam->hide();
        }
    }

    else if (strcmp(name, kSizeChoiceParam->getName()) == 0)
    {
        int s = kSizeChoiceParam->getValue() + 1;
        if (s == 1)
            kSizeParam->show();
        else
        {
            kSizeParam->setValue(2 << s);
            kSizeParam->hide();
        }
    }

    else if (strcmp(name, p_bounding_box->getName()) == 0)
    {
        if (p_bounding_box->getValue() == 0 || p_bounding_box->getValue() == 1) // automatic bounding box
        {
            p_P1bound_manual->hide();
            p_P2bound_manual->hide();
        } // manual bounding box
        else
        {
            p_P1bound_manual->show();
            p_P2bound_manual->show();
        }
    }
    else if (strcmp(name, p_algorithm->getName()) == 0)
    {
        if (p_algorithm->getValue() == SAMPLE_ACCURATE)
            epsParam->show();
        else
            epsParam->hide();
    }

    // the rest should not be done while loading a map
    if (inMapLoading)
        return;

    // if the user changes the sizes, it automatically changes to manual
    if (strcmp(name, iSizeParam->getName()) == 0)
        iSizeChoiceParam->setValue(0);
    else if (strcmp(name, jSizeParam->getName()) == 0)
        jSizeChoiceParam->setValue(0);
    else if (strcmp(name, kSizeParam->getName()) == 0)
        kSizeChoiceParam->setValue(0);

    // if the user changes the bbox, it automatically changes to manual
    else if (strcmp(name, p_P1bound_manual->getName()) == 0 || strcmp(name, p_P2bound_manual->getName()) == 0)
        p_bounding_box->setValue(2);
}

int
Sample::compute(const char *)
{
    int x_value, y_value, z_value;
    int s;
    bool nan_flag = false; //TODO: Is this the right default-value?
    float fill = .0; //TODO: Is this the right default-value?
    int outside;

    const char *object_type = NULL;

    unstruct_grid::vecFlag typeFlag = unstruct_grid::DONTKNOW;

    ias grids;
    ias data;
    grids.resize(1);
    data.resize(1);

    eps = epsParam->getValue();
    if (eps < 0.0)
    {
        sendError("eps < 0.0 not allowed - please choose a valid eps");
        return FAIL;
    }

    s = iSizeChoiceParam->getValue() + 1;
    if (s == 1)
    {
        x_value = iSizeParam->getValue();
    }
    else
    {
        x_value = 2 << s;
    }

    s = jSizeChoiceParam->getValue() + 1;
    if (s == 1)
    {
        y_value = jSizeParam->getValue();
    }
    else
    {
        y_value = 2 << s;
    }

    s = kSizeChoiceParam->getValue() + 1;
    if (s == 1)
    {
        z_value = kSizeParam->getValue();
    }
    else
    {
        z_value = 2 << s;
    }

    if (x_value < 2)
    {
        sendWarning("Size in X dimension is less than 2, assuming 2");
        iSizeParam->setValue(2);
        x_value = 2;
    }
    if (y_value < 2)
    {
        sendWarning("Size in Y dimension is less than 2, assuming 2");
        jSizeParam->setValue(2);
        y_value = 2;
    }
    if (z_value < 2)
    {
        sendWarning("Size in Z dimension is less than 2, assuming 2");
        kSizeParam->setValue(2);
        z_value = 2;
    }

    outside = outsideChoiceParam->getValue();
    if (outside == Sample::OUTSIDE_MAX_FLT)
    {
        nan_flag = true;
    }
    else if (outside == Sample::OUTSIDE_NUMBER)
    {

        fill = fillValueParam->getValue();
        nan_flag = false;
    }

    bool isStrGrid = false;
    // here we try to retrieve the data object from the required port
    const coDistributedObject *GridObj = Grid_In_Port->getCurrentObject();
    TimeSteps = 0;
    if (!GridObj)
    {
        sendError("cannot retrieve the grid object");
        return FAIL;
    }
    else if (strcmp(GridObj->getType(), "SETELE") == 0)
    {
        // Time Steps or not...
        const coDoSet *setGrid = (const coDoSet *)(GridObj);
        gridTimeSteps = setGrid->getAllElements(&num_set_ele);
        if (GridObj->getAttribute("TIMESTEP"))
        {
            // We have timesteps
            TimeSteps = 1;
            gridTimeSteps = setGrid->getAllElements(&num_set_ele);
            grids.resize(num_set_ele);
            data.resize(num_set_ele);
            for (int time = 0; time < num_set_ele; ++time)
            {
                if (gridTimeSteps[time]->isType("SETELE"))
                {
                    // extract list from the set gridTimeSteps[time]
                    const coDistributedObject *const *elemsInGridTimeSteps;
                    int numInGridTimeSteps;
                    elemsInGridTimeSteps = ((coDoSet *)(gridTimeSteps[time]))->getAllElements(&numInGridTimeSteps);
                    //ExpandSetList (elemsInGridTimeSteps, numInGridTimeSteps, "UNSGRD", grids[time]);
                    ExpandSetList(elemsInGridTimeSteps, numInGridTimeSteps, &object_type, grids[time]);
                }
                else if (!object_type)
                {
                    if (gridTimeSteps[time]->isType("POINTS")
                        || gridTimeSteps[time]->isType("UNSGRD")
                        || gridTimeSteps[time]->isType("UNIGRD")
                        || gridTimeSteps[time]->isType("RCTGRD")
                        || gridTimeSteps[time]->isType("STRGRD"))
                    {
                        object_type = gridTimeSteps[time]->getType();
                        grids[time].resize(1);
                        grids[time][0] = gridTimeSteps[time];
                    }
                    else
                    {
                        // Strange object in the grid object
                        sendError("Objects in the grid object have to be sets, unstructured grids or points");
                        return FAIL;
                    }
                }
                else if (strcmp(gridTimeSteps[time]->getType(), object_type) == 0)
                {
                    grids[time].resize(1);
                    grids[time][0] = gridTimeSteps[time];
                }
                else
                {
                    // Strange object in the grid object
                    sendError("Objects in the grid object have to be sets, unstructured grids or points");
                    return FAIL;
                }
            }
            if (object_type && (!strcmp(object_type, "UNIGRD") || !strcmp(object_type, "RCTGRD") || !strcmp(object_type, "STRGRD")))
            {
                isStrGrid = true;
            }
        }
        else
        {
            // Static grid
            // We may set up grids
            num_blocks = ExpandSetList(gridTimeSteps, num_set_ele, &object_type, grids[0]);
            if (!object_type || (strcmp(object_type, "UNSGRD") != 0
                                 && strcmp(object_type, "POINTS") != 0
                                 && strcmp(object_type, "UNIGRD") != 0
                                 && strcmp(object_type, "RCTGRD") != 0
                                 && strcmp(object_type, "STRGRD") != 0))
            {
                // Strange object in the grid object
                sendError("Objects in the grid object have to be sets, unstructured grids or points");
                return FAIL;
            }
        }
    }
    else if (strcmp(GridObj->getType(), "UNSGRD") == 0)
    {
        object_type = "UNSGRD";
        num_blocks = 1;
        grids[0].push_back(GridObj);
    }
    else if (strcmp(GridObj->getType(), "UNIGRD") == 0)
    {
        object_type = "UNIGRD";
        num_blocks = 1;
        grids[0].push_back(GridObj);
        isStrGrid = true;
    }
    else if (strcmp(GridObj->getType(), "RCTGRD") == 0)
    {
        object_type = "RCTGRD";
        num_blocks = 1;
        grids[0].push_back(GridObj);
        isStrGrid = true;
    }
    else if (strcmp(GridObj->getType(), "STRGRD") == 0)
    {
        object_type = "STRGRD";
        num_blocks = 1;
        grids[0].push_back(GridObj);
        isStrGrid = true;
    }
    else if (strcmp(GridObj->getType(), "POINTS") == 0)
    {
        object_type = "POINTS";
        num_blocks = 1;
        grids[0].push_back(GridObj);
    }
    else
    {
        sendError("Sorry, only objects of type unstructured grid or points are supported");
        return FAIL;
    }

    fprintf(stderr, "isStrGrid=%d, type=%s\n", int(isStrGrid), object_type);

    if (!object_type)
    {
        sendError("Sorry, only objects of type unstructured grid or points are supported");
        return FAIL;
    }

    if (strcmp(object_type, "POINTS") == 0)
    {
        typeFlag = unstruct_grid::POINT;
    }

    // here we try to retrieve the data object from the port
    const coDistributedObject *DataObj = Data_In_Port->getCurrentObject();
    if (!DataObj)
    {
        cerr << "data: NULL" << endl;
        if (typeFlag != unstruct_grid::POINT)
        {
            sendError("cannot retrieve data object");
            return FAIL;
        }
    }
    else if (strcmp(DataObj->getType(), "SETELE") == 0)
    {
        // Time Steps or not...: that is decided according to the grid.
        if (strcmp(GridObj->getType(), "SETELE") != 0)
        {
            sendError("Grid object is a set, but not the data");
            return FAIL;
        }
        int data_num_set_ele;
        int data_num_blocks;
        coDoSet *setData = (coDoSet *)(DataObj);
        dataTimeSteps = setData->getAllElements(&data_num_set_ele);
        if (TimeSteps)
        {
            // We have timesteps
            if (data_num_set_ele != num_set_ele)
            {
                sendError("Number of time steps in grid does not match data set");
                return FAIL;
            }
            for (int time = 0; time < num_set_ele; ++time)
            {
                if (strcmp(dataTimeSteps[time]->getType(), "SETELE") == 0)
                {
                    // extract list from the set gridTimeSteps[time]
                    int numInDataTimeSteps;
                    const coDistributedObject *const *elemsInDataTimeSteps = ((coDoSet *)(dataTimeSteps[time]))->getAllElements(&numInDataTimeSteps);
                    const char *type = NULL;
                    int num_data = ExpandSetList(elemsInDataTimeSteps, numInDataTimeSteps, &type, data[time]);
                    if (!type)
                    {
                        sendError("Invalid data");
                        return FAIL;
                    }

                    if (num_data != grids[time].size())
                    {
                        sendError("Cannot match grids and data sets -- data must be node based");
                        return FAIL;
                    }

                    // try first vector
                    if (strcmp(type, "USTVDT") == 0)
                    {
                        if (typeFlag == unstruct_grid::DONTKNOW)
                        {
                            typeFlag = unstruct_grid::VECTOR;
                        }
                        else
                        {
                            sendError("Cannot decide between vector, scalar, and point");
                            return FAIL;
                        }
                    }
                    // then try scalar
                    else if (strcmp(type, "USTSDT") == 0)
                    {
                        if (typeFlag == unstruct_grid::DONTKNOW)
                        {
                            typeFlag = unstruct_grid::SCALAR;
                        }
                        else
                        {
                            sendError("Cannot decide between vector, scalar, and point");
                            return FAIL;
                        }
                    }
                    else if (strcmp(type, "POINTS") == 0)
                    {
                        if (typeFlag == unstruct_grid::DONTKNOW)
                        {
                            typeFlag = unstruct_grid::POINT;
                        }
                        else
                        {
                            sendError("Cannot decide between vector, scalar, and point");
                            return FAIL;
                        }
                    }
                    else
                    {
                        sendError("Invalid data type");
                        return FAIL;
                    }
                }
                else if (strcmp(dataTimeSteps[time]->getType(), "USTSDT") == 0)
                {
                    data[time].resize(1);
                    data[time][0] = dataTimeSteps[time];
                    if (typeFlag == unstruct_grid::DONTKNOW)
                    {
                        typeFlag = unstruct_grid::SCALAR;
                    }
                    else if (typeFlag == unstruct_grid::VECTOR)
                    {
                        sendError("Cannot decide between vector and scalar");
                        return FAIL;
                    }
                }
                else if (strcmp(dataTimeSteps[time]->getType(), "USTVDT") == 0)
                {
                    data[time].resize(1);
                    data[time][0] = dataTimeSteps[time];
                    if (typeFlag == unstruct_grid::DONTKNOW)
                    {
                        typeFlag = unstruct_grid::VECTOR;
                    }
                    else if (typeFlag != unstruct_grid::VECTOR)
                    {
                        sendError("Cannot decide between vector and scalar");
                        return FAIL;
                    }
                }
                else if (strcmp(dataTimeSteps[time]->getType(), "POINTS") == 0)
                {
                    data[time].resize(1);
                    data[time][0] = dataTimeSteps[time];
                    if (typeFlag == unstruct_grid::DONTKNOW)
                    {
                        typeFlag = unstruct_grid::POINT;
                    }
                    else if (typeFlag != unstruct_grid::POINT)
                    {
                        sendError("Cannot decide between vector, scalar and point");
                        return FAIL;
                    }
                }
                else
                {
                    sendError("Object of incorrect type in data set");
                    return FAIL;
                }
            }
        } // static set... set up data
        else
        {
            const char *type = NULL;
            data_num_blocks = ExpandSetList(dataTimeSteps, data_num_set_ele, &type, data[0]);
            if (!type)
            {
                sendError("Invalid data");
                return FAIL;
            }

            if (data_num_blocks != num_blocks)
            {
                sendError("Grid and data sets do not match for the accepted types");
                return FAIL;
            }

            if (typeFlag == unstruct_grid::DONTKNOW)
            {
                if (strcmp(type, "USTSDT") == 0)
                {
                    typeFlag = unstruct_grid::SCALAR;
                }
                else if (strcmp(type, "USTVDT") == 0)
                {
                    typeFlag = unstruct_grid::VECTOR;
                }
                else if (strcmp(type, "POINTS") == 0)
                {
                    typeFlag = unstruct_grid::POINT;
                }
                else
                {
                    sendError("Invalid data type");
                    return FAIL;
                }
            }
        }
    }
    else if (strcmp(DataObj->getType(), "USTSDT") == 0)
    {
        data[0].resize(1);
        data[0][0] = (DataObj);
        if (typeFlag == unstruct_grid::DONTKNOW)
            typeFlag = unstruct_grid::SCALAR;
    }
    else if (strcmp(DataObj->getType(), "USTVDT") == 0)
    {
        data[0].resize(1);
        data[0][0] = (DataObj);
        if (typeFlag == unstruct_grid::DONTKNOW)
            typeFlag = unstruct_grid::VECTOR;
    }
    else if (strcmp(DataObj->getType(), "POINTS") == 0)
    {
        data[0].resize(1);
        data[0][0] = (DataObj);
        if (typeFlag == unstruct_grid::DONTKNOW)
            typeFlag = unstruct_grid::POINT;
    }
    else
    {
        sendError("Only scalar or vector unstructured data, points or sets thereof are supported");
        return FAIL;
    }

    coDoUniformGrid **destGrids = NULL;
    const coDistributedObject *refGrid = Reference_Grid_In_Port->getCurrentObject();
    if (refGrid)
    {
        if (TimeSteps)
        {
            destGrids = new coDoUniformGrid *[num_set_ele];
        }
        else
        {
            destGrids = new coDoUniformGrid *[1];
        }

        if (!strcmp(refGrid->getType(), "SETELE"))
        {
            coDoSet *set = (coDoSet *)refGrid;
            int refgrid_num_set_ele;
            const coDistributedObject *const *refGridTimeSteps = set->getAllElements(&refgrid_num_set_ele);
            if (TimeSteps)
            {
                if (refgrid_num_set_ele != num_set_ele)
                {
                    sendError("Number of time steps in reference grid does not match data set");
                    return FAIL;
                }
                for (int i = 0; i < num_set_ele; i++)
                {
                    if (strcmp(refGridTimeSteps[i]->getType(), "UNIGRD"))
                    {
                        sendError("Reference grid must be a uniform grid or a set thereof");
                        return FAIL;
                    }
                    coDoUniformGrid *ugr = (coDoUniformGrid *)refGridTimeSteps[i];
                    destGrids[i] = ugr;
                }
            }
        }
        else
        {
            if (strcmp(refGrid->getType(), "UNIGRD"))
            {
                sendError("Reference grid must be a uniform grid or a set thereof");
                return FAIL;
            }

            coDoUniformGrid *ugr = (coDoUniformGrid *)refGrid;
            if (TimeSteps)
            {
                for (int i = 0; i < num_set_ele; i++)
                {
                    destGrids[i] = ugr;
                }
            }
            else
            {
                destGrids[0] = ugr;
            }
        }
    }

    if (Diagnose(grids, data, typeFlag, isStrGrid) < 0)
    {
        sendError("Grid and data are not compatible or an undefined problem occurred");
        return FAIL;
    }
    // check struncture of grids, sdata and vdata
    // Diagnose

    // check dimension of grid and data
    /*
      int numElems, numIndices, numCoords, numData;
      ((coDoUnstructuredGrid *)GridObj)->getGridSize(&numElems, &numIndices, &numCoords);
      numData = ((coDoFloat*)DataObj)->getNumPoints();
      if (numData != numCoords)
      {
      if (numData == numElems)
      {

      sendError("Sample supports only vertex based data - you seem to have cell based data");
      return FAIL;
      }
      else
      {
      char str[500];
      sprintf (str, "Dimension of data doesn't match dimension of grid - found %d coordinates and %d data", numCoords, numData);
      sendError(str);
      return FAIL;
      }
      }
    */

    // now we create an object for the output port: get the name and make the Obj
    const char *Grid_outObjName = Grid_Out_Port->getObjName();
    const char *Data_outObjName = Data_Out_Port->getObjName();
    char time_grid_name[64];
    char time_data_name[64];

    strcpy(time_grid_name, Grid_outObjName);
    strcpy(time_data_name, Data_outObjName);

    coDistributedObject **usg = new coDistributedObject *[grids.size() + 1];
    usg[grids.size()] = 0;

    coDistributedObject **str = new coDistributedObject *[grids.size() + 1];
    str[grids.size()] = 0;
    int strelem;
    for (strelem = 0; strelem < grids.size(); ++strelem)
    {
        str[strelem] = NULL;
    }

    float gmin[3] = { FLT_MAX, FLT_MAX, FLT_MAX },
          gmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    if (!destGrids && p_bounding_box->getValue() == 1) // automatic globally
    {
        for (int time = 0; time < grids.size(); time++)
        {
            unstruct_grid *calc_grid = new unstruct_grid(grids[time],
                                                         typeFlag, isStrGrid);
            calc_grid->automaticBoundBox();
            const float *min, *max;
            calc_grid->getMinMax(min, max);
            for (int i = 0; i < 3; i++)
            {
                if (min[i] < gmin[i])
                    gmin[i] = min[i];
                if (max[i] > gmax[i])
                    gmax[i] = max[i];
            }
            delete calc_grid;
        }
    }

    for (int time = 0; time < grids.size(); time++)
    {
        if (TimeSteps)
        {
            sprintf(time_grid_name, "%s_%d", Grid_outObjName, time);
            sprintf(time_data_name, "%s_%d", Data_outObjName, time);
        }
        //        unstruct_grid *calc_grid=new unstruct_grid((coDoUnstructuredGrid*)GridObj);
        //int size = grids[time].size ();
        //@@@        int num_compressed = grids[time].num_compressed();
        unstruct_grid *calc_grid = new unstruct_grid(grids[time], typeFlag, isStrGrid);
        if (destGrids)
        {
            float min[3], max[3];
            destGrids[time]->getMinMax(&min[0], &max[0],
                                       &min[1], &max[1],
                                       &min[2], &max[2]);
            p_P1bound_manual->setValue(min[0], min[1], min[2]);
            p_P2bound_manual->setValue(max[0], max[1], max[2]);
            calc_grid->manualBoundBox(min[0], min[1], min[2],
                                      max[0], max[1], max[2]);

            destGrids[time]->getGridSize(&x_value, &y_value, &z_value);
        }
        else if (p_bounding_box->getValue() == 0) // automatic per timestep
        {
            calc_grid->automaticBoundBox();
            const float *min, *max;
            calc_grid->getMinMax(min, max);
            p_P1bound_manual->setValue(min[0], min[1], min[2]);
            p_P2bound_manual->setValue(max[0], max[1], max[2]);
        }
        else if (p_bounding_box->getValue() == 1) // automatic globally
        {
            calc_grid->manualBoundBox(gmin[0], gmin[1], gmin[2],
                                      gmax[0], gmax[1], gmax[2]);
        }
        else if (p_bounding_box->getValue() == 2) // manual
        {
            calc_grid->manualBoundBox(p_P1bound_manual->getValue(0),
                                      p_P1bound_manual->getValue(1),
                                      p_P1bound_manual->getValue(2),
                                      p_P2bound_manual->getValue(0),
                                      p_P2bound_manual->getValue(1),
                                      p_P2bound_manual->getValue(2));
        }
        const coDistributedObject **ndata = NULL;
        if (data.size() > time && data[time].size() > 0)
            ndata = &data[time][0];

        calc_grid->set_value(nan_flag, fill);

        // create data on strcutured grid
        if (grids[time].size() <= 0)
        {
            x_value = y_value = z_value = 1;
        }

        if ((typeFlag == unstruct_grid::SCALAR || typeFlag == unstruct_grid::VECTOR) && isStrGrid)
        {
            calc_grid->sample_structured(ndata,
                                         time_grid_name, &usg[time],
                                         time_data_name, &str[time], x_value, y_value, z_value);
        }
        else if (typeFlag == unstruct_grid::POINT)
        {
            calc_grid->sample_points(ndata,
                                     time_grid_name, &usg[time],
                                     time_data_name, &str[time], x_value, y_value, z_value,
                                     p_pointSampling->getValue());
        }
        else
        {
            switch (p_algorithm->getValue())
            {
            case SAMPLE_ACCURATE:
                // sl: this algorithm (the "old" one)
                //     is not as exact as it promises.
                //     The problem is that internally it calculates
                //     the indices around the element nodes approximating
                //     by default... As a result of this, strange
                //     effects appear sometimes near the borders.
                //     For this reason, and because it is very slow
                //     (that is why it is not worth the effort correcting
                //     the inaccuracies near the borders),
                //     the other new algorithms are recommended,
                //     especially SAMPLE_NO_HOLES_BETTER.
                calc_grid->sample_accu(ndata, /*&sdata[time][0] */
                                       time_grid_name, &usg[time],
                                       time_data_name, &str[time], x_value, y_value, z_value, eps);
                break;
            case SAMPLE_HOLES:
                calc_grid->sample_holes(ndata,
                                        time_grid_name, &usg[time], time_data_name, &str[time], x_value, y_value, z_value);
                break;
            case SAMPLE_NO_HOLES:
                calc_grid->sample_no_holes(ndata,
                                           time_grid_name, &usg[time],
                                           time_data_name, &str[time], x_value, y_value, z_value, 0);
                break;
            case SAMPLE_NO_HOLES_BETTER:
                calc_grid->sample_no_holes(ndata,
                                           time_grid_name, &usg[time],
                                           time_data_name, &str[time], x_value, y_value, z_value, 1);
                break;
            }
        }

        // attach the name of the input data object to the output grid object
        if ((typeFlag == unstruct_grid::SCALAR || typeFlag == unstruct_grid::VECTOR) && isStrGrid)
        {
            const char **names, **values;
            int n = grids[time][0]->getAllAttributes(&names, &values);
            for (int i = 0; i < n; i++)
            {
                if (strcmp(names[i], "Transformation") != 0)
                {
                    usg[time]->addAttribute(names[i], values[i]);
                }
            }
        }
        else
        {
            usg[time]->copyAllAttributes(grids[time][0]);
        }
        if (ndata)
        {
            usg[time]->addAttribute("DataObjectName", (*ndata)->getName());

            str[time]->copyAllAttributes((*ndata) /*sdata[time][0] */);
        }

        delete calc_grid;
    }

    // tell the output port that this is his object
    if (TimeSteps == 0)
    {
        Grid_Out_Port->setCurrentObject(usg[0]);
        Data_Out_Port->setCurrentObject(str[0]);
    }
    else
    {
        coDoSet *gridOutTime = new coDoSet(Grid_outObjName, usg);
        sprintf(time_grid_name, "1 %d", (int)grids.size());
        gridOutTime->addAttribute("TIMESTEP", time_grid_name);
        Grid_Out_Port->setCurrentObject(gridOutTime);
        //      if(typeFlag == unstruct_grid::SCALAR){
        coDoSet *dataOutTime = new coDoSet(Data_outObjName, str);
        dataOutTime->addAttribute("TIMESTEP", time_grid_name);
        Data_Out_Port->setCurrentObject(dataOutTime);
        //      }
    }

    // Add INTERACTOR attribute to the grid
    coFeedback feedback("Sample");
    feedback.addPara(iSizeParam);
    feedback.addPara(jSizeParam);
    feedback.addPara(kSizeParam);
    feedback.addPara(p_P1bound_manual);
    feedback.addPara(p_P2bound_manual);

    coDistributedObject *obj = Grid_Out_Port->getCurrentObject();
    if (obj)
        feedback.apply(obj);

    return SUCCESS;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Quit callback: as the name tells...
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void
Sample::quit()
{
    //cerr << "Ende" << endl;
}

int
ExpandSetList(const coDistributedObject *const *objects, int h_many, const char **type, std::vector<const coDistributedObject *> &out_array)
{
    int i, j;
    int count = 0;
    *type = NULL;
    for (i = 0; i < h_many; ++i)
    {
        const char *obj_type = objects[i]->getType();
        if (*type && strcmp(obj_type, *type) == 0)
        {
            out_array.push_back(objects[i]);
            count++;
        }
        else if (strcmp(obj_type, "SETELE") == 0)
        {
            std::vector<const coDistributedObject *> partial;
            const coDistributedObject *const *in_set_elements;
            int in_set_len;
            in_set_elements = ((coDoSet *)(objects[i]))->getAllElements(&in_set_len);
            int add = ExpandSetList(in_set_elements, in_set_len, type, partial);
            for (j = 0; j < add; ++j)
            {
                out_array.push_back(partial[j]);
            }
            count += add;
        }
        else
        {
            if (*type)
                fprintf(stderr, "already have %s, now got %s\n", *type, objects[i]->getType());
            *type = objects[i]->getType();
            out_array.push_back(objects[i]);
            count++;
        }
    }
    return count;
}

int
Sample::Diagnose(ias &grids, ias &data, unstruct_grid::vecFlag typeFlag, bool isStrGrid)
{
    if (typeFlag != unstruct_grid::SCALAR
        && typeFlag != unstruct_grid::VECTOR
        && typeFlag != unstruct_grid::POINT)
    {
        sendInfo("Could not decide between vector and scalar");
        return -1;
    }

    if (typeFlag == unstruct_grid::POINT || isStrGrid)
    {
        return 0;
    }

    if (grids.size() != data.size())
    {
        return -1;
    }

    for (int time = 0; time < grids.size(); ++time)
    {
        for (int block = 0; block < grids[time].size();)
        {
            int num_el = 0, num_conn = 0, num_coor_grid = 0, num_coor_data = 0;
            coDoUnstructuredGrid *p_grid = (coDoUnstructuredGrid *)(grids[time][block]);
            p_grid->getGridSize(&num_el, &num_conn, &num_coor_grid);

            if (typeFlag == unstruct_grid::SCALAR)
            {
                coDoFloat *p_sdata = (coDoFloat *)(data[time][block]);
                num_coor_data = p_sdata->getNumPoints();
            }
            else if (typeFlag == unstruct_grid::VECTOR)
            {
                coDoVec3 *p_vdata = (coDoVec3 *)(data[time][block]);
                num_coor_data = p_vdata->getNumPoints();
            }

            if (num_coor_data == 0 && grids[time].size() > 1)
            {
                // we have to "forget" that the corresponding grid exists...
                compress(grids[time], block);
                compress(data[time], block);
                continue;
            }

            if (num_coor_grid != num_coor_data)
            {
                sendInfo("Detected grid and data object with incompatible size: data has to be node-based");
                return -1;
            }
            ++block;
        }
    }
    return 0;
}

MODULE_MAIN(Interpolator, Sample)
