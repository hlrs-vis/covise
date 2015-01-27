/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   (C)2002 VirCinity    **
 **                                                                        **
 ** Description: Read module Cadmould data format                          **
 **                                                                        **
\**************************************************************************/

#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoUnstructuredGrid.h>
#include "ReadCadmould.h"
#include "CadmouldGrid.h"
#include "CadmouldData.h"
#include <float.h>

ReadCadmould::ReadCadmould(int argc, char *argv[])
    : coModule(argc, argv, "Read Cadmould")
{
    createParam();
    d_grid = NULL;
    d_numDataSets = 0;
    retOpenFile = -1;
    fillChoice = 0;
#ifdef BYTESWAP
    byteswap_ = false;
#else
    byteswap_ = true;
#endif
}

ReadCadmould::~ReadCadmould()
{
    for (int i = 0; i < d_numDataSets; i++)
        delete d_data[i];
}

coDoSet *ReadCadmould::readField(const std::string &objName, CadmouldData *data,
                                 int field, int step)
{
    int numGroups = d_grid->numGroups();

    // buffer for object names
    char partName[1024];

    // get data into buffer
    float *fglobal = NULL;
    int *iglobal = NULL;
    switch (data->getFieldType(field))
    {
    case CadmouldData::SCALAR_FLOAT:
        fglobal = new float[data->numVert()];

        if (!fglobal)
        {
            sendError("Check if byte-swapping is correctly (de)activated");
            return NULL;
        }
        break;
    case CadmouldData::SCALAR_INT:
        iglobal = new int[data->numVert()];
        if (!iglobal)
        {
            sendError("Check if byte-swapping is correctly (de)activated");
            return NULL;
        }
        break;
    default:
        sendError("Forbidden option in ReadCadmould::readField");
        return NULL; // this is an error
    }
    void *global = (fglobal) ? static_cast<void *>(fglobal) : static_cast<void *>(iglobal);
    data->readField(field, step, global);

    // prepare for set-building
    coDistributedObject **setElem = new coDistributedObject *[numGroups + 1];
    setElem[numGroups] = NULL;

    // loop over all groups
    for (int grp = 0; grp < numGroups; ++grp)
    {
        // get group sizes
        int numElem, numConn, numVert;
        d_grid->gridSizes(grp, numElem, numConn, numVert);
        const int *map = d_grid->globalVertex(grp);

        // create object
        sprintf(partName, "%s_%d", objName.c_str(), grp);
        if (fglobal)
        {
            coDoFloat *s3d = new coDoFloat(partName, numVert);
            float *val;
            s3d->getAddress(&val);
            s3d->addAttribute("SPECIES", data->getName(field));

            for (int i = 0; i < numVert; i++)
            {
                int index = map[i];
                if (index < data->numVert() && index >= 0)
                {
                    val[i] = fglobal[index];
                }
                else // point with no value
                {
                    val[i] = -1.0;
                }
            }

            for (int i = 0; i < numVert; i++)
            {
                if (val[i] == -1.0)
                {
                    val[i] = FLT_MAX;
                }
            }

            setElem[grp] = s3d;
        }
        else
        {
            coDoIntArr *i3d = new coDoIntArr(partName, 1, &numVert);
            int *val;
            i3d->getAddress(&val);
            i3d->addAttribute("SPECIES", data->getName(field));

            int i, index;
            for (i = 0; i < numVert; i++)
            {
                index = map[i];
                if (index < data->numVert())
                {
                    val[i] = iglobal[index];
                }
                else // point with no value
                {
                    val[i] = -1;
                }
            }
            setElem[grp] = i3d;
        }
    }
    if (fglobal)
    {
        delete[] fglobal;
    }
    else
    {
        delete[] iglobal;
    }

    coDoSet *set = new coDoSet(objName, setElem);
    set->addAttribute("NO_DATA_COLOR", p_no_data_color->getValue());

    return set;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void
ReadCadmould::readData(coOutputPort *port, int useDataSet, int useField)
{
    // check if we are reading the variable describing filling time
    // and remember port
    if (useDataSet == fillTimeDataSets && useField == fillField)
    {
        whereisFillTime = port;
    }

    // fprintf(stderr,"Reading data set %d, field %d\n",useDataSet,useField);
    // read the data - either timesteps or not

    int numTimesteps = d_data[useDataSet]->numTimeSteps();
    const char *objName = port->getObjName();

    if (numTimesteps == 0)
    {
        port->setCurrentObject(readField(objName, d_data[useDataSet], useField, 0));
    }
    else
    {

        // buffer for set elements
        coDistributedObject **setElem = new coDistributedObject *[numTimesteps + 1];
        setElem[numTimesteps] = NULL;

        // buffer for object names
        char partName[1024];
        for (int i = 0; i < numTimesteps; i++)
        {
            sprintf(partName, "%s_%d", objName, i);
            setElem[i] = readField(partName, d_data[useDataSet], useField, i);
        }

        coDoSet *set = new coDoSet(objName, setElem);
        sprintf(partName, "0 %d", numTimesteps);
        set->addAttribute("TIMESTEP", partName);
        set->addAttribute("NO_DATA_COLOR", p_no_data_color->getValue());
        port->setCurrentObject(set);

        // we need a multiplied grid at the output port - but only once
        if (NULL == p_stepMesh->getCurrentObject())
        {
            coDistributedObject *mesh = p_mesh->getCurrentObject();
            objName = p_stepMesh->getObjName();
            for (int i = 0; i < numTimesteps; i++)
            {
                sprintf(partName, "%s_%d", objName, i);
                setElem[i] = mesh;
                mesh->incRefCount();
            }
            coDoSet *set = new coDoSet(objName, setElem);
            sprintf(partName, "0 %d", numTimesteps);
            set->addAttribute("TIMESTEP", partName);
            p_stepMesh->setCurrentObject(set);
        }
        delete[] setElem;
    }
}

void ReadCadmould::readData(coOutputPort *port, int choice)
{
    // find correct data set + field
    int useDataSet = dataLoc[choice].datasetNo;
    int useField = dataLoc[choice].fieldNo;

    readData(port, useDataSet, useField);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ReadCadmould::readGrid()
{
    /// read the basic grid - always needed
    int numGroups = d_grid->numGroups();

    /// prepare a set
    coDistributedObject **setElem = new coDistributedObject *[numGroups + 1];
    setElem[numGroups] = NULL;

    /// Base name and buffer for set names
    const char *basename = p_mesh->getObjName();
    char buffer[1024];

    /// loop over groups
    for (int grp = 0; grp < numGroups; ++grp)
    {

        // prepare object name
        sprintf(buffer, "%s_%d", basename, grp);

        // get grid sizes and create object for it
        int numElem, numConn, numVert;
        d_grid->gridSizes(grp, numElem, numConn, numVert);
        coDoUnstructuredGrid *usg
            = new coDoUnstructuredGrid(buffer, numElem, numConn, numVert, 1);

        // get table pointers and have grid object filling it
        int *elemList, *typeList, *connList;
        float *x, *y, *z;
        usg->getAddresses(&elemList, &connList, &x, &y, &z);
        usg->getTypeList(&typeList);
        d_grid->copyTables(grp, elemList, typeList, connList, x, y, z);

        // attach part-ID
        sprintf(buffer, "%d", d_grid->getGroupID(grp));
        usg->addAttribute("PART", buffer);

        setElem[grp] = usg;
    }

    coDoSet *set = new coDoSet(basename, setElem);

    for (int grp = 0; grp < numGroups; ++grp)
    {
        delete setElem[grp];
    }
    delete[] setElem;

    p_mesh->setCurrentObject(set);

    // we add the transient mesh when reading transient data
    p_stepMesh->setCurrentObject(NULL);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ReadCadmould::readThick()
{
    /// read the basic grid - always needed
    int numGroups = d_grid->numGroups();

    /// prepare a set
    coDistributedObject **setElem = new coDistributedObject *[numGroups + 1];
    setElem[numGroups] = NULL;

    /// Base name and buffer for set names
    const char *basename = p_thick->getObjName();
    char buffer[1024];

    /// loop over groups
    for (int grp = 0; grp < numGroups; ++grp)
    {
        // prepare object name
        sprintf(buffer, "%s_%d", basename, grp);

        // get grid sizes and create object for it
        int numElem, numConn, numVert;
        d_grid->gridSizes(grp, numElem, numConn, numVert);
        coDoFloat *scal
            = new coDoFloat(buffer, numElem);

        // get table pointers and have grid object filling it
        float *thick;
        scal->getAddress(&thick);
        d_grid->copyThickness(grp, thick);

        // attach part-ID
        sprintf(buffer, "%d", d_grid->getGroupID(grp));
        scal->addAttribute("PART", buffer);

        setElem[grp] = scal;
    }

    coDoSet *set = new coDoSet(basename, setElem);
    for (int grp = 0; grp < numGroups; ++grp)
    {
        delete setElem[grp];
    }
    delete[] setElem;
    p_thick->setCurrentObject(set);
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadCadmould::compute(const char *)
{
    fillChoice = p_fillField->getValue();
    // pointer where we may look for filling time
    whereisFillTime = NULL;

    // maybe loadad from map - no grid read so far
    if (!d_grid)
    {
        openFiles();
        if (retOpenFile == -1) // apparently incorrect byte-swapping
        {
            //         FlipState(p_byteswap);
            byteswap_ = !byteswap_;
            openFiles();
        }
        // incorrect byte-swapping option
        if (retOpenFile < 0)
        {
            sendError("Information from mesh and filling files are found to be incompatible");
            return STOP_PIPELINE;
        }
    }

    // now we must have one - otherwise die silently, openFiles cries itself
    if (!d_grid)
        return STOP_PIPELINE;

    // read basic grid - save in output port
    readGrid();
    readThick();

    // get filling params
    fillingParams();

    // read data fields at the ports
    for (int i = 0; i < NUM_PORTS; i++)
    {
        int choice = p_choice[i]->getValue();
        if (choice > 0)
            readData(p_data[i], choice);
    }

    return fillingAnimation();
}

void
ReadCadmould::fillingParams()
{
    if (d_type == TYPE_CAR)
    {
        if (fillChoice == 0) //no choice/automatic
        {
            fillChoice = 1;
        }
        fillTimeDataSets = dataLoc[fillChoice].datasetNo;
        fillField = dataLoc[fillChoice].fieldNo;
    }
}

int
ReadCadmould::fillingAnimation()
{
    // read the data - either timesteps or not
    int numTimesteps = p_no_time_steps->getValue();
    if (numTimesteps <= 1)
    {
        sendInfo("If you wish a filling animation choose a number of time steps greater than 1");
        return STOP_PIPELINE;
    }

    // check if the filling time has already been read in
    coDistributedObject *fillTime = NULL;
    if (whereisFillTime && whereisFillTime->getCurrentObject())
    {
        fillTime = whereisFillTime->getCurrentObject();
    }
    else // if not read it now
    {
        std::string objName(p_fillData->getObjName());
        objName += "_Auxiliar";
        // delete fillTime in this case
        if (fillTimeDataSets >= 0)
        {
            fillTime = readField(objName, d_data[fillTimeDataSets], fillField, 0);
        }
    }

    // warn if there has been any problem
    if (!fillTime || !fillTime->objectOk())
    {
        sendWarning("fillingAnimation: could not read fill time information");
        return STOP_PIPELINE;
    }

    float min = FLT_MAX;
    float max = -FLT_MAX;
    getMinMax(fillTime, min, max); //well, in fact only max is needed...

    // create time steps
    coDistributedObject **fillStepFilm = new coDistributedObject *[numTimesteps + 1];
    fillStepFilm[numTimesteps] = NULL;
    std::string objName(p_fillData->getObjName());
    for (int time = 0; time < numTimesteps; ++time)
    {
        // calculate real time
        std::string stepName(objName);
        char buf[64];
        sprintf(buf, "_%d", time);
        stepName += buf;
        float realtime = min + ((max - min) * time) / (numTimesteps - 1);
        coDistributedObject *ogrid = p_mesh->getCurrentObject();
        fillStepFilm[time] = FillStep(stepName, realtime, max, ogrid, fillTime);
    }

    // deletions for a set
    coDoSet *fillingProcess = new coDoSet(objName, fillStepFilm);
    for (int time = 0; time < numTimesteps; ++time)
    {
        delete fillStepFilm[time];
    }
    delete[] fillStepFilm;

    // set object and set time attribute
    char buf[64];
    sprintf(buf, "%d %d", 1, numTimesteps);
    fillingProcess->addAttribute("TIMESTEP", buf);
    fillingProcess->addAttribute("SPECIES", "Animated Filling");
    p_fillData->setCurrentObject(fillingProcess);

    // make static grid transient
    coDistributedObject *ogrid = p_mesh->getCurrentObject();
    coDistributedObject **trgrid = new coDistributedObject *[numTimesteps + 1];
    trgrid[numTimesteps] = NULL;
    for (int time = 0; time < numTimesteps; ++time)
    {
        trgrid[time] = ogrid;
        ogrid->incRefCount();
    }
    coDoSet *trGrid = new coDoSet(p_fillMesh->getObjName(), trgrid);
    trGrid->addAttribute("TIMESTEP", buf);
    p_fillMesh->setCurrentObject(trGrid);

    // delete auxiliary object *fillTime only if necessary:
    // i.e. if it had to be created in this function
    if (!whereisFillTime || !whereisFillTime->getCurrentObject())
    {
        delete fillTime;
    }

    return CONTINUE_PIPELINE;
}

coDistributedObject *
ReadCadmould::FillStep(const std::string &stepName, float realtime, float max,
                       const coDistributedObject *ogrid, const coDistributedObject *fillTime)
{
    if (ogrid->isType("SETELE") && fillTime->isType("SETELE"))
    {
        int no_elems;
        const coDoSet *sgrid = (const coDoSet *)(ogrid);
        const coDistributedObject *const *glist = sgrid->getAllElements(&no_elems);
        const coDoSet *sdata = (const coDoSet *)(fillTime);
        int no_d_elems;
        const coDistributedObject *const *dlist = sdata->getAllElements(&no_d_elems);
        if (no_elems != no_d_elems)
            return NULL;
        const coDistributedObject **outlist = new const coDistributedObject *[no_elems + 1];
        outlist[no_elems] = NULL;
        int elem;
        char buf[64];
        for (elem = 0; elem < no_elems; ++elem)
        {
            std::string elemName(stepName);
            sprintf(buf, "_%d", elem);
            elemName += buf;
            outlist[elem] = FillStep(elemName, realtime, max, glist[elem], dlist[elem]);
        }
        coDoSet *outset = new coDoSet(stepName, outlist);
        for (elem = 0; elem < no_elems; ++elem)
        {
            delete outlist[elem];
        }
        delete[] outlist;
        return outset;
    }
    else if (ogrid->isType("UNSGRD") && fillTime->isType("USTSDT"))
    {
        int no_p, no_e, no_v;
        const coDoUnstructuredGrid *grid = (const coDoUnstructuredGrid *)(ogrid);
        const coDoFloat *sdata = (const coDoFloat *)(fillTime);
        grid->getGridSize(&no_e, &no_v, &no_p);
        int *e_l, *v_l;
        float *x_c, *y_c, *z_c;
        grid->getAddresses(&e_l, &v_l, &x_c, &y_c, &z_c);
        float *sarray;
        sdata->getAddress(&sarray);
        int no_data = sdata->getNumPoints();
        if (no_data != no_p)
        {
            return NULL;
        }
        // AQUI!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // inspect all elements, and check for each one if for all
        // of its nodes are already filled (time > actual time)
        coDoFloat *outdata = new coDoFloat(stepName, no_e);
        outdata->addAttribute("NO_DATA_COLOR", p_no_data_color->getValue());
        float *outarray;
        outdata->getAddress(&outarray);

        for (int elem = 0; elem < no_e; ++elem)
        {
            outarray[elem] = -FLT_MAX;
            if (elem < no_e - 1)
            {
                for (int vertex = e_l[elem]; vertex < e_l[elem + 1]; ++vertex)
                {
                    int node = v_l[vertex];
                    if (sarray[node] > outarray[elem])
                    {
                        outarray[elem] = sarray[node];
                    }
                }
            }
            else
            {
                for (int vertex = e_l[elem]; vertex < no_v; ++vertex)
                {
                    int node = v_l[vertex];
                    if (sarray[node] > outarray[elem])
                    {
                        outarray[elem] = sarray[node];
                    }
                }
            }
            // now check for outarray[elem] > realtime
            if (outarray[elem] > realtime)
            {
                outarray[elem] = FLT_MAX; // statt max
            }
        }
        return outdata;
    }
    return NULL;
}

void
ReadCadmould::getMinMax(const coDistributedObject *fillTime, float &min, float &max)
{
    if (!fillTime)
        return;

    if (const coDoSet *inSet = dynamic_cast<const coDoSet *>(fillTime))
    {
        int no_elements;
        const coDistributedObject *const *inList = inSet->getAllElements(&no_elements);
        for (int elem = 0; elem < no_elements; ++elem)
        {
            getMinMax(inList[elem], min, max);
        }
    }
    else if (const coDoFloat *scalar = dynamic_cast<const coDoFloat *>(fillTime))
    {
        float *uin = NULL;
        scalar->getAddress(&uin);
        int no_points = scalar->getNumPoints();
        for (int point = 0; point < no_points; ++point)
        {
            if (uin[point] < min)
                min = uin[point];
            if (uin[point] > max && uin[point] != FLT_MAX)
                max = uin[point];
        }
    }
    else
    {
        fprintf(stderr, "unhandled type: %s\n", fillTime->getType());
    }
}

MODULE_MAIN(IO, ReadCadmould)
