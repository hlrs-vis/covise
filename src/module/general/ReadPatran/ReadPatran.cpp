/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1998 RUS  **
 **                                                                        **
 ** Description:  Patran Reader                                            **
 **                                                                        **
 **                                                                        **
 **                             (C) Vircinity 2000                         **
 **                                                                        **
 **                                                                        **
 ** Author:  R.Lang, Uwe Woessner, Sasha Cioringa, Sven Kufer              **
 **                                                                        **
 **                                                                        **
 ** Date:  18.05.94  V1.0                                                  **
 ** Date:  05.09.98                                                        **
 ** Date:  08.11.00                                                        **
 ** Date:  05.03.01				 			  **
\**************************************************************************/

#include "ReadPatran.h"
#include <do/coDoData.h>
#include <do/coDoIntArr.h>
#include <do/coDoSet.h>

Patran::Patran(int argc, char *argv[])
    : coModule(argc, argv, "Read Patran Neutral Files")
{
    const char *ChoiseVal[] = {
        "Nodal_Results", "Element_Results",
    };
    strcpy(init_path, "data/nofile");

    //parameters
    p_gridpath = addFileBrowserParam("grid_path", "Neutral File path");
    p_gridpath->setValue(init_path, "*");
    p_displpath = addFileBrowserParam("nodal_displ_force_path", "Nodal Displacement File path");
    p_displpath->setValue(init_path, "*");
    p_nshpath = addFileBrowserParam("nodal_result_path", "Nodal Results File path");
    p_nshpath->setValue(init_path, "*");
    p_elempath = addFileBrowserParam("element_result_path", "Element Results File path");
    p_elempath->setValue(init_path, "*");
    p_option = addChoiceParam("Option", "perNode od perElement data");
    p_option->setValue(2, ChoiseVal, 0);
    p_timesteps = addInt32Param("timesteps", "timesteps");
    p_timesteps->setValue(1);
    p_skip = addInt32Param("skipped_files", "number of skip files for each timestep");
    p_skip->setValue(0);
    p_columns = addInt32Param("nb_columns", "number of column in the result file");
    p_columns->setValue(1);

    //ports

    //p_inPort1->setRequired(0);
    p_outPort1 = addOutputPort("mesh", "UnstructuredGrid", "Mesh output");
    p_outPort2 = addOutputPort("data1", "Vec3", "Vector Data Field 1 output");
    p_outPort3 = addOutputPort("data2", "Float", "Scalar Data Field 1 output");
    p_outPort4 = addOutputPort("type", "IntArr", "IDs");
    //private data
    gridFile = NULL;
    grid_path = NULL;
    nsh_path = NULL;
    displ_path = NULL;
}

void Patran::param(const char *paramName, bool /*inMapLoading*/)
{
    (void)paramName;
}

int Patran::compute(const char *)
{
    coDoUnstructuredGrid *mesh;
    coDistributedObject **time_outputgrid;

    StepFile *displ = NULL, *nsh = NULL, *elem = NULL;
    char *next_path = NULL;

    char buf[512];
    int i;
    float *x, *y, *z, *s;

    //read parameters
    grid_path = p_gridpath->getValue();
    displ_path = p_displpath->getValue();
    nsh_path = p_nshpath->getValue();
    elem_path = p_elempath->getValue();

    int timesteps = p_timesteps->getValue();
    int skip_value = p_skip->getValue();
    int nb_col = p_columns->getValue();

    int has_neutral_file = 0, has_displ_file = 0, has_nsh_file = 0, has_elem_file = 0;

    has_timesteps = 0;

    if (timesteps <= 0)
    {
        sendError("ERROR: The value of the timesteps should be >= 1!");
        return STOP_PIPELINE;
    }

    //
    // extracts the number of existing displ, nsh and elem files and then set the number of timesteps
    //

    if (strcmp(displ_path, init_path) && strcmp(displ_path, " ") && timesteps > 1)
    {
        int nb_disp = 0;
        displ = new StepFile(this, displ_path);
        displ->set_skip_value(skip_value);
        displ->get_nextpath(&next_path);
        while (next_path != NULL)
        {
            delete[] next_path;
            nb_disp++;
            displ->get_nextpath(&next_path);
        }
        delete displ;
        if (timesteps > 1 && nb_disp > 1)
            has_timesteps = 1;
        if (timesteps > nb_disp)
            if (nb_disp > 0)
                timesteps = nb_disp;
        //else timesteps = 1;
    }

    int stress = p_option->getValue();

    if (stress == 0)
    {
        if (strcmp(nsh_path, init_path) && strcmp(nsh_path, " ") && timesteps > 1)
        {
            int nb_nsh = 0;
            nsh = new StepFile(this, nsh_path);
            nsh->set_skip_value(skip_value);
            nsh->get_nextpath(&next_path);
            while (next_path != NULL)
            {
                delete[] next_path;
                nb_nsh++;
                nsh->get_nextpath(&next_path);
            }
            delete nsh;
            if (timesteps > 1 && nb_nsh > 1)
                has_timesteps = 1;
            if (timesteps > nb_nsh)
                if (nb_nsh > 0)
                    timesteps = nb_nsh;
            //else timesteps = 1;
        }
    }
    else if (strcmp(elem_path, init_path) && strcmp(elem_path, " ") && timesteps > 1)
    {
        int nb_elem = 0;
        elem = new StepFile(this, elem_path);
        elem->set_skip_value(skip_value);
        elem->get_nextpath(&next_path);
        while (next_path != NULL)
        {
            delete[] next_path;
            nb_elem++;
            elem->get_nextpath(&next_path);
        }
        delete elem;
        if (timesteps > 1 && nb_elem > 1)
            has_timesteps = 1;
        if (timesteps > nb_elem)
            if (nb_elem > 0)
                timesteps = nb_elem;
        //else timesteps = 1;
    }

    if (!has_timesteps)
    {
        next_path = new char[100];
    }

    //
    // grid file
    //

    if (gridFile)
        delete gridFile;
    sendInfo("Reading grid file: %s ...", grid_path);

    gridFile = new NeutralFile(grid_path);
    if ((!gridFile) || (!gridFile->isValid()))
    {
        sendError("Could not read %s as PATRAN Neutral File", grid_path);
        if (gridFile)
            delete gridFile;
        gridFile = NULL;
        return STOP_PIPELINE;
    }
    else
        has_neutral_file = 1;

    sendInfo("Successfully read the file '%s': %i nodes, %i elements, %i components.",
             grid_path, gridFile->num_nodes, gridFile->num_elements,
             gridFile->num_components);
    gridFile->eval_num_connections();

    if (timesteps > 1 && has_timesteps)
        sprintf(buf, "%s_0", p_outPort1->getObjName());
    else
        sprintf(buf, "%s", p_outPort1->getObjName());
    mesh = new coDoUnstructuredGrid(buf, gridFile->num_elements, gridFile->num_connections, gridFile->num_nodes, 1);

    if (!mesh->objectOk())
    {
        sendError("ERROR: Failed to create the object '%s' for the port '%s'", p_outPort1->getObjName(), p_outPort1->getName());
        return STOP_PIPELINE;
    }

    int *clPtr, *tlPtr, *elPtr;
    float *xPtr, *yPtr, *zPtr;

    mesh->getAddresses(&elPtr, &clPtr, &xPtr, &yPtr, &zPtr);
    mesh->getTypeList(&tlPtr);

    //
    //	ID´s
    //

    int *id;
    int size[2];
    size[0] = gridFile->num_elements;
    size[1] = 3;
    coDoIntArr *type = new coDoIntArr(p_outPort4->getObjName(), 2, size);
    if (!type->objectOk())
    {
        sendError("ERROR: Failed to create the object '%s' for the port '%s'", p_outPort4->getObjName(), p_outPort4->getName());
        return STOP_PIPELINE;
    }
    type->getAddress(&id);
    int *type_array = type->getAddress();

    gridFile->getMesh(elPtr, clPtr, tlPtr, xPtr, yPtr, zPtr, type_array);

    if (has_neutral_file)
    {
        if (timesteps > 1 && has_timesteps)
        {
            time_outputgrid = new coDistributedObject *[timesteps + 1];
            time_outputgrid[timesteps] = NULL;

            time_outputgrid[0] = mesh;
            for (i = 1; i < timesteps; i++)
            {
                time_outputgrid[i] = mesh;
                mesh->incRefCount();
            }
            time_outputgrid[i] = NULL;

            coDoSet *outputgrid = new coDoSet(p_outPort1->getObjName(), time_outputgrid);
            sprintf(buf, "1 %d", timesteps);
            outputgrid->addAttribute("TIMESTEP", buf);
            p_outPort1->setCurrentObject(outputgrid);
        }
        else
            p_outPort1->setCurrentObject(mesh);
    }

    //
    // 	vector data, displacement
    //

    coDistributedObject **time_outputdata;
    coDoVec3 *displ_data = NULL;

    if (strcmp(displ_path, init_path) && strcmp(displ_path, " "))
    {
        time_outputdata = new coDistributedObject *[timesteps + 1];
        if (has_timesteps)
        {
            displ = new StepFile(this, displ_path);
            displ->set_skip_value(skip_value);
        }
        for (i = 0; i < timesteps; i++)
        {
            if (has_timesteps)
                displ->get_nextpath(&next_path);
            else
                strcpy(next_path, displ_path);
            if (next_path)
            {
                FILE *fd = fopen(next_path, "r");
                if (fd)
                {
                    bool isAsciiFile = true;
                    char firstChars[80];
                    if (fread(&firstChars, 1, 80, fd) != 80)
                    {
                        fprintf(stderr, "ReadPatran::compute: fread failed\n");
                    }
                    fclose(fd);
                    int number;
                    for (number = 0; number < 80; number++)
                    {
                        isAsciiFile = isAsciiFile && (firstChars[number] > 7) && (firstChars[number] < 127);
                    }
                    sendInfo("Reading nodal displacement file: %s ...", next_path);
                    if (isAsciiFile)
                    {
                        nodal_displFile = new NodalFile(next_path, NASCII);
                    }
                    else
                    {
                        nodal_displFile = new NodalFile(next_path, NBINARY);
                    }
                    if ((!nodal_displFile) || (!nodal_displFile->isValid()))
                    {
                        sendError("Could not read %s as PATRAN Displacements File", next_path);
                        if (nodal_displFile)
                            delete nodal_displFile;
                        nodal_displFile = NULL;
                        //return STOP_PIPELINE;
                    }
                    else
                    {
                        has_displ_file = 1;
                        sendInfo("Successfully read the file '%s': %i data nodes, %i data columns.", next_path, nodal_displFile->nnodes, nodal_displFile->header.nwidth);
                        if (timesteps > 1 && has_timesteps)
                            sprintf(buf, "%s_%d", p_outPort2->getObjName(), i);
                        else
                            sprintf(buf, "%s", p_outPort2->getObjName());
                        displ_data = new coDoVec3(buf, gridFile->num_nodes);

                        if (!displ_data->objectOk())
                        {
                            sendError("ERROR: Failed to create the object '%s' for the port '%s'", buf, p_outPort2->getName());
                            return STOP_PIPELINE;
                        }
                        displ_data->getAddresses(&x, &y, &z);
                        if (nodal_displFile->getDataField(NDISPLACEMENTS, gridFile->nodeMap, x, y, z, gridFile->num_nodes - nodal_displFile->nnodes, gridFile->getMaxnode()) < 0)
                            sendError("ERROR: Cannot read Nodal Displacements for Port Data%s", p_outPort2->getName());
                        time_outputdata[i] = displ_data;
                        delete nodal_displFile;
                    }
                }
                else
                {
                    sendWarning("Could not read %s as PATRAN Displacements File", next_path);
                    if (nodal_displFile)
                        delete nodal_displFile;
                    nodal_displFile = NULL;
                    i = timesteps; //get out of the loop
                }

                if (has_timesteps)
                    delete[] next_path;
            }
        }
        time_outputdata[i] = NULL;

        if (has_displ_file)
        {
            if (timesteps > 1 && has_timesteps)
            {
                coDoSet *outputdata = new coDoSet(p_outPort2->getObjName(), time_outputdata);
                sprintf(buf, "1 %d", timesteps);
                outputdata->addAttribute("TIMESTEP", buf);
                p_outPort2->setCurrentObject(outputdata);
            }
            else
                p_outPort2->setCurrentObject(displ_data);
        }

        if (has_timesteps)
            delete displ;
        //setobj later
    }

    //
    //	scalar nsh data
    //

    if (strcmp(nsh_path, init_path) && strcmp(nsh_path, " ") && stress == 0)
    {
        coDistributedObject **time_scalardata = new coDistributedObject *[timesteps + 1];
        if (has_timesteps)
        {
            nsh = new StepFile(this, nsh_path);
            nsh->set_skip_value(skip_value);
        }

        coDoFloat *nsh_data = NULL;

        for (i = 0; i < timesteps; i++)
        {
            if (has_timesteps)
                nsh->get_nextpath(&next_path);
            else
                strcpy(next_path, nsh_path);
            if (next_path)
            {
                FILE *fd = fopen(next_path, "r");
                if (fd)
                {
                    bool isAsciiFile = true;
                    char firstChars[180];
                    if (fread(&firstChars, 1, 80, fd) != 80)
                    {
                        fprintf(stderr, "fread failed\n");
                    }
                    fclose(fd);

                    int number;
                    for (number = 0; number < 80; number++)
                    {
                        isAsciiFile = isAsciiFile && ((firstChars[number] > 7) && ((firstChars[number] < 127)));
                    }

                    if (isAsciiFile)
                    {
                        nodal_stressFile = new NodalFile(next_path, NASCII);
                    }
                    else
                    {
                        nodal_stressFile = new NodalFile(next_path, NBINARY);
                    }
                    //binary file
                    sendInfo("Reading nodal stress file: %s ...", next_path);
                    if ((!nodal_stressFile) || (!nodal_stressFile->isValid()))
                    {
                        sendError("Could not read %s as PATRAN nodal results File", next_path);
                        if (nodal_stressFile)
                            delete nodal_stressFile;
                        nodal_stressFile = NULL;
                        //return STOP_PIPELINE;
                    }
                    else
                    {
                        has_nsh_file = 1;
                        sendInfo("Successfully read the file '%s': %i nodes", next_path, nodal_stressFile->nnodes);
                        if (timesteps > 1 && has_timesteps)
                            sprintf(buf, "%s_%d", p_outPort3->getObjName(), i);
                        else
                            sprintf(buf, "%s", p_outPort3->getObjName());
                        nsh_data = new coDoFloat(buf, gridFile->num_nodes);

                        if (!nsh_data->objectOk())
                        {
                            sendError("ERROR: Failed to create the object '%s' for the port '%s'", buf, p_outPort3->getName());
                            return STOP_PIPELINE;
                        }
                        nsh_data->getAddress(&s);
                        if (nodal_stressFile->getDataField(NNODALSTRESS, gridFile->nodeMap, nb_col, s, gridFile->num_nodes - nodal_stressFile->nnodes, gridFile->getMaxnode()) < 0)
                            sendError("ERROR: Cannot read Nodal Result for Port Data %s", p_outPort3->getName());
                        time_scalardata[i] = nsh_data;
                        delete nodal_stressFile;
                    }
                }
                else
                {
                    sendWarning("Could not read %s as PATRAN nodal results File", next_path);
                    if (nodal_stressFile)
                        delete nodal_stressFile;
                    nodal_stressFile = NULL;
                    i = timesteps; //get out of the loop
                }
                if (has_timesteps)
                    delete[] next_path;
            }
        }
        time_scalardata[i] = NULL;

        if (has_nsh_file)
        {
            if (timesteps > 1 && has_timesteps)
            {
                coDoSet *scalardata = new coDoSet(p_outPort3->getObjName(), time_scalardata);
                sprintf(buf, "1 %d", timesteps);
                scalardata->addAttribute("TIMESTEP", buf);
                p_outPort3->setCurrentObject(scalardata);
            }
            else
                p_outPort3->setCurrentObject(nsh_data);
        }

        if (has_timesteps)
            delete nsh;
    }

    //
    //	scalar element results data
    //

    if (strcmp(elem_path, init_path) && strcmp(elem_path, " ") && stress == 1)
    {
        coDistributedObject **time_elementdata = new coDistributedObject *[timesteps + 1];
        if (has_timesteps)
        {
            elem = new StepFile(this, elem_path);
            elem->set_skip_value(skip_value);
        }
        else
            strcpy(next_path, elem_path);

        coDoFloat *elem_data = NULL;

        for (i = 0; i < timesteps; i++)
        {
            if (has_timesteps)
                elem->get_nextpath(&next_path);
            else
                strcpy(next_path, elem_path);

            if (next_path)
            {
                FILE *fd = fopen(next_path, "r");
                if (fd)
                {
                    bool isAsciiFile = true;
                    char firstChars[80];
                    if (fread(&firstChars, 1, 80, fd) != 80)
                    {
                        fprintf(stderr, "ReadPatran::compute: fread failed\n");
                    }
                    fclose(fd);
                    int number;
                    for (number = 0; number < 80; number++)
                    {
                        isAsciiFile = isAsciiFile && (firstChars[number] > 7) && (firstChars[number] < 127);
                    }
                    if (isAsciiFile)
                    {
                        //ASCII file
                        sendInfo("Reading element result file: %s ...", next_path);
                        elemAscFile = new ElementAscFile(next_path, nb_col);
                        if ((!elemAscFile) || (!elemAscFile->isValid()))
                        {
                            sendError("Could not read %s as PATRAN Element Result File", next_path);
                            if (elemAscFile)
                                delete elemAscFile;
                            elemAscFile = NULL;
                        }
                        else
                        {
                            has_elem_file = 1;
                            sendInfo("Successfully read the file '%s': %i nodes and width %i.", next_path, elemAscFile->nnodes, elemAscFile->nwidth);
                            if (timesteps > 1 && has_timesteps)
                                sprintf(buf, "%s_%d", p_outPort3->getObjName(), i);
                            else
                                sprintf(buf, "%s", p_outPort3->getObjName());
                            elem_data = new coDoFloat(buf, gridFile->num_elements);

                            if (!elem_data->objectOk())
                            {
                                sendError("ERROR: Failed to create the object '%s' for the port '%s'", buf, p_outPort3->getName());
                                return STOP_PIPELINE;
                            }
                            elem_data->getAddress(&s);
                            // 3 = ELEMENTSTRESS
                            if (elemAscFile->getDataField(3, gridFile->elemMap, nb_col, s, gridFile->num_elements - elemAscFile->nnodes, gridFile->getMaxelem()) < 0)
                                sendError("ERROR: Cannot read Nodal result for Port Data %s", p_outPort3->getName());

                            time_elementdata[i] = elem_data;
                            delete elemAscFile;
                        }
                    }

                    else
                    { // binary
                        sendInfo("Reading element result file: %s ...", next_path);
                        elemFile = new ElementFile(::open(next_path, O_RDONLY));
                        if ((!elemFile) || (!elemFile->isValid()))
                        {
                            sendError("Could not read %s as PATRAN element result File", next_path);
                            if (elemFile)
                                delete elemFile;
                            elemFile = NULL;
                            return STOP_PIPELINE;
                        }
                        else
                        {
                            has_elem_file = 1;
                            sendInfo("Successfully read the file '%s': %i lines.", next_path, elemFile->numlines);
                            if (timesteps > 1 && has_timesteps)
                                sprintf(buf, "%s_%d", p_outPort3->getObjName(), i);
                            else
                                sprintf(buf, "%s", p_outPort3->getObjName());
                            elem_data = new coDoFloat(buf, gridFile->num_elements);

                            if (!elem_data->objectOk())
                            {
                                sendError("ERROR: Failed to create the object '%s' for the port '%s'", buf, p_outPort3->getName());
                                return STOP_PIPELINE;
                            }
                            elem_data->getAddress(&s);
                            if (elemFile->getDataField(3, gridFile->elemMap, nb_col, s, gridFile->num_elements - elemFile->numlines, gridFile->getMaxelem()) < 0)
                                sendError("ERROR: Cannot read Element Results for Port Data %s", p_outPort3->getName());
                            time_elementdata[i] = elem_data;
                            delete elemFile;
                            // delete [] next_path;
                        }
                    }
                }
                else
                {
                    sendError("Could not read %s as PATRAN Element Result File", next_path);
                    if (elemAscFile)
                        delete elemAscFile;
                    elemAscFile = NULL;
                    i = timesteps; //get out of the loop
                }
                if (has_timesteps)
                    delete[] next_path;
            }
        }
        time_elementdata[i] = NULL;

        if (has_elem_file)
        {
            if (timesteps > 1 && has_timesteps)
            {
                coDoSet *elementdata = new coDoSet(p_outPort3->getObjName(), time_elementdata);
                sprintf(buf, "1 %d", timesteps);
                elementdata->addAttribute("TIMESTEP", buf);
                p_outPort3->setCurrentObject(elem_data);
            }
            else
                p_outPort3->setCurrentObject(elem_data);
        }

        //      delete [] time_elementdata;
        if (has_timesteps)
            delete elem;
    }
    if (!has_timesteps)
        delete[] next_path;

    return SUCCESS;
}

Patran::~Patran()
{
    delete gridFile;
    // delete displFile;
}

MODULE_MAIN(IO, Patran)
