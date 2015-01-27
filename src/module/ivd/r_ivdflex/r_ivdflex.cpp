/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                   	                  **
 **                                                                        **
 ** R_IVDFLEX                                          	                  **
 ** Author: Michael Junge, IVD Uni-Stutgart                                **
 **                                                                        **
 ** Date: Oct, 2000 up to Nov 2001                                         **
 **                                                                        **
 ** Description: This module imports data written in a IBM-OpenDX          **
 **              compatible format by the export routines of IOLOS.        **
 **              The read data information is assigned to the output-ports **
 **              of the module.                                            **
 **                                                                        **
\**************************************************************************/

//lenght of a line

// portion for resizing data
#define CHUNK_SIZE 4096

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <do/coDoSet.h>
#include "r_ivdflex.h"

/**************************************************************************\ 
 **  Swaps binary data: This procedure is used to swap data when the binary**
 **                     file was created on a PC and afterwards read on    **
 **                     a SGI system or via verse.                         **
\**************************************************************************/
inline void swap_int(int &d)
{
    unsigned int &data = (unsigned int &)d;
    data = ((data & 0xff000000) >> 24)
           | ((data & 0x00ff0000) >> 8)
           | ((data & 0x0000ff00) << 8)
           | ((data & 0x000000ff) << 24);
}

inline void swap_float(float *d, int num)
{
    unsigned int *data = (unsigned int *)d;
    int i;
    for (i = 0; i < num; i++)
    {
        *data = (((*data) & 0xff000000) >> 24)
                | (((*data) & 0x00ff0000) >> 8)
                | (((*data) & 0x0000ff00) << 8)
                | (((*data) & 0x000000ff) << 24);
        data++;
    }
}

/**************************************************************************\ 
 **   ReadIVDdata: this method is called, when module is initialized       **
\**************************************************************************/
ReadIVDdata::ReadIVDdata(int argc, char *argv[])
    : coModule(argc, argv)
{
    int i; //counter
    char portname[STR_MAX];

    // this info appears in the module setup window
    set_module_description("IVD Import Routine");
    //define output ports choices

    // create the output port for Grid
    p_Grid = addOutputPort("Grid", "UnstructuredGrid", "Grid");
    // create the output ports for the vector data. To increase the number of available output ports
    // change MAXvector parameter in header-file
    for (i = 0; i < MAXvector; i++)
    {
        sprintf(portname, "veloctity%d", i);
        p_Vector[i] = addOutputPort(portname, "Vec3", "velocity field");
    }
    // create the output ports for the vector data. To increase the number of available output ports
    // change MAXscalar parameter in header-file
    for (i = 0; i < MAXscalar; i++)
    {
        sprintf(portname, "scalar%d", i);
        p_Scalar[i] = addOutputPort(portname, "Float", "Scalar data");
    }

    // select the General file name with a file browser
    p_FileBrowser = addFileBrowserParam("GeneralFile", "Path to General file");
    // set default path and default filter parameters
    p_FileBrowser->setValue("~/", "*.COVgen");

    // add two parameters. To choose the starting and ending timesteps
    // Start counting at number 0
    pnumberTimestart = addInt32Param("StartingTimestep", "Anfang (0,1,?)");
    pnumberTimestart->setValue(0);
    pnumberTimestop = addInt32Param("EndingTimestep", "Ende (0,1,?)");
    pnumberTimestop->setValue(0);

    //If set binary data will be swapped from big to little endian and via verse
    pswapdata = addBooleanParam("SwapBinaryData", "If set binary data will be swapped from big to little endian and via verse");
    pswapdata->setValue(false);

    //    pSelectPort = addChoiceParam("SelectOutput","select the other choice's values");
    //    pSelectPort->setValue(3,selectableChoice,0);
}

ReadIVDdata::~ReadIVDdata()
{
}

void ReadIVDdata::quit()
{
}

/**************************************************************************\ 
 **  compute : this method is called, when module is executed              **
\**************************************************************************/

int ReadIVDdata::compute(const char *)
{
    int stepnr; // counter of Timestepss
    char filename[STR_MAX]; // constructed filename
    int i;
    int gridcounter; // counter of for loop ovewr all grids
    int counts, countv; //counter for vector function
    int data_offset; // offset of data from the beginning of data
    // pointer to reserved memory to the actual position
    // each time when a grid information field has been
    // read the data_offset counter is increased
    int var_index;
    int scalar_count;
    int vector_count;
    char buffer[STR_MAX];

    coDistributedObject **Scalar_set_t[MAXscalar];
    coDistributedObject **Vector_set_t[MAXvector];

    // get the file name of general file
    general_filename = p_FileBrowser->getValue();
    startTimesteps = pnumberTimestart->getValue();
    // number of grids; Mehrgebiet
    numberTimesteps = pnumberTimestop->getValue() - (startTimesteps) + 1;
    swapdata = pswapdata->getValue(); //should binary data be swaped after reading?

    //create a set of grid;
    coDistributedObject **Grid_set_t = new coDistributedObject *[numberTimesteps + 1];
    Grid_set_t[0] = NULL;

    //create a set of vector and scalar data;
    for (i = 0; i < MAXscalar; i++)
    {
        Scalar_set_t[i] = new coDistributedObject *[numberTimesteps + 1];
        Scalar_set_t[i][0] = NULL;
    }
    for (i = 0; i < MAXvector; i++)
    {
        Vector_set_t[i] = new coDistributedObject *[numberTimesteps + 1];
        Vector_set_t[i][0] = NULL;
    }
#ifdef WIN32
    if ((fp = fopen((char *)general_filename, "rb")) != NULL)
#else
    if ((fp = fopen((char *)general_filename, "r")) != NULL)
#endif
    {
        sendInfo("File %s will be read", general_filename);
        /* get the general information about data and grid */
        read_general(fp);
        fclose(fp);
        /* now we know the grid size and data structure */
        /* generate a grid */
        /* get the COVISE output object name from the controller */
        GridObjectName = p_Grid->getObjName();
        // generate name  GridObjectName + _i
        sprintf(buffer, "%s_Grid", GridObjectName);
        // Pointer to grid field
        d_Grid = new coDoUnstructuredGrid(buffer, total_elem, total_conn, total_coord, 1);
        if (!(d_Grid->objectOk()))
        {
            sendError("Could not create Grid Data");
            return STOP_PIPELINE;
        }
        //   /* close general file and reopen it to get back to the */
        //   /* beginning. Get connectivity */
        //   fp = Covise::fopen((char *)general_filename, "r");
        //   read_gridconnectivity(fp,d_Grid);
        read_gridconnectivity2(d_Grid);
        /* read the grid coordinates. This information is located */
        /* within the data file at first position                 */
        read_grid(d_Grid);
        Grid_set_t[0] = d_Grid;
        Grid_set_t[numberTimesteps] = NULL;
        for (i = 1; i < numberTimesteps; i++)
        {
            Grid_set_t[i] = Grid_set_t[0];
            Grid_set_t[0]->incRefCount();
        }
        //fclose(fp);

        /*get now the data for each gridpart and timestep */
        for (stepnr = 0; stepnr < numberTimesteps; stepnr++)
        {
            scalar_count = 0;
            vector_count = 0;
            data_offset = 0;
            /*generate data field for thew whole data information. All information
           of one concentration over all grids is kept in one data field*/
            /* how many data fields have to generated */
            for (var_index = 0; var_index < MAXvector + MAXscalar; var_index++)
            {
                if (structure[var_index] == SCALAR)
                    scalar_count++;
                if (structure[var_index] == VECTOR)
                    vector_count++;
            }
            /* now I know how many scalar and vector data are present */
            /*generate now data fields */
            for (var_index = 0; var_index < scalar_count; var_index++)
            {
                GridObjectName = p_Scalar[var_index]->getObjName();
                // generate name  VectorName + _i
                sprintf(buffer, "scal%s_%d_%d", GridObjectName, var_index, stepnr);
                d_Scalar[var_index] = new coDoFloat(buffer, total_coord);
                if (!(d_Scalar[var_index]->objectOk()))
                {
                    sendInfo("Error while creating data field scalar nr %d name %s", var_index, buffer);
                    return STOP_PIPELINE;
                }
            }
            for (var_index = 0; var_index < vector_count; var_index++)
            {
                GridObjectName = p_Vector[var_index]->getObjName();
                // generate name  VectorName + _i
                sprintf(buffer, "vec%s_%d_%d", GridObjectName, var_index, stepnr);
                d_Vector[var_index] = new coDoVec3(buffer, total_coord);
                if (!(d_Vector[var_index]->objectOk()))
                {
                    sendInfo("Error while creating data field vector nr %d name %s", var_index, buffer);
                    return STOP_PIPELINE;
                }
            }
            /* read_data of all grids and all data fields  */

            for (gridcounter = 0; gridcounter < num_grids; gridcounter++)
            { /* gridcounter */
                if (!open_data(&dfp, general_filename, data_filename[gridcounter], stepnr + startTimesteps))
                {
                    sendInfo("Error : Could not open datafile %s ", data_filename[gridcounter]);
                    return STOP_PIPELINE;
                }
                /* skip grid information when timestep is #0 */
                if (stepnr + startTimesteps == 0)
                {
                    skip_block(dfp);
                    skip_block(dfp);
                    skip_block(dfp);
                }
                counts = 0;
                countv = 0;
                for (var_index = 0; var_index < MAXvector + MAXscalar; var_index++)
                {
                    if (structure[var_index] == SCALAR)
                    {
                        sendInfo("Readscalar var_index %d ", var_index);
                        /* read information of one scalar of one field */
                        if (!read_scalar(dfp, counts, local_coord[gridcounter], data_offset))
                            return STOP_PIPELINE;
                        counts++;
                    }
                    if (structure[var_index] == VECTOR)
                    {
                        sendInfo("Readvector var_index %d ", var_index);
                        /* read information of one vector of one field */
                        if (!read_vector(dfp, countv, local_coord[gridcounter], data_offset))
                            return STOP_PIPELINE;
                        countv++;
                    }
                }
                /* move data_offset forward by the size of olf grid */
                data_offset += local_coord[gridcounter];
                fclose(dfp);
            } /* gridcounter */
            counts = 0;
            countv = 0;
            /* put the read data of each scalar/vector data
            of one timestep into the array  */
            for (var_index = 0; var_index < MAXvector + MAXscalar; var_index++)
            {
                if (structure[var_index] == SCALAR)
                {
                    Scalar_set_t[counts][stepnr] = d_Scalar[counts];
                    Scalar_set_t[counts][stepnr + 1] = NULL;
                    counts++;
                }
                if (structure[var_index] == VECTOR)
                {
                    Vector_set_t[countv][stepnr] = d_Vector[countv];
                    Vector_set_t[countv][stepnr + 1] = NULL;
                    countv++;
                }
            }

        } // for timestepsschleife
    } // if Covise::fopen
    else
    {
        sendInfo("Generalfile %s was not found! Please select general file", filename);
        return FAIL;
    }

    GridObjectName = p_Grid->getObjName();
    coDoSet *Grid_set_all = new coDoSet(GridObjectName, Grid_set_t);
    delete Grid_set_t[0];
    delete[] Grid_set_t;
    Grid_set_all->addAttribute("TIMESTEP", "1 2000");

    for (var_index = 0; var_index < num_scalar; var_index++)
    {
        GridObjectName = p_Scalar[var_index]->getObjName();
        new coDoSet(GridObjectName, Scalar_set_t[var_index]);
        //coDoSet *Scalar_set_all= new coDoSet(GridObjectName,Scalar_set_t[var_index]);

        for (i = 0; Scalar_set_t[var_index][i]; i++)
        {
            delete Scalar_set_t[var_index][i];
        }
        delete[] Scalar_set_t[var_index];
    }

    for (var_index = 0; var_index < num_vector; var_index++)
    {
        GridObjectName = p_Vector[var_index]->getObjName();
        new coDoSet(GridObjectName, Vector_set_t[var_index]);
        for (i = 0; Vector_set_t[var_index][i]; i++)
        {
            delete Vector_set_t[var_index][i];
        }
        delete[] Vector_set_t[var_index];
    }

    for (i = 0; i < num_grids; i++)
    {
        delete data_filename[i];
    }
    for (i = 0; i < num_scalar + num_vector; i++)
    {
        delete field[i];
    }
    delete connection_filename;
    return SUCCESS;
}

int ReadIVDdata::read_general(FILE *filepointer)
/* This routine reads the information given in the general file                   */
/* In the general file should be retrieved information the grid structure         */
/* the data names and types, etc. For more details take a look at the demo        */
/* general file "demo.general"                                                    */
/* The routines quits when the first grid information area is found               */
{
    char dummy[STR_MAX]; // buffer for COVISE info and error messages
    char line[LINE_size]; // line in an obj file
    char *first; // current position in line
    char *seperator; // position of seperator. seperates id and data
    // in general file
    //    char *namepointer;           // pointer to name
    int i; // counter
    char *p_token; // pointer to token

    //initialize variables
    num_scalar = 0;
    num_vector = 0;
    total_elem = 0;
    total_coord = 0;
    total_elem = 0;
    total_conn = 0;
    num_grids = 0;
    for (i = 0; i < MAXscalar + MAXvector; i++)
    {
        field[i] = NULL;
        structure[i] = EMPTY;
    }
    // read one line after another of general file
    /* Parse through general file. When grid information field is reached
      quit while loop */
    while (fgets(line, LINE_size, filepointer) != NULL)
    {
        // find first non-space character
        first = line;
        while (*first != '\0' && isspace(*first))
            first++;

        // skip blank lines and comments
        if (*first == '\0' || *first == '#')
            // read the next line
            continue;

        // look for "end". -> general information complete
        //seperator=strstr(first,"end");
        //if (seperator!=NULL) abbruch=TRUE; //end. Stop reading. Greid data field folows

        // look for the seperator =
        seperator = strstr(first, "=");
        if (seperator == NULL)
            continue; //no seperator --> read next line

        seperator++; // move pointer one char behind
        /***************** look for keywords*******************/
        if (strstr(first, "field") != NULL)
        {
            //found field description --> parse through each description

            p_token = strtok(seperator, ",");
            if (strstr(p_token, "locations") == NULL)
            {
                sendError("no grid information found");
                break;
            }
            i = 0;
            while ((p_token = strtok(NULL, ",")) != NULL)
            {
                field[i] = new char[strlen(p_token) + 1];
                strcpy(field[i], p_token);
                sendInfo("Found data %d description %s", i, field[i]);
                i++;
                if (i > MAXscalar + MAXvector)
                {
                    sendError("too many data fields. Enlarge MAXscalar+MAXvector constants");
                    break;
                }
            }
        }
        else if (strstr(first, "structure") != NULL)
        {
            //found structure description --> parse through each description

            p_token = strtok(seperator, ",");
            if (strstr(p_token, "3-vector") == NULL)
            {
                sendError("wrong grid information found");
                break;
            }
            i = 0;
            while ((p_token = strtok(NULL, ",")) != NULL)
            {
                if (strstr(p_token, "3-vector") != NULL)
                {
                    structure[i] = VECTOR;
                }
                else if (strstr(p_token, "scalar") != NULL)
                {
                    structure[i] = SCALAR;
                }
                else
                {
                    sendError("wrong datatype information found");
                    break;
                }

                i++;
                if (i > MAXscalar + MAXvector)
                {
                    sendError("too many data fields. Enlarge MAXscalar+MAXvector constants");
                    break;
                }
            }
        }
        else if (strstr(first, "data file") != NULL)
        {
            if (num_grids >= MAXgrids)
            {
                sendError("too many grid fields");
                return STOP_PIPELINE; //no grid information
            }
            sscanf(seperator, " %s ", dummy);
            data_filename[num_grids] = new char[strlen(dummy) + 1];
            //at the end of datafilename is a CR. Don"t copy that char.
            strcpy(data_filename[num_grids], dummy);
            sendInfo("Found Datafilename %d: %s", num_grids, data_filename[num_grids]);
            num_grids++;
        }
        else if (strstr(first, "connections file") != NULL)
        {
            sscanf(seperator, " %s ", dummy);
            connection_filename = new char[strlen(dummy) + 1];
            //at the end of datafilename is a CR. Don"t copy that char.
            strcpy(connection_filename, dummy);
            sendInfo("Found Filename for connectivity %s", connection_filename);
        }
        else if (strstr(first, "localcoord") != NULL)
        {
            sscanf(seperator, " %d ", &local_coord[num_grids - 1]);
            total_coord += local_coord[num_grids - 1];
        }
        else if (strstr(first, "localelem") != NULL)
        {
            sscanf(seperator, " %d ", &local_elem[num_grids - 1]);
            total_elem += local_elem[num_grids - 1];
        }
        else if (strstr(first, "localconnections") != NULL)
        {
            sscanf(seperator, "%d", &local_conn[num_grids - 1]);
            total_conn += local_conn[num_grids - 1];
        }
    }
    /* parsed through the whole file. Check now if all necesarry data
      is available, sort data */

    /*        if (abbruch==FALSE)
             {
               sendError("no grid grid connetion field information found!");
               return(FALSE); //no grid information
             }
   */
    if ((total_coord == 0) | (total_elem == 0) | (total_conn == 0))
    {
        sendError("no information for total grid found");
        return STOP_PIPELINE; //no grid information
    }
    if (num_grids == 0)
    {
        sendError("no information about number of grids found");
        return STOP_PIPELINE; //no grid information
    }
    if (num_grids >= MAXgrids)
    {
        sendError("too many grid fields");
        return STOP_PIPELINE; //no grid information
    }

    /* count number of vector and scalar variables. Already made sure that
      number of variables doesn't exceed allocated ports.*/

    for (i = 0; i < MAXvector + MAXscalar; i++)
    {
        if (structure[i] == SCALAR)
            num_scalar++;
        else if (structure[i] == VECTOR)
            num_vector++;
    }
    sendInfo("%d scalar %d vector fields found", num_scalar, num_vector);
    /*        for (i=0; i<MAXvector+MAXscalar; i++)
           {
               if ((structure[i]==SCALAR)||(structure[i]==VECTOR))
                 read_var[i]=TRUE;
               else
                 read_var[i]=FALSE;
           }
   */
    if ((num_scalar + num_vector) == 0)
    {
        sendError("no datafields found");
        return STOP_PIPELINE;
    }
    return CONTINUE_PIPELINE;
}

bool ReadIVDdata::read_gridconnectivity2(coDoUnstructuredGrid *pGrid_data)
{
    int dummy1, dummy2, dummy3; // will hold fortran block marker
    int vl_Offset_local_2_global; //Offset between local and global vertices number
    int el_Offset_local_2_global; //Offset between local and global elements number
    int i;
    int z;
    int Grid_nr;
    //    int abbruch;
    //    char infobuf[STR_MAX];           // buffer for COVISE info and error messages
    //    char line[LINE_size];		 // line in an obj file
    //    char *first;			     // current position in line
    FILE *dfp;

    pGrid_data->getAddresses(&p_el, &p_vl, &p_xcoord, &p_ycoord, &p_zcoord);
    pGrid_data->getTypeList(&p_tl);

    vl_Offset_local_2_global = 0;
    el_Offset_local_2_global = 0;
    //open datafile
    if (!open_data(&dfp, general_filename, connection_filename, 0))
    {
        sendError("Error opening Connectivity-File");
        return false;
    }
    for (Grid_nr = 0; Grid_nr < num_grids; Grid_nr++)
    { //for
        //       abbruch=FALSE;
        //read connectivity information
        //       for ( i=0; i<local_elem[Grid_nr]; i++)
        {
            /*        fgets(line, LINE_size, fp);
         //        sprintf(infobuf,"Zeile %d Inhalt: %s ",i,line );
         //        sendInfo(infobuf);
                 //get the vertext list for one Hexaeder
                 sscanf(line,"%d %d %d %d %d %d %d %d",p_vl+5,p_vl+4
                              ,p_vl+1,p_vl,p_vl+6,p_vl+7,p_vl+2,p_vl+3);
                 //now we read local grid. The number given for a vertex is a local
                 //number. This number has to be transformed to the global grid
         */
            //This is realized by an offset.

            fread(&dummy1, sizeof(int), 1, dfp);
            if (swapdata)
                swap_int(dummy1);

            fread(p_vl, sizeof(int), (local_elem[Grid_nr] * 8), dfp);
            if (swapdata)
                for (z = 0; z < local_elem[Grid_nr] * 8; z++)
                {
                    dummy3 = *(p_vl + z);
                    swap_int(dummy3);
                    *(p_vl + z) = dummy3;
                }
            fread(&dummy2, sizeof(int), 1, dfp);
            if (swapdata)
                swap_int(dummy2);
            if (dummy1 != dummy2)
            {
                Covise::sendError("ERROR wrong FORTRAN block marker reading connectivity");
                return false;
            }
            for (z = 0; z < local_elem[Grid_nr] * 8; z++)
            {
                *(p_vl + z) = *(p_vl + z) + vl_Offset_local_2_global;
            }
            //each hexaeder has 8 vertexes, move to the next empty field
            //        p_vl=p_vl+8;
            p_vl = p_vl + local_elem[Grid_nr] * 8;
            //move pointer to next element
            for (i = 0; i < local_elem[Grid_nr]; i++)
            {
                //        *p_el++=i*8+el_Offset_local_2_global;
                //        *p_tl++=TYPE_HEXAEDER;  // all elements are hexaeders
                *p_el++ = i * 8 + el_Offset_local_2_global;
                *p_tl++ = TYPE_HEXAEDER; // all elements are hexaeders
            }
        }
        vl_Offset_local_2_global += local_coord[Grid_nr];
        el_Offset_local_2_global += local_elem[Grid_nr] * 8;
    } //for
    fclose(dfp);
    return (SUCCESS);
}

bool ReadIVDdata::read_grid(coDoUnstructuredGrid *pGrid_data)
{
    int dummy1, dummy2; // will hold fortran block marker
    int Grid_nr;
    FILE *dfp;

    pGrid_data->getAddresses(&p_el, &p_vl, &p_xcoord, &p_ycoord, &p_zcoord);

    // read all grid fields step by step
    for (Grid_nr = 0; Grid_nr < num_grids; Grid_nr++)
    { //for
        //open datafile
        if (!open_data(&dfp, general_filename, data_filename[Grid_nr], 0))
        {
            sendError("Error opening Data-File");
            return false;
        }
        /*--------  x-Daten ----------*/
        fread(&dummy1, sizeof(int), 1, dfp);
        fread(p_xcoord, sizeof(int), local_coord[Grid_nr], dfp);
        if (swapdata)
            swap_float(p_xcoord, local_coord[Grid_nr]);
        fread(&dummy2, sizeof(int), 1, dfp);
        if (dummy1 != dummy2)
        {
            Covise::sendError("ERROR wrong FORTRAN block marker reading header grid1");
            return false;
        }
        p_xcoord = p_xcoord + local_coord[Grid_nr];
        /*--------  y-Daten ----------*/
        fread(&dummy1, sizeof(int), 1, dfp);
        fread(p_ycoord, sizeof(int), (local_coord[Grid_nr]), dfp);
        if (swapdata)
            swap_float(p_ycoord, local_coord[Grid_nr]);

        fread(&dummy2, sizeof(int), 1, dfp);
        if (dummy1 != dummy2)
        {
            Covise::sendError("ERROR wrong FORTRAN block marker reading header grid2");
            return false;
        }

        p_ycoord = p_ycoord + local_coord[Grid_nr];
        /*--------  z-Daten ----------*/
        fread(&dummy1, sizeof(int), 1, dfp);
        fread(p_zcoord, sizeof(int), local_coord[Grid_nr], dfp);
        if (swapdata)
            swap_float(p_zcoord, local_coord[Grid_nr]);

        fread(&dummy2, sizeof(int), 1, dfp);
        if (dummy1 != dummy2)
        {
            Covise::sendError("ERROR wrong FORTRAN block marker reading header grid3");
            return false;
        }
        p_zcoord = p_zcoord + local_coord[Grid_nr];

        fclose(dfp);
    } /* for */

    return true;
}

/**************************************************************************\ 
 **  open_data :   generates filename combined out of the path of the      **
 **                general-file and the name provided in datafilename,     **
 **                open this file and returns a file pointer to it in dfp  **
 **                If stepnr>0 numberation is added to filename            **
\**************************************************************************/
bool ReadIVDdata::open_data(FILE **dfp,
                            const char *generalfilename,
                            const char *datafilename,
                            int stepnr)
{
    char composedfn[STR_MAX];
    char dummy2[STR_MAX];
    char nummer[STR_MAX];
    int i;

    // get the path of general file and combine path with datafilename
    strcpy(dummy2, generalfilename);
    i = strlen(dummy2) - 1;
    while ((dummy2[i] != '/') & (i >= 0))
        i--;
    //found path
    dummy2[i + 1] = '\0';
    strcpy(composedfn, dummy2);
    strcat(composedfn, (const char *)datafilename);

    /* for more timesteps then one, add .0001 .0002, etc  */
    if (stepnr > 0)
    {
        sprintf(nummer, ".%04d", stepnr);
        strcat(composedfn, nummer);
    }
#ifdef WIN32
    if ((*dfp = fopen((char *)composedfn, "rb")) == NULL)
#else
    if ((*dfp = fopen((char *)composedfn, "r")) == NULL)
#endif
    {
        sendError("ERROR: Can't open data file >> %s", (const char *)data_filename);
        return false;
    }
    else
    {
        sendInfo("Data file %s opened", composedfn);
        return true;
    }
}

/**************************************************************************\ 
 ** read_scalar  : reads one scalar of the size local_coord from file      **
 **                provided in dfp and assigned it  to d_scalar[index]     **
 **                with the provided offset                                **
\**************************************************************************/
bool ReadIVDdata::read_scalar(FILE *dfp, int index, int local_coord, int offset)
{
    //     char buffer[STR_MAX];
    int dummy1, dummy2; // will hold fortran block marker

    d_Scalar[index]->getAddress(&p_scalarcoord);
    /*--------  x-Daten ----------*/
    fread(&dummy1, sizeof(int), 1, dfp);
    fread(p_scalarcoord + offset, sizeof(int), local_coord, dfp);
    if (swapdata)
        swap_float(p_scalarcoord + offset, local_coord);

    fread(&dummy2, sizeof(int), 1, dfp);
    if (dummy1 != dummy2)
    {
        Covise::sendError("ERROR wrong FORTRAN block marker reading scalar");
        return false;
    }

    return true;
}

/**************************************************************************\ 
 ** read_vector  : reads one vector of the size local_coord from file      **
 **                provided in dfp and assigned it   to d_Vector[index]    **
 **                with the provided offset                                **
\**************************************************************************/
bool ReadIVDdata::read_vector(FILE *dfp, int index, int local_coord, int offset)
{
    //     char buffer[STR_MAX];
    int dummy1, dummy2; // will hold fortran block marker

    d_Vector[index]->getAddresses(&p_xcoord, &p_ycoord, &p_zcoord);
    /*--------  x-Daten ----------*/
    fread(&dummy1, sizeof(int), 1, dfp);
    fread(p_xcoord + offset, sizeof(int), local_coord, dfp);
    if (swapdata)
        swap_float(p_xcoord + offset, local_coord);
    fread(&dummy2, sizeof(int), 1, dfp);
    if (dummy1 != dummy2)
    {
        Covise::sendError("ERROR wrong FORTRAN block marker reading vector");
        return false;
    }
    /*--------  y-Daten ----------*/
    fread(&dummy1, sizeof(int), 1, dfp);
    fread(p_ycoord + offset, sizeof(int), local_coord, dfp);
    if (swapdata)
        swap_float(p_ycoord + offset, local_coord);
    fread(&dummy2, sizeof(int), 1, dfp);
    if (dummy1 != dummy2)
    {
        Covise::sendError("ERROR wrong FORTRAN block marker reading vector");
        return false;
    }
    /*--------  z-Daten ----------*/
    fread(&dummy1, sizeof(int), 1, dfp);
    fread(p_zcoord + offset, sizeof(int), local_coord, dfp);
    if (swapdata)
        swap_float(p_zcoord + offset, local_coord);
    fread(&dummy2, sizeof(int), 1, dfp);
    if (dummy1 != dummy2)
    {
        Covise::sendError("ERROR wrong FORTRAN block marker reading vector");
        return false;
    }

    return true;
}

/**************************************************************************\ 
 **  skip_block : skips one block of binary data, can be used to move to  **
 **               another information field within binary file            **
\**************************************************************************/
bool ReadIVDdata::skip_block(FILE *dfp)
{
    int dummy1, dummy2; // will hold fortran block marker
    fread(&dummy1, sizeof(int), 1, dfp);
    if (swapdata)
        swap_int(dummy1);
    fseek(dfp, dummy1, SEEK_CUR);
    fread(&dummy2, sizeof(int), 1, dfp);
    if (swapdata)
        swap_int(dummy2);

    if (dummy1 != dummy2)
    {
        Covise::sendError("ERROR wrong FORTRAN block marker reading scalar. Take al look at byteswapping!!");
        return false;
    }
    return true;
}

MODULE_MAIN(IO, ReadIVDdata)
