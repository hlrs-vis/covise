/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/***********************************************************************
 *									*
 *          								*
 *              Computer Centre University of Stuttgart			*
 *                         Allmandring 30a				*
 *                       D-70550 Stuttgart				*
 *                            Germany					*
 *									*
 *									*
 ************************************************************************/

#define ProgrammName "Generic ASCII-File Reader for N3S 3.2"

#define Kurzname "ReadN3s"

#define Copyright "(c) 2000 RUS Rechenzentrum der Uni Stuttgart"

#define Autor "M. Wierse (SGI)"

#define letzteAenderung "13.3.2000"

/************************************************************************/

#include <util/coviseCompat.h>
#include "ReadN3s.h"

#define VERBOSE

void ReadN3s::param(const char *name, bool inMapLoading)
{
    static char *resFileNameTmp = NULL;

    cerr << "\n ------- Parameter Callback for '"
         << name
         << "'" << endl;

    if (strcmp(name, resFileParam->getName()) == 0)
    {
        cerr << "begin scanning\n";

        resfilename = resFileParam->getValue();
        if (resfilename == NULL)
            return;

        if (resFileNameTmp == NULL)
        {
            resFileNameTmp = new char[strlen(resfilename) + 1];
            strcpy(resFileNameTmp, resfilename);
        }
        else
        {
            if (strcmp(resFileNameTmp, resfilename) == 0)
                return;
            else
            {
                delete[] resFileNameTmp;
                resFileNameTmp = new char[strlen(resfilename) + 1];
                strcpy(resFileNameTmp, resfilename);
            }
        }

        strcpy(infobuf, "Opening file ");
        strcat(infobuf, resfilename);
        sendInfo(infobuf);

        if ((fpres = fopen((char *)resfilename, "r")) == NULL)
        {
            strcpy(infobuf, "ERROR: Can't open file >> ");
            strcat(infobuf, resfilename);
            sendError(infobuf);
            return;
        }
        else
        {
            scanResultFile();
            fclose(fpres);
            return;
        }
    }
}

ReadN3s::ReadN3s(int argc, char *argv[])
{
    char buf[255], buf1[255];
    char buf2[255];
    int i;
    char *geo_data_path;
    char *res_data_path;
    set_module_description("Generic ASCII-File Reader for N3S 3.2");

    // the output port
    unsgridPort = addOutputPort("mesh", "coDoUnstructuredGrid", "geometry ");

    for (i = 0; i < MAX_PORTS_N3S; i++)
    {
        sprintf(buf, "dataport%d", i + 1);
        dataPort[i] = addOutputPort(buf, "coDoVec3 | coDoFloat | DO_Unstructured_V2D_Data | coDoVec2", buf);
    }
    // select the OBJ file name with a file browser
    geoFileParam = addFileBrowserParam("n3s geofile", "N3S geofile");
    resFileParam = addFileBrowserParam("n3s result file", "N3S result file");
    char *cov_path = getenv("COVISEDIR");
    geo_data_path = new char[255];
    res_data_path = new char[255];
    if (cov_path)
    {
        sprintf(geo_data_path, "%s/data/n3s/* ", cov_path);
        sprintf(res_data_path, "%s/data/n3s/* ", cov_path);
    }
    else
    {
        sprintf(geo_data_path, "./* ");
        sprintf(res_data_path, "./* ");
    }
    cerr << "buf: " << geo_data_path << endl;
    geoFileParam->setValue(geo_data_path, "geom");
    resFileParam->setValue(res_data_path, "post.res");
    //    resFileParam->setValue("post.res result", buf);

    return;

    choice_of_data[0] = new char[10]; // die gibt's immer
    strcpy(choice_of_data[0], "(none)\n");
    for (i = 0; i < MAX_PORTS_N3S; i++)
    {
        sprintf(buf1, "data%d", i);
        sprintf(buf2, "select data%d", i);
        choiceData[i] = addChoiceParam(buf1, buf2);
        choiceData[i]->setValue(1, choice_of_data, 1);
    }
}

ReadN3s::~ReadN3s()
{
}

void ReadN3s::quit(void *)
{
}

void ReadN3s::compute(void *)
{

    // get the file name
    geofilename = geoFileParam->getValue();
    resfilename = resFileParam->getValue();
    if (geofilename != NULL || resfilename != NULL)
    {
        // open the file
        if (openFiles())
        {
            sprintf(infobuf, "Files %s %s open", geofilename, resfilename);
            sendInfo(infobuf);

            // read the file, create the lists and create a COVISE unstructured grid object

            readGeoFile();
            fclose(fpgeo);

            readResultFile();
            fclose(fpres);
        }
        else
        {
            sprintf(infobuf, "Error opening files %s %s", geofilename, resfilename);
            sendError(infobuf);
            return;
        }
    }
    else
    {
        sendError("ERROR: fileName for geo or result file is NULL");
    }
}

void ReadN3s::createCoviseUnsgrd()

{
    // get the COVISE output object name from the controller

    unsgridObjectName = (char *)unsgridPort->getObjName();
    unsgridObject = new coDoUnstructuredGrid(unsgridObjectName, num_elements, num_elements * (dim + 1), num_nodes, 1);
    unsgridObject->getTypeList(&tl);
    unsgridObject->getAddresses(&el, &vl, &x, &y, &z);
    unsgridPort->setCurrentObject(unsgridObject);

    if (dim == 3)
        sprintf(infobuf, "found %d coordinates, %d tetrahedra", num_nodes, num_elements);
    else
        sprintf(infobuf, "found %d coordinates, %d triangles", num_nodes, num_elements);
    sendInfo(infobuf);
}

void
ReadN3s::scanResultFile()
{
    char buffer[MAXLINE];
    int i, j, k, dummy, end_of_file, first, velocity_data_available, timesteps, number_of_values, timestep;
    int number_of_choices;
    float time;
    char counting[4];
    ;

    // scanning the result file to get  num_elements, num_nodes_P2 and num_nodes
    // scanning the result file to get data available and position in the result file

    for (i = 0; i < 6; i++)
        fgets(buffer, MAXLINE, fpres);
    sscanf(buffer, "        %d %d %d %d %d %d %d %d %d",
           &dummy,
           &num_elements, &dim, &num_nodes, &num_nodes_P2,
           &dummy, &dummy, &dummy, &dummy);
#ifdef VERBOSE
    cout << dim << num_elements << " " << num_nodes_P2 << " " << num_nodes << "\n" << flush;
#endif
    if (dim == 1)
    {
        dim = 2;
        cout << "axisymmetrical case !!!! "
             << "\n" << flush;
    }

    end_of_file = 0;
    first = 1;
    while (!end_of_file)
    {

        if (first)
        {

            // get the names of variables which are stored, we assume they are the same for the different time steps

            fgets(buffer, MAXLINE, fpres);
            sscanf(buffer, "%s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s",
                   dnames[0], dnames[1], dnames[2], dnames[3], dnames[4],
                   dnames[5], dnames[6], dnames[7], dnames[8], dnames[9],
                   dnames[10], dnames[11], dnames[12], dnames[13], dnames[14],
                   dnames[15], dnames[16], dnames[17], dnames[18], dnames[19],
                   dnames[20], dnames[21], dnames[22], dnames[23], dnames[24]);

#ifdef VERBOSE
            cout << dnames[0] << " " << dnames[7] << " " << dnames[3] << " " << dnames[4] << endl;
#endif

            // get the information if variable is stored and where in the file

            fgets(buffer, MAXLINE, fpres);
            sscanf(buffer, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d",
                   &dset[0], &dset[1], &dset[2], &dset[3], &dset[4],
                   &dset[5], &dset[6], &dset[7], &dset[8], &dset[9],
                   &dset[10], &dset[11], &dset[12], &dset[13], &dset[14],
                   &dset[15], &dset[16], &dset[17], &dset[18], &dset[19],
                   &dset[20], &dset[21], &dset[22], &dset[23], &dset[24]);
#ifdef VERBOSE
            cout << buffer << endl;
#endif

            //check if velocity data is available

            velocity_components[0] = velocity_components[1] = velocity_components[2] = -1;

            for (i = 0; i < MAX_DATA_COMPONENTS; i++)
            {
                if (dset[i] != 0 && strncasecmp(dnames[i], "U", 1) == 0)
                    velocity_components[0] = i;
                if (dset[i] != 0 && strncasecmp(dnames[i], "V", 1) == 0)
                    velocity_components[1] = i;
                if (dset[i] != 0 && strncasecmp(dnames[i], "W", 1) == 0)
                    velocity_components[2] = i;
            }
            velocity_data_available = 0;
            if (dim == 3 && velocity_components[0] != -1 && velocity_components[1] != -1 && velocity_components[2] != -1)
                velocity_data_available = 1;
            if (dim == 2 && velocity_components[0] != -1 && velocity_components[1] != -1)
                velocity_data_available = 1;

            if (velocity_data_available)
                cout << " Vector data available " << endl;
            for (i = 0; i < 3; i++)
                fgets(buffer, MAXLINE, fpres);

            // create choice parameter

            number_of_choices = 0;
            for (i = 0; i < MAX_DATA_COMPONENTS; i++)
                if (dset[i] != 0)
                    number_of_choices++;

            if (number_of_choices != 0) // data available
            {
                number_of_choices = 1; // the first one is set to none!!
                for (i = 0; i < MAX_DATA_COMPONENTS; i++)
                    if (dset[i] != 0)
                    {
                        // +1 since choice_of_data has already none!
                        choice_of_data[number_of_choices] = new char[strlen(dnames[i]) + 2];
                        sprintf(choice_of_data[number_of_choices], "%s\n", dnames[i]);
                        mapping_of_choice_data[number_of_choices + 1] = i;
                        cout << " to choice param " << number_of_choices << dnames[i] << mapping_of_choice_data[number_of_choices] << endl;
                        number_of_choices++;
                    }
                if (velocity_data_available)
                {
                    choice_of_data[number_of_choices] = new char[strlen("velocity") + 2];
                    sprintf(choice_of_data[number_of_choices], "%s\n", "velocity");
                    mapping_of_choice_data[number_of_choices + 1] = MAX_DATA_COMPONENTS;
                    number_of_choices++;
                }
                for (k = 0; k < MAX_PORTS_N3S; k++)
                    choiceData[k]->setValue(number_of_choices, choice_of_data, 1);
            }

            first = 0;
        }
        else // skip this information , necessary ??????
        {
            for (i = 0; i < 11; i++)
                fgets(buffer, MAXLINE, fpres);
        }

        fgets(buffer, MAXLINE, fpres);
        // wohin mit timesteps!!
        sscanf(buffer, " %d %s %s", &timesteps, version, date);
        cout << "Timestep: " << timesteps << endl;
        cout << "Data created: " << date << endl;
        fgets(buffer, MAXLINE, fpres);
        fgets(buffer, MAXLINE, fpres);
        sscanf(buffer, " %f ", &time);
        // wohin mit time???
        cout << "Data calculated for time: " << time << endl;

        fgets(buffer, MAXLINE, fpres);
        timestep = 0;
        for (i = 0; i < MAX_DATA_COMPONENTS; i++)
            if (dset[i] != 0)
            {
                cout << " reading of component: " << dnames[i] << endl;

                fgets(buffer, MAXLINE, fpres);
                // get kind of node numbering
                sscanf(buffer + strlen(buffer) - 3, "%s", counting);

                if (strncasecmp(counting, "P2", 2) == 0)
                    number_of_values = num_nodes_P2;
                else if (strncasecmp(counting, "P1", 2) == 0)
                    number_of_values = num_nodes;
                else
                {
                    sprintf(infobuf, "Numbering not known %s ", counting);
                    sendError(infobuf);
                }
                fgets(buffer, MAXLINE, fpres); //-----
                position_in_file[timestep][i] = ftell(fpres);

#ifdef VERBOSE
                cout << counting << " " << strncasecmp(counting, "P2", 2) << " " << number_of_values << endl;
#endif

                // how can I do this with fseek??
                for (j = 0; j < number_of_values / VALUES_IN_LINE + 1; j++)
                    fgets(buffer, MAXLINE, fpres);
            }
        end_of_file = 1;
    }
#ifdef VERBOSE
    cout << "Test of file positioning"
         << "\n" << endl;
    for (i = 0; i < MAX_DATA_COMPONENTS; i++)
    {
        fseek(fpres, position_in_file[timestep][i], SEEK_SET);
        fgets(buffer, MAXLINE, fpres);
        cout << dnames[i] << ": " << buffer << "\n" << endl;
    }
#endif

    // timedependent!!
}

void
ReadN3s::readGeoFile()
{
    char buffer[MAXLINE]; // line in an obj file
    int i, k, j, dummy, end_of_file, node_number, element_number, v[4];
    long int dummyl;
    int length, number_of_edges, elem_type;

    createCoviseUnsgrd();

    end_of_file = 0;

    while (!end_of_file)
    {
        fgets(buffer, MAXLINE, fpgeo);
#ifdef DEEPVERBOSE
        cout << " string compare: " << strncasecmp(buffer, "CSECTION:", 9) << " " << buffer << flush;
#endif
        if (strncasecmp(buffer, "CSECTION:", 9) != 0)
            continue;
#ifdef DEEPVERBOSE
        cout << " 2. string compare: " << strncasecmp(buffer, "CSECTION:   NODES", 17) << " " << buffer << flush;
#endif
        if (strncasecmp(buffer, "CSECTION:   NODES", 17) == 0)
        { // reading of coordinates

            fgets(buffer, MAXLINE, fpgeo);
            for (i = 0; i < num_nodes; i++)
            {
                fgets(buffer, MAXLINE, fpgeo);
                sscanf(buffer, "        %d %d %d %d %d %d %d %d %d %d %ld",
                       &node_number,
                       &dummy, &dummy, &dummy, &dummy, &dummy,
                       &dummy, &dummy, &dummy, &dummy, &dummyl);
                length = strlen(buffer) - 1;
                sscanf(buffer + length - DIGITALS_OF_COORD, "%f", &(z[node_number - 1]));
                buffer[length - DIGITALS_OF_COORD] = '\0';
                sscanf(buffer + length - 2 * DIGITALS_OF_COORD, "%f", &(y[node_number - 1]));
                buffer[length - 2 * DIGITALS_OF_COORD] = '\0';
                sscanf(buffer + length - 3 * DIGITALS_OF_COORD, "%f", &(x[node_number - 1]));

#ifdef VERBOSE
                if (i < 10)
                    cout << node_number << " coord: " << x[node_number - 1] << " " << y[node_number - 1] << " " << z[node_number - 1] << "\n" << flush;
#endif
            }
        }
        if (strncasecmp(buffer, "CSECTION:   V ELEMENT", 21) == 0)
        { //  Reading of connectivity

            fgets(buffer, MAXLINE, fpgeo);
            switch (dim)
            {
            case 2:
                number_of_edges = EDGES_2D;
                elem_type = TYPE_TRIANGLE;
                break;
            case 3:
                number_of_edges = EDGES_3D;
                elem_type = TYPE_TETRAHEDER;
                break;
            default:
                cout << "problems with dimension" << dim << endl;
                exit(1);
            }

            for (k = 0; k < num_elements; k++) // format(I8,I2,((dim+1)+number_of_edges)*I7)
            {
                fgets(buffer, MAXLINE, fpgeo);

                // skip mid edge points
                for (i = (dim + 1) + number_of_edges - 1; i >= (dim + 1); i--)
                {
                    sscanf(buffer + i * I7 + I8 + I2, "%d", &dummy);
                    buffer[I8 + I2 + i * I7] = '\0';
                }

                for (i = dim; i >= 0; i--) // reading connectivity
                {
                    sscanf(buffer + I8 + I2 + i * I7, "%d", &v[i]);
                    buffer[I8 + I2 + i * I7] = '\0';
#ifdef DEEP_VERBOSE
                    cout << "vert" << v[i] << endl;
#endif
                }
                sscanf(buffer + I8, "%d", &dummy);
                buffer[I8] = '\0';
                sscanf(buffer, "%d", &element_number);
                for (j = 0; j < dim + 1; j++)
                    vl[k * (dim + 1) + j] = v[j] - 1;
                el[k] = k * (dim + 1);
                tl[k] = elem_type;
#ifdef DEEP_VERBOSE
                cout << "elem" << element_number << endl;
#endif
            }
        }
        if (strncasecmp(buffer, "CSECTION:   END", 15) == 0)
            end_of_file = 1;
    }
}

void
ReadN3s::readResultFile()
{
    char buffer[MAXLINE]; // line in an obj file
    int i, j, k, l, count, rest, timestep, position, choice_value;
    float *u, *v, *w, *value, dummy, *s;

    fseek(fpres, 0L, SEEK_SET);

    timestep = 0;
    for (i = 0; i < MAX_PORTS_N3S; i++)
    {
        choice_value = choiceData[i]->getValue();
        if (choice_value == 1)
            continue;
        cout << " reading of component: " << dnames[mapping_of_choice_data[choice_value]] << " " << choice_value << endl;
        // RAUS!
        choice_value = mapping_of_choice_data[choice_value];
        if (choice_value == MAX_DATA_COMPONENTS) // velocity is required
        {
            cout << "Reading velocity data" << endl;

            VdataName = dataPort[i]->getObjName();
            coDoVec3 *VdataObject = new coDoVec3((char *)VdataName, num_nodes);
            dataPort[i]->setCurrentObject(VdataObject);
            VdataObject->getAddresses(&u, &v, &w);

            for (l = 0; l < 3; l++) // velocity components
            {

                switch (l)
                {
                case 0:
                    value = u;
                    position = position_in_file[timestep][velocity_components[0]];
                    break;
                case 1:
                    value = v;
                    position = position_in_file[timestep][velocity_components[1]];
                    break;
                case 2:
                    if (dim == 3)
                    {
                        value = w;
                        position = position_in_file[timestep][velocity_components[2]];
                    }
                    else
                    {
                        for (k = 0; k < num_nodes; k++)
                            w[k] = 0.;
                        continue;
                    }
                    break;
                }

                fseek(fpres, position, SEEK_SET);

                // only get num_nodes values and neglect the others from the num_nodes_P2 numbering
                for (j = 0; j < num_nodes / VALUES_IN_LINE; j++)
                {
                    fgets(buffer, MAXLINE, fpres);
                    for (k = VALUES_IN_LINE - 1; k >= 0; k--)
                    {
                        sscanf(buffer + k * DIGITALS_OF_VALUE, "%f", value + VALUES_IN_LINE * j + k);
                        buffer[k * DIGITALS_OF_VALUE] = '\0';
                    }
                }
                count = num_nodes / VALUES_IN_LINE * VALUES_IN_LINE;
                rest = num_nodes % VALUES_IN_LINE;
                cout << "rest: " << rest << endl;

                fgets(buffer, MAXLINE, fpres);
                // get the rest of the line
                for (k = strlen(buffer) / DIGITALS_OF_VALUE - 1; k >= 0; k--)
                {
                    if (k >= rest)
                        sscanf(buffer + k * DIGITALS_OF_VALUE, "%f", &dummy);
                    else
                        sscanf(buffer + k * DIGITALS_OF_VALUE, "%f", value + count + k);
                    buffer[k * DIGITALS_OF_VALUE] = '\0';
#ifdef VERBOSE
                    cout << *(value + count + k) << " " << k * DIGITALS_OF_VALUE << " " << buffer << "\n" << flush;
#endif
                }
            } // l
        }
        else
        {
            SdataName = dataPort[i]->getObjName();
            coDoFloat *SdataObject = new coDoFloat((char *)SdataName, num_nodes);
            dataPort[i]->setCurrentObject(SdataObject);
            SdataObject->getAddress(&s);

            fseek(fpres, position_in_file[timestep][choice_value], SEEK_SET);

            for (j = 0; j < num_nodes / VALUES_IN_LINE; j++) // only get num_nodes vlaues and neglect the others from the num_nodes_P2 numbering
            {
                fgets(buffer, MAXLINE, fpres);
                for (k = VALUES_IN_LINE - 1; k >= 0; k--)
                {
                    sscanf(buffer + k * DIGITALS_OF_VALUE, "%f", s + VALUES_IN_LINE * j + k);
                    buffer[k * DIGITALS_OF_VALUE] = '\0';
                }
            }
            count = num_nodes / VALUES_IN_LINE * VALUES_IN_LINE;
            rest = num_nodes % VALUES_IN_LINE;
            fgets(buffer, MAXLINE, fpres);
            // get the rest of the line
            for (k = strlen(buffer) / DIGITALS_OF_VALUE - 1; k >= 0; k--)
            {
                if (k >= rest)
                    sscanf(buffer + k * DIGITALS_OF_VALUE, "%f", &dummy);
                else
                    sscanf(buffer + k * DIGITALS_OF_VALUE, "%f", s + count + k);
                buffer[k * DIGITALS_OF_VALUE] = '\0';
                cout << *(s + count + k) << " " << k * DIGITALS_OF_VALUE << " " << buffer << "\n" << flush;
            }
        }
    } // ports
}

int ReadN3s::openFiles()
{

    strcpy(infobuf, "Opening file ");
    strcat(infobuf, geofilename);
    strcpy(infobuf, "Opening file ");
    strcat(infobuf, resfilename);

    sendInfo(infobuf);

    // open the obj file
    if ((fpgeo = fopen((char *)geofilename, "r")) == NULL)
    {
        strcpy(infobuf, "ERROR: Can't open file >> ");
        strcat(infobuf, geofilename);
        sendError(infobuf);
        return (FALSE);
    }

    if ((fpres = fopen((char *)resfilename, "r")) == NULL)
    {
        strcpy(infobuf, "ERROR: Can't open file >> ");
        strcat(infobuf, resfilename);
        sendError(infobuf);
        return (FALSE);
    }
    else
    {
        return (TRUE);
    }
}

MODULE_MAIN(Reader, ReadN3s)
