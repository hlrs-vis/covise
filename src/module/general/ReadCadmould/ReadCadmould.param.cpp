/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/coTypes.h>
#include "ReadCadmould.h"
#include "CadmouldGrid.h"
#include "CadmouldData.h"
#include "FuellDruckData.h"
#include "MultiFuellData.h"
#include "FillingData.h"
#include "CarFiles.h"
#include "CarData.h"
#include "MultiCarData.h"
#include <config/CoviseConfig.h>
#ifdef WIN32
#define MAXPATHLEN 2048
#endif

//////////////////////////////////////////////////////////////////////////////////////////

void ReadCadmould::createParam()
{
    // file browser parameter
    p_filename = addFileBrowserParam("filename", "file name of Fuellbild or .cfe file");
    p_filename->setValue("data/nofile", "?????;*.cfe");
    // p_filename->setValue("data/Kunden/faurecia/CADMOULD-Test/nofile","?????");//

    // 3 grid ports : stationary, transient, filling
    p_mesh = addOutputPort("stMesh", "UnstructuredGrid", "stationary mesh");
    p_stepMesh = addOutputPort("trMesh", "UnstructuredGrid", "transient mesh");
    p_thick = addOutputPort("thick", "Float", "thickness of elements");

    // the output ports and choices
    // Loop for data fields: choices and ports
    char name[32];
    const char *defaultChoice[] = { "---" };
    for (int i = 0; i < NUM_PORTS; i++)
    {
        sprintf(name, "Choice_%d", i);
        p_choice[i] = addChoiceParam(name, "Select data for port");
        p_choice[i]->setValue(1, defaultChoice, 0);

        sprintf(name, "Data_%d", i);
        p_data[i] = addOutputPort(name, "Float|IntArr", name);
    }

    p_no_time_steps = addInt32Param("fillTimeStep", "time steps for filling");
    p_no_time_steps->setValue(25);

    const char *defaultFill[] = { "automatic" };
    p_fillField = addChoiceParam("fillField", "Select field for filling");
    p_fillField->setValue(1, defaultFill, 0);

    p_no_data_color = addStringParam("noDataColor", "RGBA color for non-filled elements");

    p_no_data_color->setValue("0xd0d0d0ff");

    //   p_byteswap = addBooleanParam("byteSwapping","byte_swapping");
    p_fillMesh = addOutputPort("fiMesh", "UnstructuredGrid", "mesh for filling");
    p_fillData = addOutputPort("fiValuw", "Float", "data for filling");
}

//////////////////////////////////////////////////////////////////////////////////////////

void ReadCadmould::postInst()
{
    // (unused) int bswap=coCoviseConfig::getInt("Module.ReadCadmould.ByteSwapping",1);
    // p_byteswap->setValue(bswap);

    p_filename->show();
    p_choice[0]->show();
    p_choice[1]->show();
    p_choice[2]->show();
}

//////////////////////////////////////////////////////////////////////////////////////////
//       || (   0==strcmp(paramName,p_byteswap->getName())

void ReadCadmould::param(const char *paramName, bool inMapLoading)
{
    if ((0 == strcmp(paramName, p_filename->getName())
                  //           || 0==strcmp(paramName,p_byteswap->getName())
         )
        && !inMapLoading)
    {
        if (checkFiles(p_filename->getValue()))
        {
            openFiles();
            if (retOpenFile == -1) // apparently incorrect byte-swapping
            {
                byteswap_ = !byteswap_;
                openFiles();
            }
        }
    }

    if (strstr(paramName, "Choice_") && !inMapLoading && retOpenFile == 0)
    {
        int number;
        if (sscanf(paramName, "Choice_%d", &number) != 1)
        {
            fprintf(stderr, "ReadCadmould::param: sscanf failed\n");
        }
        // cout << p_choice[number]->getActLabel() <<endl;
        p_data[number]->setInfo(p_choice[number]->getActLabel());
    }
}

//////////////////////////////////////////////////////////////////////////////////////////

bool ReadCadmould::checkFiles(const std::string &filename)
{
    // need at least two chars (should be 5)
    if (filename.length() < 2)
    {
        sendError("Illegal Filename: %s", filename.c_str());
        return false;
    }

    // make sure we can work with the file name
    char buffer[MAXPATHLEN + 1];
    strncpy(buffer, filename.c_str(), MAXPATHLEN);
    buffer[MAXPATHLEN] = '\0';

    // try to open the "Füllbild" file - this is the one we select
    FILE *fi = fopen(filename.c_str(), "r");
    if (!fi)
    {
        sendWarning("Fuellbild %s: %s", filename.c_str(), strerror(errno));
        return false;
    }
    fclose(fi);

    setCaseType(buffer);

    // try to open the "Mesh" file - Fuellbild except last char
    if (d_type != TYPE_CAR)
    {
        buffer[strlen(buffer) - 1] = '\0';
    }

    fi = fopen(buffer, "r");
    if (!fi)
    {
        sendError("Mesh %s: %s", buffer, strerror(errno));
        return false;
    }
    fclose(fi);

    return true;
}

void
ReadCadmould::setCaseType(const std::string &filename)
{
    if (filename.rfind(".cfe") == filename.length() - 4)
        d_type = TYPE_CAR;
    else
        d_type = TYPE_STD;
}

//////////////////////////////////////////////////////////////////////////////////////////
/// Open the grid and all identifiable data files

void
ReadCadmould::openFiles()
{
    retOpenFile = 0;
    // cout << "p_byteswap: "<<p_byteswap->getValue()<<endl;

    /// shut down all old if exist
    delete d_grid;
    for (int i = 0; i < d_numDataSets; i++)
    {
        delete d_data[i];
        d_data[i] = NULL;
    }
    d_numDataSets = 0;

    // the filename parameter
    const char *selFile = p_filename->getValue();

    // make sure we can work with the file name
    char basename[MAXPATHLEN + 1], filename[MAXPATHLEN + 16];

    //// Open mesh file: clip off last char if old style
    strncpy(basename, selFile, MAXPATHLEN);
    basename[MAXPATHLEN] = '\0';

    setCaseType(basename);

    if (d_type != TYPE_CAR)
    {
        basename[strlen(basename) - 1] = '\0';
    }

    d_grid = new CadmouldGrid(basename);
    int gridState = -1;
    int gridNoVert = -1;
    if (d_grid)
        gridState = d_grid->getState();
    if (gridState != 0)
    {
        if (gridState < 0)
            sendError("Could not read %s as Cadmould mesh", basename);

        else
            sendError("Could not read %s as Cadmould mesh: %s", basename, strerror(gridState));
        delete d_grid;
        d_grid = NULL;
    }
    else
    {
        gridNoVert = d_grid->getNumVert();
    }

    // from now on: base is Fuellbild file ( case+Kenner )
    strncpy(basename, selFile, MAXPATHLEN);
    basename[MAXPATHLEN] = '\0';

    if (d_type == TYPE_CAR)
    {
        char car_base[MAXPATHLEN];
        strcpy(car_base, basename);
        *(strstr(car_base, ".cfe")) = '\0';

        CarFiles files(car_base);

        for (int grp_nb = 0; grp_nb < files.numGroups(); grp_nb++)
        {

            MultiCarData *data = new MultiCarData(files, grp_nb, byteswap_);
            if (data)
            {
                if (data->getState() == 0)
                {
                    d_data[d_numDataSets] = data;
                    d_numDataSets++;
                }
                else
                {
                    delete data;
                }
            }
        }
    }

    else if (d_type == TYPE_STD)
    {
        // check, whether we have a E0 file = Rheologie-File
        {
            sprintf(filename, "%sE0", basename);
            static const char *labelsE0[] = {
                "Fliessfront-Temperatur",
                "Fliessfront-Druckbedarf",
                "Fliessfront-Schubspannung",
                "Fliessfront-Geschwindigkeit"
            };
            FuellDruckData *data = new FuellDruckData(filename, 4, labelsE0, byteswap_);
            if (data)
            {
                if (data->getState() == 0)
                {
                    d_data[d_numDataSets] = data;
                    d_numDataSets++;
                }
                else
                    delete data;
            }
        }

        // check, whether we have E## files = Zwischenergebnisse
        {
            sprintf(filename, "%sE", basename);
            static const char *labelsExx[] = {
                "Momentan-Temperatur(t)",
                "Momentan-Druck(t)",
                "Momentan-Schubspannung(t)",
                "Momentan-Geschwindigkeit(t)"
            };
            MultiFuellData *data = new MultiFuellData(filename, 4, labelsExx, byteswap_);
            if (data)
            {
                if (data->getState() == 0)
                {
                    d_data[d_numDataSets] = data;
                    d_numDataSets++;
                }
                else
                    delete data;
            }
        }

        // check, whether we have S## files = Schichtergebnisse
        {
            sprintf(filename, "%sS", basename);
            static const char *labelsSxx[] = { "Randschichtddicke(t)" };
            MultiFuellData *data = new MultiFuellData(filename, 1, labelsSxx, byteswap_);
            if (data)
            {
                if (data->getState() == 0)
                {
                    d_data[d_numDataSets] = data;
                    d_numDataSets++;
                }
                else
                    delete data;
            }
        }

        //////// other data sets here ...

        // filling process, file: basename
        // data: Fuellstand, connect, fill_time
        {
            static const char *labelsFilling[] = { "Fuellstand", "Connect", "fill_time" };
            static const CadmouldData::FieldType fieldType[] = {
                CadmouldData::SCALAR_FLOAT,
                CadmouldData::SCALAR_INT,
                CadmouldData::SCALAR_FLOAT
            };
            FillingData *data = new FillingData(basename, 3, labelsFilling, fieldType, byteswap_);
            fillTimeDataSets = -1;
            fillChoice = 0;

            if (data)
            {
                if (data->getState() == 0)
                {
                    fillTimeDataSets = d_numDataSets;
                    fillField = 0;
                    d_data[d_numDataSets] = data;
                    d_numDataSets++;
                    if (gridNoVert != -1 && data->numVert() != ((gridNoVert + 3) / 4) * 4)
                    {
                        retOpenFile = -1;
                        sendWarning("Number of nodes from mesh and fill-data does not match. Trying with the contrary byte-swapping option");
                    }
                }
                else
                    delete data;
            }
        }

        // check, whether we have .car files
        {
            CarFiles files(basename);
            for (int i = 0; i < files.numFiles(); i++)
            {
                CarData *data = new CarData(files.get(i), byteswap_);
                if (data)
                {
                    if (data->getState() == 0)
                    {
                        d_data[d_numDataSets] = data;
                        d_numDataSets++;
                    }
                    else
                        delete data;
                }
            }
        }
    }

    if (retOpenFile == 0)
    {
        // create the labels for the choices
        int numLabels = 1; // the "---" label
        for (int i = 0; i < d_numDataSets; i++)
            numLabels += d_data[i]->numFields();
        const char **labels = new const char *[numLabels];
        labels[0] = "---";
        int labelNo = 1;

        for (int i = 0; i < d_numDataSets; i++)
        {
            for (int j = 0; j < d_data[i]->numFields(); j++)
            {

                labels[labelNo] = d_data[i]->getName(j);
                if (fillChoice == 0 && strstr(labels[labelNo], "Fuellzeit") != NULL)
                {
                    fillChoice = labelNo;
                }
                else if (fillChoice == 0 && strstr(labels[labelNo], "llzeit"))
                {
                    fillChoice = labelNo;
                }
                // +1 because choices return 1 for 1st label ([0])
                dataLoc[labelNo].datasetNo = i;
                dataLoc[labelNo].fieldNo = j;
                labelNo++;
            }
        }

        // attach to all choices, keep old values if valid
        for (int i = 0; i < NUM_PORTS; i++)
        {
            int oldVal = p_choice[i]->getValue();
            //if (oldVal<d_numDataSets-2)
            if (oldVal < numLabels)
                p_choice[i]->setValue(numLabels, labels, oldVal);
            else
                p_choice[i]->setValue(numLabels, labels, 0);
        }

        labels[0] = "automatic";
        p_fillField->setValue(numLabels, labels, fillChoice);
    }
}
