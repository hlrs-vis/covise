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

#define ProgrammName "Generic ASCII-File Reader for Radioss"

#define Kurzname "ReadRadioss"

#define Copyright "(c) 2000 RUS Rechenzentrum der Uni Stuttgart"

#define Autor "M. Wierse (SGI)"

#define letzteAenderung "23.4.2000"

/************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include "ReadRadioss.h"

//#define VERBOSE

void main(int argc, char *argv[])
{
    ReadRadioss *application = new ReadRadioss();
    application->start(argc, argv);
}

void ReadRadioss::param(const char *name)
{
}

ReadRadioss::ReadRadioss()
{
    char buf[255];
    int i;
    set_module_description("Generic ASCII-File Reader for Radioss");

    // the output port
    polygonsPort = addOutputPort("polygons", "DO_Polygon", "solid shell ");
    unsgridPort = addOutputPort("unsgrid", "coDoUnstructuredGrid", "solid shell ");
    linesPort = addOutputPort("lines", "DO_Line", "truss spring beam");
    polygonsShell3nQuadPort = addOutputPort("shell3n", "DO_Polygon", "shell3n quad");

    for (i = 0; i < MAX_PORTS_RADIOSS; i++)
    {
        if (i == 0)
        {
            sprintf(buf, "displacements");
            dataPort[i] = addOutputPort(buf, "coDoVec3", buf);
        }
        else
        {
            sprintf(buf, "scalar value %d", i);
            dataPort[i] = addOutputPort(buf, "coDoFloat", buf);
        }
    }
    // select the OBJ file name with a file browser
    geoFileParam = addFileBrowserParam("Radioss geofile 000", "Radioss geofile 000");
    resFileParam = addFileBrowserParam("Radioss result file 001", "Radioss result file 001");
    inputFileParam = addFileBrowserParam("Radioss input file", "Radioss input file");
}

ReadRadioss::~ReadRadioss()
{
    delete xglobal;
    delete yglobal;
    delete zglobal;
    delete[] solidslist;
    delete[] quadslist;
    delete[] shellslist;
    delete[] trussslist;
    delete[] beamslist;
    delete[] springslist;
    delete[] shell3nslist;
    delete polys.corner_list;
    delete polys.polygon_list;
    delete polys.x;
    delete polys.y;
    delete polys.z;
    delete lines.corner_list;
    delete lines.line_list;
    delete lines.x;
    delete lines.y;
    delete lines.z;
    delete global_to_local;
    delete local_to_global;
    delete mapping_propid_to_subset;
    delete collect_subsetids;
    int i;
    for (i = 0; i < MAX_PORTS_RADIOSS + 2; i++)
    {
        delete Data[i];
    }
    delete displacementsfield[0];
    delete displacementsfield[1];
    delete displacementsfield[2];
}

void ReadRadioss::quit(void)
{
}

int ReadRadioss::compute(void)
{
    // get the file name

    geofilename = geoFileParam->getValue();
    inputfilename = inputFileParam->getValue();
    resfilename = resFileParam->getValue();

    /*
   #ifdef OLDFILES
      inputfilename = new char[strlen("/usr2/people/andreas/covise_maus/data/Radioss/REAL_VI2.RAD.codage") + 1];
       geofilename = new char[strlen("/usr2/people/andreas/covise_maus/data/Radioss/REAL_VI2Y000.RAD.codage") + 1];
       resfilename = new char[strlen("/usr2/people/andreas/covise_maus/data/Radioss/REAL_VI2Y001.RAD.codage") + 1];
   #else
       inputfilename = new char[strlen("/usr2/people/andreas/covise_maus/data/Radioss/R21V4_3D.RAD") + 1];
       geofilename = new char[strlen("/usr2/people/andreas/covise_maus/data/Radioss/R21V4Y000_3D.RAD") + 1];
       resfilename = new char[strlen("/usr2/people/andreas/covise_maus/data/Radioss/R21V4Y001_3D.RAD") + 1];
   #endif
   #ifdef OLDFILES
   sprintf((char *)inputfilename,"/usr2/people/andreas/covise_maus/data/Radioss/REAL_VI2.RAD.codage") ;
   sprintf((char *)geofilename,"/usr2/people/andreas/covise_maus/data/Radioss/REAL_VI2Y000.RAD.codage") ;
   sprintf((char *)resfilename,"/usr2/people/andreas/covise_maus/data/Radioss/REAL_VI2Y001.RAD.codage") ;
   #else
   sprintf((char *)inputfilename,"/usr2/people/andreas/covise_maus/data/Radioss/R21V4_3D.RAD") ;
   sprintf((char *)geofilename,"/usr2/people/andreas/covise_maus/data/Radioss/R21V4Y000_3D.RAD") ;
   sprintf((char *)resfilename,"/usr2/people/andreas/covise_maus/data/Radioss/R21V4Y001_3D.RAD") ;
   #endif
   */

    if (geofilename == NULL || inputfilename == NULL || resfilename == NULL)
    {
        sprintf(infobuf, "Input file for Radioss , geo file and result file has to be specified %s %s %s", geofilename, inputfilename, resfilename);
        sendError(infobuf);
        return FAIL;
    }

    if (openFiles())
    {
        int i;
        sprintf(infobuf, "Files %s %s open", geofilename, resfilename);
        sendInfo(infobuf);

        // read the file, create the lists and create a set of COVISE polygon object

        readGeoFile();
        fclose(fpgeo);

        readInputFile();
        fclose(fpinput);

        for (i = 0; i < 3; i++)
            Data[i] = new float[numnod]; // displacements
        for (i = 3; i < MAX_PORTS_RADIOSS + 2; i++) //  3 scalars
            Data[i] = new float[numsol + numshel];

        readResultFile();
        fclose(fpres);

        global_to_local = new int[numnod]; // where the index is gone in the new numbering
        local_to_global = new int[numnod]; // to what the new index corresponds originally
        polys.corner_list = new int[numsol * 6 * 4 + numquad * 4 + numshel * 4 + numsh3n * 3];
        polys.polygon_list = new int[numsol * 6 + numquad + numshel + numsh3n + 1];
        displacementsfield[0] = new float[numnod];
        displacementsfield[1] = new float[numnod];
        displacementsfield[2] = new float[numnod];

        createPolygon();
        createCoviseUnsgrid();
        createScalarDataSubsets();
        createPolygonShell3nQuad(); // not added to the polygons in general since no data will be available!

        lines.corner_list = new int[numtrus * 2 + numbeam * 2 + numspri * 2];
        lines.line_list = new int[numtrus + numbeam + numspri + 1];

        createLine();
    }
    else
    {
        sprintf(infobuf, "Error opening files %s %s %s", geofilename, resfilename, inputfilename);
        sendError(infobuf);
        return FAIL;
    }
}

void ReadRadioss::splitSolid(int *sysnod)
{
    int j, k;

    for (j = 0; j < 6; j++)
    {
        for (k = 0; k < 4; k++)
            polys.corner_list[polys.num_corners + k] = sysnod[split[j][k]];
        polys.num_corners += 4;
        polys.num_polygons++;
        polys.polygon_list[polys.num_polygons] = polys.num_corners;
    }
}

void ReadRadioss::createCovisePolygon(int subset)
{
    // get the COVISE output object name from the controller

    polygonsObjectName = (char *)polygonsPort->getObjName();
    polygonsObject = new coDoPolygons(polygonsObjectName, polys.num_points, polys.x, polys.y, polys.z, polys.num_corners,
                                      polys.corner_list, polys.num_polygons, polys.polygon_list);
    if (polygonsObject->objectOk() == 0)
        cerr << "Something wrong\n";
    polygonsObject->addAttribute("vertexOrder", "2");
    sprintf(infobuf, "found %d coordinates, %d polygons %d corners", polys.num_points, polys.num_polygons, polys.num_corners);
    sendInfo(infobuf);
}

void ReadRadioss::createCovisePolygonShell3nQuad(int subset)
{
    // get the COVISE output object name from the controller

    polygonsShell3nQuadObjectName = (char *)polygonsShell3nQuadPort->getObjName();
    sprintf(buffer, "%s_%d", polygonsShell3nQuadObjectName, subset);
#ifdef VERBOSE
    cout << "polygonsShell3nQuadObjectName" << buffer << endl;
#endif
    polygonsShell3nQuadObject = new coDoPolygons(buffer, polys.num_points, polys.x, polys.y, polys.z, polys.num_corners,
                                                 polys.corner_list, polys.num_polygons, polys.polygon_list);
    if (polygonsShell3nQuadObject->objectOk() == 0)
        cerr << "Something wrong\n";
    polygonsShell3nQuadObject->addAttribute("vertexOrder", "2");
    sprintf(infobuf, "found %d coordinates, %d polygons %d corners", polys.num_points, polys.num_polygons, polys.num_corners);
    sendInfo(infobuf);
}

void ReadRadioss::createCoviseLine(int subset)
{
    // get the COVISE output object name from the controller

    linesObjectName = (char *)linesPort->getObjName();
    sprintf(buffer, "%s_%d", linesObjectName, subset);
#ifdef VERBOSE
    cout << "linesObjectName" << buffer << endl;
#endif
    linesObject = new coDoLines(buffer, lines.num_points, lines.x, lines.y, lines.z, lines.num_corners, lines.corner_list, lines.num_lines, lines.line_list);
    sprintf(infobuf, "found %d coordinates, %d lines %d corners", lines.num_points, lines.num_lines, lines.num_corners);
    sendInfo(infobuf);
}

void ReadRadioss::createCoviseV3D(int subset)
{
    VdataName = dataPort[0]->getObjName();
    coDoVec3 *VdataObject = new coDoVec3(VdataName, polys.num_points, displacementsfield[0],
                                         displacementsfield[1], displacementsfield[2]);
    sprintf(infobuf, "V3D build with %d coordinates", polys.num_points);
    sendInfo(infobuf);
}

void ReadRadioss::createDisplacementsDataSubsets(int num_points)
{
    int i;

    // create Sets for the displacement component according the subsets for the polygons!
    // compressNodes must be run before to have the correct values in the

    for (i = 0; i < num_points; i++)
    {
        displacementsfield[0][i] = Data[0][local_to_global[i]];
        displacementsfield[1][i] = Data[1][local_to_global[i]];
        displacementsfield[2][i] = Data[2][local_to_global[i]];
        /*
           if(i< 10)
            cout << displacementsfield[1][i] ;
      */
    }
}

void ReadRadioss::createPolygon()
{
    int j, i, k;
    polys.num_corners = 0;
    polys.num_polygons = 0;
    polys.polygon_list[polys.num_polygons] = polys.num_corners;
    for (i = 0; i < numsol; i++)
    {
        splitSolid(solidslist[i].sysnod);
    }
    for (i = 0; i < numshel; i++)
    {
        for (k = 0; k < 4; k++)
            polys.corner_list[polys.num_corners + k] = shellslist[i].sysnod[k];
        polys.num_corners += 4;
        polys.num_polygons++;
        polys.polygon_list[polys.num_polygons] = polys.num_corners;
    }
    if (polys.num_corners > (numsol * 6 * 4 + numquad * 4 + numshel * 4 + numsh3n * 3) || polys.num_polygons > (numsol * 6 + numquad + numshel + numsh3n))
    {
        cout << polys.num_corners << " " << numsol * 6 * 4 + numquad * 4 + numshel * 4 + numsh3n * 3
             << " " << polys.num_polygons << " " << numsol * 6 + numquad + numshel + numsh3n << endl;
        exit(1);
    }

    polys.num_points = compressNodes(polys.num_corners, polys.corner_list, &polys.x, &polys.y, &polys.z);
    createDisplacementsDataSubsets(polys.num_points);
    createCovisePolygon(1);
    createCoviseV3D(1);
    polygonsPort->setCurrentObject(polygonsObject);
    dataPort[0]->setCurrentObject(VdataObject);
}

void ReadRadioss::createCoviseUnsgrid()
{
    int i;
    // get the COVISE output object name from the controller
    int *tl = new int[polys.num_polygons];
    for (i = 0; i < polys.num_polygons; i++)
        if ((polys.polygon_list[i + 1] - polys.polygon_list[i]) == 3)
            tl[i] = TYPE_TRIANGLE;
        else
            tl[i] = TYPE_QUAD;
    char *unsgridObjectName = (char *)unsgridPort->getObjName();
    coDoUnstructuredGrid *unsgridObject = new coDoUnstructuredGrid(unsgridObjectName, polys.num_polygons, polys.num_corners, polys.num_points, polys.polygon_list, polys.corner_list,
                                                                   polys.x, polys.y, polys.z, tl);
    if (unsgridObject->objectOk() == 0)
        cerr << "Something wrong\n";
    unsgridObject->addAttribute("vertexOrder", "2");
    sprintf(infobuf, "found %d coordinates, %d polygons %d corners", polys.num_points, polys.num_polygons, polys.num_corners);
    sendInfo(infobuf);
}

void ReadRadioss::createPolygonShell3nQuad()
{
    int j, i, k, collect_for_subset;
    coDistributedObject **subSetsPolygons = new coDistributedObject *[num_subsets + 1];
    int collect_for_subset_compressed = 0;

    for (j = 0; j < numpid; j++)
    {

        polys.num_corners = 0;
        polys.num_polygons = 0;
        polys.polygon_list[polys.num_polygons] = polys.num_corners;
        if (collect_subsetids[j] != -1)
            collect_for_subset = collect_subsetids[j];
        else
            continue;

        for (i = 0; i < numsh3n; i++)
        {
            if (mapping_propid_to_subset[shell3nslist[i].syspid] != collect_for_subset)
                continue;
            for (k = 0; k < 3; k++)
                polys.corner_list[polys.num_corners + k] = shell3nslist[i].sysnod[k];
            polys.num_corners += 3;
            polys.num_polygons++;
            polys.polygon_list[polys.num_polygons] = polys.num_corners;
        }

        if (polys.num_corners > (numsol * 6 * 4 + numquad * 4 + numshel * 4 + numsh3n * 3) || polys.num_polygons > (numsol * 6 + numquad + numshel + numsh3n))
        {
            cout << polys.num_corners << " " << numsol * 6 * 4 + numquad * 4 + numshel * 4 + numsh3n * 3
                 << " " << polys.num_polygons << " " << numsol * 6 + numquad + numshel + numsh3n << endl;
            exit(1);
        }
        if (polys.num_polygons != 0)
        {
            polys.num_points = compressNodes(polys.num_corners, polys.corner_list, &polys.x, &polys.y, &polys.z);
        }
        else
            continue;
        createCovisePolygonShell3nQuad(collect_for_subset);
        subSetsPolygons[collect_for_subset_compressed] = polygonsShell3nQuadObject;
        collect_for_subset_compressed++;
    }
    subSetsPolygons[collect_for_subset_compressed] = NULL;
    polygonsShell3nQuadObjectName = (char *)polygonsShell3nQuadPort->getObjName();
    coDoSet *dset = new coDoSet(polygonsShell3nQuadObjectName, subSetsPolygons);
    //  dset->addAttribute("POLYGON SUBSET",buffer) ;  // WAS in Buffer??
    polygonsShell3nQuadPort->setCurrentObject(dset);

    delete[] subSetsPolygons;
}

void ReadRadioss::createLine()
{
    int j, i, k, collect_for_subset;
    coDistributedObject **subSetsLines = new coDistributedObject *[num_subsets + 1];
    int collect_for_subset_compressed = 0;

    for (j = 0; j < numpid; j++)
    {
        if (collect_subsetids[j] != -1)
            collect_for_subset = collect_subsetids[j];
        else
            continue;
        lines.num_corners = 0;
        lines.num_lines = 0;
        lines.line_list[lines.num_lines] = lines.num_corners;
        for (i = 0; i < numtrus; i++)
        {
            if (mapping_propid_to_subset[trussslist[i].syspid] != collect_for_subset)
                continue;
            for (k = 0; k < 2; k++)
                lines.corner_list[lines.num_corners + k] = trussslist[i].sysnod[k];
            lines.num_corners += 2;
            lines.num_lines++;
            lines.line_list[lines.num_lines] = lines.num_corners;
        }
        for (i = 0; i < numbeam; i++)
        {
            if (mapping_propid_to_subset[beamslist[i].syspid] != collect_for_subset)
                continue;
            for (k = 0; k < 2; k++)
                lines.corner_list[lines.num_corners + k] = beamslist[i].sysnod[k];
            lines.num_corners += 2;
            lines.num_lines++;
            lines.line_list[lines.num_lines] = lines.num_corners;
        }
        for (i = 0; i < numspri; i++)
        {
            if (mapping_propid_to_subset[springslist[i].syspid] != collect_for_subset)
                continue;
            for (k = 0; k < 2; k++)
                lines.corner_list[lines.num_corners + k] = springslist[i].sysnod[k];
            lines.num_corners += 2;
            lines.num_lines++;
            lines.line_list[lines.num_lines] = lines.num_corners;
        }
        if (lines.num_corners > (numtrus * 2 + numbeam * 2 + numspri * 2) || lines.num_lines > (numtrus + numbeam + numspri))
        {
            cout << lines.num_corners << " " << numtrus * 2 + numbeam * 2 + numspri * 2
                 << " " << lines.num_lines << " " << numtrus + numbeam + numspri << endl;
            exit(1);
        }
        if (lines.num_lines != 0)
            lines.num_points = compressNodes(lines.num_corners, lines.corner_list, &lines.x, &lines.y, &lines.z);
        else
            continue;
        createCoviseLine(collect_for_subset);
        subSetsLines[collect_for_subset_compressed] = linesObject;
        collect_for_subset_compressed++;
    }
    subSetsLines[collect_for_subset_compressed] = NULL;
    linesObjectName = (char *)linesPort->getObjName();
    coDoSet *dset = new coDoSet(linesObjectName, subSetsLines);
    //  dset->addAttribute("LINE SUBSET",buffer) ;  // WAS in Buffer??
    linesPort->setCurrentObject(dset);

    delete[] subSetsLines;
}

int ReadRadioss::compressNodes(int num_corners, int *corner_list, float **xp, float **yp, float **zp)
{
    int i, num_points;
    float *x, *y, *z;

    for (i = 0; i < numnod; i++)
    {
        global_to_local[i] = -1;
        local_to_global[i] = -1;
    }
    num_points = 0;
    for (i = 0; i < num_corners; i++)
        if (global_to_local[corner_list[i]] == -1) //not set yet
        {
            global_to_local[corner_list[i]] = num_points;
            local_to_global[num_points] = corner_list[i];
            corner_list[i] = num_points;
            num_points++;
        }
        else
            corner_list[i] = global_to_local[corner_list[i]];

    *xp = new float[num_points];
    *yp = new float[num_points];
    *zp = new float[num_points];
    x = *xp;
    y = *yp;
    z = *zp;
    for (i = 0; i < num_points; i++)
    {
        x[i] = xglobal[local_to_global[i]];
        y[i] = yglobal[local_to_global[i]];
        z[i] = zglobal[local_to_global[i]];
    }
    return num_points;
}

void ReadRadioss::readGeoFile()
{

    int end_of_file;

    end_of_file = 0;

    while (!end_of_file)
    {
        fgets(buffer, MAXLINE, fpgeo);
        if (strncasecmp(buffer, "/ENDDATA", 8) == 0)
        {
            end_of_file = 1;
            continue;
        }
        if (strncasecmp(buffer, "/HEAD", 5) == 0)
        {
            readHead();
            continue;
        }
        if (strncasecmp(buffer, "/CONTROL", 8) == 0)
        {
            readControl();
            continue;
        }
        if (strncasecmp(buffer, "/NODE", 5) == 0)
        {
            readCoordinates();
            continue;
        }
        if (strncasecmp(buffer, "/SOLIDE", 6) == 0)
        {
            readSolids();
            continue;
        }
        if (strncasecmp(buffer, "/QUAD", 5) == 0)
        {
            readQuad();
            continue;
        }
        if (strncasecmp(buffer, "/SHELL3N", 8) == 0)
        {
            readShell3n();
            continue;
        }
        if (strncasecmp(buffer, "/SHELL", 6) == 0)
        {
            readShell();
            continue;
        }
        if (strncasecmp(buffer, "/TRUSS", 6) == 0)
        {
            readTruss(); // ??
            continue;
        }
        if (strncasecmp(buffer, "/BEAM", 5) == 0)
        {
            readBeam();
            continue;
        }
        if (strncasecmp(buffer, "/SPRING", 7) == 0)
        {
            readSpring();
            continue;
        }
    }
}

void ReadRadioss::readInputFile()
{ // out of the input file only get the mapping from the propid of the elements
    // to the subsets

    int end_of_file, subset_id, mat_id, part_id, i;
    mapping_propid_to_subset = new int[numpid];
    collect_subsetids = new int[numpid];
    end_of_file = 0;

    for (i = 0; i < numpid; i++)
        collect_subsetids[i] = -1;

    num_subsets = 0;
    while (!end_of_file)
    {
        if (fgets(buffer, MAXLINE, fpinput) == NULL)
        {
            end_of_file = 1;
            continue;
        }
        if (strncasecmp(buffer, "/ENDDATA", 8) == 0)
        {
            end_of_file = 1;
            continue;
        }
        if (strncasecmp(buffer, "/PART/", 6) == 0)
        {
            fgets(buffer, MAXLINE, fpinput);
            subset_id = getLastI8();
            mat_id = getLastI8();
            part_id = getLastI8();
            if (part_id > numpid)
            {
                sprintf(infobuf, "Input file for Radioss and Geo file as to be specified %s %s opened", geofilename, inputfilename);
                sendError(infobuf);
            }
            mapping_propid_to_subset[part_id - 1] = subset_id - 1;
            if (collect_subsetids[subset_id - 1] == -1)
            {
                collect_subsetids[subset_id - 1] = subset_id - 1;
                num_subsets++;
            }
            if (part_id == numpid)
            {

                // stopping reading since we are not sure that /ENDDATA is existing

                end_of_file = 1;
                continue;
            }
        }
    }
    sprintf(infobuf, "found %d subsets", num_subsets);
    sendInfo(infobuf);
}

void
ReadRadioss::getDisplacements()
{
    int i;
    for (i = 0; i < 3; i++)
        fgets(buffer, MAXLINE, fpres);

    for (i = 0; i < numnod; i++) //#FORMAT: (I8,1P3E16.9)
    {
        fgets(buffer, MAXLINE, fpres);
        if (buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';

        Data[2][i] = getLastE16() - zglobal[i];
        Data[1][i] = getLastE16() - yglobal[i];
        Data[0][i] = getLastE16() - xglobal[i];
    }
}

void
ReadRadioss::getScalarData(int scalar, int elements, int offset)
{
    float value;
    int i, j, rest;

    for (i = 0; i < 2; i++)
        fgets(buffer, MAXLINE, fpres);

    for (i = 0; i < elements / 6; i++) //#FORMAT: (I8,1P3E16.9)#FORMAT: (1P6E12.5) (VAR(I),I=1,NUMSOL)
    {
        fgets(buffer, MAXLINE, fpres);
        if (buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';
        for (j = 5; j >= 0; j--)
        {
            value = getLastE12();
            Data[scalar][offset + i * 6 + j] = value;
        }
    }

    // get the last line which is may be not completely full
    fgets(buffer, MAXLINE, fpres);
    if (buffer[strlen(buffer) - 1] == '\n')
        buffer[strlen(buffer) - 1] = '\0';
    rest = elements % 6;
    for (j = rest - 1; j >= 0; j--)
    {
        value = getLastE12();
        Data[scalar][offset + elements - rest + j] = value;
    }
#ifdef VERBOSE
    for (i = offset; i < offset + elements; i++)
        cout << "scalar values for shell " << Data[scalar][i] << endl;
#endif
}

void ReadRadioss::createScalarDataSubsets()
{
    float *scalarfield;
    int i, j, k, scalar;

    scalarfield = new float[numsol * 6 + numshel];
    for (scalar = 0; scalar < 3; scalar++) // create Set for each scalar component according the subsets for the polygons!
    {
        int num_polygons = 0;

        for (i = 0; i < numsol; i++)
        {
            for (k = 0; k < 6; k++)
            {
                scalarfield[num_polygons + k] = Data[scalar + 3][i];
            }
            num_polygons += 6;
        }
        for (i = 0; i < numshel; i++)
        {
            scalarfield[num_polygons] = Data[scalar + 3][numsol + i];
            num_polygons++;
        }

        if (num_polygons != 0)
        {
            // the first is for the displacements
            SdataName = (char *)dataPort[scalar + 1]->getObjName();
            coDoFloat *SdataObject = new coDoFloat(SdataName, num_polygons, scalarfield);
        }

        dataPort[scalar + 1]->setCurrentObject(SdataObject);

    } // scalar

    delete scalarfield;
}

void
ReadRadioss::readResultFile() // we assume the data is stored as in the example file
{
    int end_of_file = 0;
    int scalar = 3;
    int first = 1;

    while (!end_of_file)
    {
        fgets(buffer, MAXLINE, fpres);

        if (strncasecmp(buffer, "/ENDDATA", 8) == 0)
        {
            end_of_file = 1;
            continue;
        }
        if (strncasecmp(buffer, "/NODAL", 6) == 0)
        {
            getDisplacements();
        }
        if (strncasecmp(buffer, "/SOLID", 6) == 0)
        {
            getScalarData(scalar, numsol, 0);
            scalar++;
        }
        if (strncasecmp(buffer, "/SHELL", 6) == 0)
        {
            if (first)
            {
                scalar = 3;
                first = 0;
            }
            getScalarData(scalar, numshel, numsol);
            scalar++;
        }
    }
}

int ReadRadioss::openFiles()
{

    strcpy(infobuf, "Opening file ");
    strcat(infobuf, geofilename);
    strcpy(infobuf, "Opening file ");
    strcat(infobuf, resfilename);

    sendInfo(infobuf);

    // open the geo file resulting after the calculation with RADIOSS
    if ((fpgeo = fopen((char *)geofilename, "r")) == NULL)
    {
        strcpy(infobuf, "ERROR: Can't open file >> ");
        strcat(infobuf, geofilename);
        sendError(infobuf);
        return (FALSE);
    }

    // open the input file to RADIOSS to get the mapiing from propid to subset id

    if ((fpinput = fopen((char *)inputfilename, "r")) == NULL)
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

void ReadRadioss::readHead()
{
    char dummy[255];
#ifdef VERBOSE
    cout << "reading of Head " << endl;
#endif
    fgets(buffer, MAXLINE, fpgeo);
    sscanf(buffer, " %s %s %s", casename, dummy, date);
#ifdef VERBOSE
    cout << date << " " << casename << "\n" << endl;
#endif
}

int ReadRadioss::getLastI8()
{
    int length, intvalue;
    length = strlen(buffer);
    sscanf(buffer + length - I8, "%d", &intvalue);
    buffer[length - I8] = '\0';
    return (intvalue);
}

float ReadRadioss::getLastE16()
{
    int length;
    float floatvalue;
    length = strlen(buffer);
    sscanf(buffer + length - E16, "%f", &floatvalue);
    buffer[length - E16] = '\0';
    return (floatvalue);
}

float ReadRadioss::getLastE12()
{
    int length;
    float floatvalue;
    length = strlen(buffer);
    sscanf(buffer + length - E12, "%f", &floatvalue);
    buffer[length - E12] = '\0';
    return (floatvalue);
}

void ReadRadioss::readControl()
{
    int i;
#ifdef VERBOSE
    cout << "reading of Control " << endl;
#endif
    for (i = 0; i < 3; i++)
        fgets(buffer, MAXLINE, fpgeo);

    fgets(buffer, MAXLINE, fpgeo);
    numnod = getLastI8();
    numpid = getLastI8();
    nummid = getLastI8();

    for (i = 0; i < 2; i++)
        fgets(buffer, MAXLINE, fpgeo);

    fgets(buffer, MAXLINE, fpgeo);

    numsh3n = getLastI8();
    numspri = getLastI8();
    numbeam = getLastI8();
    numtrus = getLastI8();
    numshel = getLastI8();
    numquad = getLastI8();
    numsol = getLastI8();
    cout << "Numnod " << numnod << " numsh3n " << numsh3n << endl;
}

void ReadRadioss::readCoordinates()
{
    float x, y, z, mass;
    int i, node_number;

#ifdef VERBOSE
    cout << "reading of Coordinates " << endl;
#endif
    for (i = 0; i < 3; i++)
        fgets(buffer, MAXLINE, fpgeo);

    xglobal = new float[numnod];
    yglobal = new float[numnod];
    zglobal = new float[numnod];

    for (i = 0; i < numnod; i++) // #FORMAT: (2I8,1P4E16.9)
    {
        fgets(buffer, MAXLINE, fpgeo);
        if (buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';
        mass = getLastE16();
        z = getLastE16();
        y = getLastE16();
        x = getLastE16();
        node_number = getLastI8();
        xglobal[node_number - 1] = x;
        yglobal[node_number - 1] = y;
        zglobal[node_number - 1] = z;
#ifdef VERBOSE
        if (i < 10)
            cout << node_number << " coord: " << xglobal[node_number - 1] << " " << yglobal[node_number - 1] << " " << zglobal[node_number - 1] << "\n" << flush;
#endif
    }
}

void ReadRadioss::readSolids()
{
    int i, j;
#ifdef VERBOSE
    cout << "reading of Solids " << endl;
#endif
    for (i = 0; i < 4; i++)
        fgets(buffer, MAXLINE, fpgeo);

    solidslist = new Solids[numsol];
    for (i = 0; i < numsol; i++)
    {

        /*#FORMAT: (4I8/8X,8I8)
      # SYSSOL  USRSOL  SYSMID  SYSPID
      #SYSNOD1 SYSNOD2 SYSNOD3 SYSNOD4 SYSNOD5 SYSNOD6 SYSNOD7 SYSNOD8 */

        fgets(buffer, MAXLINE, fpgeo);
        if (buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';
        solidslist[i].syspid = getLastI8() - 1;
        solidslist[i].sysmid = getLastI8();
        solidslist[i].usrsol = getLastI8() - 1;
        solidslist[i].syssol = getLastI8() - 1;
        fgets(buffer, MAXLINE, fpgeo);
        for (j = 7; j >= 0; j--)
            solidslist[i].sysnod[j] = getLastI8() - 1;
    }
#ifdef VERBOSE
    for (i = 0; i < 1; i++)
    {
        cout << solidslist[i].syspid << endl;
        cout << solidslist[i].sysmid << endl;
        cout << solidslist[i].usrsol << endl;
        cout << solidslist[i].syssol << endl;
        for (j = 0; j < 8; j++)
            cout << solidslist[i].sysnod[j] << endl;
    }
#endif
}

void ReadRadioss::readQuad()
{

    int i, j;
#ifdef VERBOSE
    cout << "reading of Quad " << endl;
#endif

    for (i = 0; i < 3; i++)
        fgets(buffer, MAXLINE, fpgeo);

    quadslist = new Quads[numquad];
    for (i = 0; i < numquad; i++)
    {

        /*/QUAD
      2d Solid Elements
      #FORMAT: (8I8)
      #SYSQUAD USRQUAD  SYSMID  SYSPID SYSNOD1 SYSNOD2 SYSNOD3 SYSNOD4 */

        fgets(buffer, MAXLINE, fpgeo);
        if (buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';
        for (j = 3; j >= 0; j--)
            quadslist[i].sysnod[j] = getLastI8() - 1;
        quadslist[i].syspid = getLastI8() - 1;
        quadslist[i].sysmid = getLastI8();
        quadslist[i].usrquad = getLastI8() - 1;
        quadslist[i].sysquad = getLastI8() - 1;
    }
#ifdef VERBOSE
    for (i = 0; i < numquad; i++)
    {
        cout << quadslist[i].syspid << endl;
        cout << quadslist[i].sysmid << endl;
        cout << quadslist[i].usrquad << endl;
        cout << quadslist[i].sysquad << endl;
        for (j = 0; j < 4; j++)
            cout << quadslist[i].sysnod[j] << endl;
    }
#endif
}

void ReadRadioss::readShell()
{
    int i, j;
#ifdef VERBOSE
    cout << "reading of Shell " << endl;
#endif

    /*/SHELL
   3d Shell Elements
   #FORMAT: (8I8)
   #SYSSHEL USRSHEL  SYSMID  SYSPID SYSNOD1 SYSNOD2 SYSNOD3 SYSNOD4
    */

    for (i = 0; i < 3; i++)
        fgets(buffer, MAXLINE, fpgeo);

    shellslist = new Shells[numshel];
    for (i = 0; i < numshel; i++)
    {
        fgets(buffer, MAXLINE, fpgeo);
        if (buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';
        for (j = 3; j >= 0; j--)
            shellslist[i].sysnod[j] = getLastI8() - 1;
        shellslist[i].syspid = getLastI8() - 1;
        shellslist[i].sysmid = getLastI8();
        shellslist[i].usrshel = getLastI8() - 1;
        shellslist[i].sysshel = getLastI8() - 1;
    }
#ifdef VERBOSE
    for (i = 0; i < numshel; i++)
    {
        /* cout <<  shellslist[i].syspid << endl ;
       cout <<  shellslist[i].sysmid << endl ;
       cout <<  shellslist[i].usrshel << endl ;
       cout <<  shellslist[i].sysshel << endl ; */
        for (j = 0; j < 4; j++)
            cout << shellslist[i].sysnod[j] << endl;
    }
#endif
}

void ReadRadioss::readTruss()
{
    int i, j;
#ifdef VERBOSE
    cout << "reading of Truss " << endl;
#endif

    /*
   /TRUSS
   3d Truss Elements
   #FORMAT: (6I8)
   #SYSTRUS USRTRUS  SYSMID  SYSPID SYSNOD1 SYSNOD2
    */

    for (i = 0; i < 3; i++)
        fgets(buffer, MAXLINE, fpgeo);

    trussslist = new Trusss[numtrus];
    for (i = 0; i < numtrus; i++)
    {
        fgets(buffer, MAXLINE, fpgeo);
        if (buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';
        for (j = 1; j >= 0; j--)
            trussslist[i].sysnod[j] = getLastI8() - 1;
        trussslist[i].syspid = getLastI8() - 1;
        trussslist[i].sysmid = getLastI8();
        trussslist[i].usrtrus = getLastI8() - 1;
        trussslist[i].systrus = getLastI8() - 1;
    }
#ifdef VERBOSE
    for (i = 0; i < 1; i++)
    {
        cout << trussslist[i].syspid << endl;
        cout << trussslist[i].sysmid << endl;
        cout << trussslist[i].usrtrus << endl;
        cout << trussslist[i].systrus << endl;
        for (j = 0; j < 2; j++)
            cout << trussslist[i].sysnod[j] << endl;
    }
#endif
}

void ReadRadioss::readBeam()
{

    int i, j, dummy;
#ifdef VERBOSE
    cout << "reading of Beam " << endl;
#endif

    /*
   /BEAM
   3d Beam Elements
   #FORMAT: (7I8)
   #SYSBEAM USRBEAM  SYSMID  SYSPID SYSNOD1 SYSNOD2 SYSNOD3
    */

    for (i = 0; i < 3; i++)
        fgets(buffer, MAXLINE, fpgeo);

    beamslist = new Beams[numbeam];
    for (i = 0; i < numbeam; i++)
    {
        fgets(buffer, MAXLINE, fpgeo);
        if (buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';
        dummy = getLastI8();
        for (j = 1; j >= 0; j--)
            beamslist[i].sysnod[j] = getLastI8() - 1;
        beamslist[i].syspid = getLastI8() - 1;
        beamslist[i].sysmid = getLastI8();
        beamslist[i].usrbeam = getLastI8() - 1;
        beamslist[i].sysbeam = getLastI8() - 1;
    }
#ifdef VERBOSE
    for (i = 0; i < 1; i++)
    {
        cout << beamslist[i].syspid << endl;
        cout << beamslist[i].sysmid << endl;
        cout << beamslist[i].usrbeam << endl;
        cout << beamslist[i].sysbeam << endl;
        for (j = 0; j < 2; j++)
            cout << beamslist[i].sysnod[j] << endl;
    }
#endif
}

void ReadRadioss::readSpring()
{

    int i, j;
#ifdef VERBOSE
    cout << "reading of Spring " << endl;
#endif

    /*
   /SPRING
   3d Spring Elements
   #FORMAT: (6I8)
   #SYSSPRI USRSPRI  SYSMID  SYSPID SYSNOD1 SYSNOD2
   */

    for (i = 0; i < 3; i++)
        fgets(buffer, MAXLINE, fpgeo);

    springslist = new Springs[numspri];
    for (i = 0; i < numspri; i++)
    {
        fgets(buffer, MAXLINE, fpgeo);
        if (buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';
        for (j = 1; j >= 0; j--)
            springslist[i].sysnod[j] = getLastI8() - 1;
        springslist[i].syspid = getLastI8() - 1;
        springslist[i].sysmid = getLastI8();
        springslist[i].usrspri = getLastI8() - 1;
        springslist[i].sysspri = getLastI8() - 1;
    }
#ifdef VERBOSE
    for (i = 0; i < 1; i++)
    {
        cout << springslist[i].syspid << endl;
        cout << springslist[i].sysmid << endl;
        cout << springslist[i].usrspri << endl;
        cout << springslist[i].sysspri << endl;
        for (j = 0; j < 2; j++)
            cout << springslist[i].sysnod[j] << endl;
    }
#endif
}

void ReadRadioss::readShell3n()
{

    int i, j;
#ifdef VERBOSE
    cout << "reading of Shell3n " << endl;
#endif

    /*
   /SHELL3N
   3d Shell Elements (Triangle)
   #FORMAT: (7I8)
   #SYSSH3N USRSH3N  SYSMID  SYSPID SYSNOD1 SYSNOD2 SYSNOD3
   */

    for (i = 0; i < 3; i++)
        fgets(buffer, MAXLINE, fpgeo);

    shell3nslist = new Shell3ns[numsh3n];
    for (i = 0; i < numsh3n; i++)
    {
        fgets(buffer, MAXLINE, fpgeo);
        if (buffer[strlen(buffer) - 1] == '\n')
            buffer[strlen(buffer) - 1] = '\0';
        for (j = 2; j >= 0; j--)
            shell3nslist[i].sysnod[j] = getLastI8() - 1;
        shell3nslist[i].syspid = getLastI8() - 1;
        shell3nslist[i].sysmid = getLastI8();
        shell3nslist[i].usrsh3n = getLastI8() - 1;
        shell3nslist[i].syssh3n = getLastI8() - 1;
    }
#ifdef VERBOSE
    for (i = 0; i < 1; i++)
    {
        cout << shell3nslist[i].syspid << endl;
        cout << shell3nslist[i].sysmid << endl;
        cout << shell3nslist[i].usrsh3n << endl;
        cout << shell3nslist[i].syssh3n << endl;
        for (j = 0; j < 3; j++)
            cout << shell3nslist[i].sysnod[j] << endl;
    }
#endif
}
