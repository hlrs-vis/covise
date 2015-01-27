/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// 16.11.2001 / 1 / file ReadIWSGeo.cpp

/******************************************************************************\ 
 **                                                              (C)2001 RUS **
 **                                                                          **
 ** Description:Reader for IWS data files                                    **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 **                                                                          **
 ** Author: M. Muench                                                        **
 **                                                                          **
 ** History:                                                                 **
 ** Someday 01 v1                                                            **
 ** XXXXXXXXX xx new covise api                                              **
 **                                                                          **
\******************************************************************************/

#include "ReadIWSGeo.h"

/********************\ 
 *                  *
 * Covise main loop *
 *                  *
\********************/

int main(int argc, char *argv[])
{
    ReadIWSGeo *application = new ReadIWSGeo();
    application->start(argc, argv);

    return 0;
}

/******************************\ 
 *                            *
 * Ingredients of Application *
 *                            *
\******************************/

ReadIWSGeo::ReadIWSGeo()
{
    // Parameters
    // create the parameters
    scaleWorld = addFloatParam("scale factor", "scale factor for VR-world scaling");
    // set the default values
    const float defaultP = 1.0;
    scaleWorld->setValue(defaultP);

    //char buffer[128];

    // this info appears in the module setup window
    set_module_description("IWS Reader / IWS format");

    // the output port2
    gridPort = addOutputPort("mesh", "coDoUnstructuredGrid", "Unstructured Grid");
    polyPort = addOutputPort("poly", "coDoPolygons", "Polygons");
    linePort = addOutputPort("line", "coDoLines", "Lines");

    gridDataPort = addOutputPort("gridData", "coDoFloat", "Grid Data");
    polyDataPort = addOutputPort("polyData", "coDoFloat", "Polygon Data");
    lineDataPort = addOutputPort("lineData", "coDoFloat", "Line Data");

    // select the DX file name with a file browser
    iwsFileParam = addFileBrowserParam("iwsFile", "IWS file");
    iwsFileParam->setValue("testdata/michael/IWS/", "*.geo");
}

ReadIWSGeo::~ReadIWSGeo()
{
}

void ReadIWSGeo::quit()
{
}

int ReadIWSGeo::compute()
{
    // retrieve parameters
    float scale_factor;
    scale_factor = scaleWorld->getValue();

    char infobuf[500]; // buffer for COVISE info and error messages

    // get the file name
    iwsFile = iwsFileParam->getValue();

    if (iwsFile != NULL)
    {
        // open the file
        if (iwsFp = openFile(iwsFile))
        {
            sprintf(infobuf, "File %s open", iwsFile);
            Covise::sendInfo(infobuf);

            // read the file, create the lists and create a COVISE Unstructured Grid object
            readFile(scale_factor);
            closeFile(iwsFile);

            sprintf(infobuf, "File %s closed", iwsFile);
            Covise::sendInfo(infobuf);
        }
        else
        {
            sprintf(infobuf, "Error opening file %s", iwsFile);
            Covise::sendError(infobuf);
            return FAIL;
        }
    }
    else
    {
        Covise::sendError("ERROR: iwsFile is NULL");
        return FAIL;
    }
    return SUCCESS;
}

/***************************************************************\ 
 *                                                             *
 * reads the file in ascii mode and creates the covise objects *
 *                                                             *
\***************************************************************/

FILE *ReadIWSGeo::openFile(const char *filename)
{
    char infobuf[3000];

    //strcpy(infobuf, "Opening file ");
    //strcat(infobuf, filename);
    //Covise::sendInfo(infobuf);

    // open the iws file
    if ((iwsFp = Covise::fopen(filename, "r")) == NULL)
    {
        strcpy(infobuf, "ERROR: Can't open file >> ");
        strcat(infobuf, filename);
        Covise::sendError(infobuf);
        return (NULL);
    }
    else
    {
        fprintf(stderr, "File %s open\n", filename);
        return (iwsFp);
    }
}

void ReadIWSGeo::closeFile(const char *filename)
{
    fclose(iwsFp);
    fprintf(stderr, "File %s closed\n", filename);
}

void ReadIWSGeo::readFile(float scale_factor)
{
    char fileLine[LINE_SIZE]; // line in an iws file
    char *key = NULL;
    //char buffer[100];

    //number of vertices, lines, faces, elements
    int vertexNumber = 0;
    int edgeNumber = 0;
    int cornerNumber = 0;
    int faceNumber = 0;
    int elementNumber = 0;

    //coordinates
    float *x_coord, *y_coord, *z_coord;

    //lines
    int *corner_list, *line_list;

    //polygons
    int *poly_id;
    int *polygon_list;
    int *polygon_corners = NULL; //for coDoPolygons
    int *edge_list;
    int num_corners; //for coDoPolygons

    //auxilliary variables
    int coord_index = 0;
    int corner_index = 0;
    int line_index = 0;
    int edge_index = 0;
    int polygon_index = 0;

    int old_size = 0;
    int new_size = 0;

    //elements
    int num_faces = 0;

    while ((fgets(fileLine, LINE_SIZE, iwsFp) != NULL))
    {
        char *line = &fileLine[0];

        //skip comments and blank lines
        if (skipLine(fileLine))
        {
            continue;
        }
        else
        {
        }

        if (strstr(line, "%%") != NULL)
        {
            scanHeader(line, &vertexNumber, &edgeNumber, &cornerNumber, &faceNumber, &elementNumber);
        }
        else
        {
        }

        //set key (Vertices, Edges, Faces, Elements or something senseless found
        if (testInput(line) != NULL)
        {
            key = testInput(line);
            printf("%s\n", key);
        }
        else
        {
        }

        //allocate coordinate lists
        if ((strstr(line, "% Vertices") != NULL) && (strstr(key, "Vertices") != NULL))
        {
            allocCoord(line, key, vertexNumber, &x_coord, &y_coord, &z_coord);
        }

        //allocate corner list and line list
        else if ((strstr(line, "% Edges") != NULL) && (strstr(key, "Edges") != NULL))
        {
            allocLines(line, key, cornerNumber, edgeNumber, &corner_list, &line_list);
        }

        //allocate polygon list and polygon id list
        else if ((strstr(line, "% Faces") != NULL) && (strstr(key, "Faces") != NULL))
        {
            allocPoly(line, key, faceNumber, &polygon_list, &poly_id);
            old_size = CHUNK_SIZE;
            edge_list = new int[old_size];
        }

        else
        {
        }

        //fill coordinate lists
        if ((key != NULL) && (strcmp(key, "Vertices") == 0) && (strstr(line, "% Vertices") == NULL) && (strstr(line, "$") == NULL))
        {
            sscanf(line, "%f %f %f", &(x_coord[coord_index]), &(y_coord[coord_index]), &(z_coord[coord_index]));

            //scale the VR-world's dimension  with scale_factor
            x_coord[coord_index] *= scale_factor;
            y_coord[coord_index] *= scale_factor;
            z_coord[coord_index] *= scale_factor;

            ++coord_index;
        }
        else
        {
        }

        //fill edge list and corner list
        if ((key != NULL) && (strcmp(key, "Edges") == 0) && (strstr(line, "% Edges") == NULL) && (strstr(line, "$") == NULL))
        {
            while (isspace(*line))
            {
                ++line;
            }
            while (!isspace(*line))
            {
                ++line;
            }

            sscanf(line, "%d %d", &(corner_list[corner_index]), &(corner_list[corner_index + 1]));
            line_list[line_index] = corner_index;
            ++corner_index;
            ++corner_index;
            ++line_index;
        }
        else
        {
        }

        //fill polygon list
        if ((key != NULL) && (strcmp(key, "Faces") == 0) && (strstr(line, "% Faces") == NULL) && (strstr(line, "$") == NULL))
        {
            //poly_id < 0 ? -> cleft, poly_id > 0 -> rest
            poly_id[polygon_index] = setPolyId(line, key);

            //set pointer to first index of the polygonal face
            while (isspace(*line))
            {
                ++line;
            }
            while (!isspace(*line))
            {
                ++line;
            }

            //now set polygon list and edge list
            edge_index = setPolygons(line, key, &old_size, &new_size, polygon_index, &polygon_list, edge_index, &edge_list);

            //the end of this line => increment polygon_index
            ++polygon_index;
        }

        else
        {
        }
    }

    //convert line index representation of the polygons (edge_list) to
    //a coordinate index representation of the polygons(corner_list)

    if (polygon_corners == NULL)
    {
        for (int i = 0; i < polygon_index - 1; i++)
        {
            if (poly_id[i] < 0)
            {
                num_corners += polygon_list[i + 1] - polygon_list[i];
            }
            else if (poly_id[i] > 0)
            {
                num_faces += polygon_list[i + 1] - polygon_list[i];
            }
            else
            {
            }
        }
        if (poly_id[polygon_index - 1] < 0)
        {
            num_corners += edge_index - polygon_list[polygon_index - 1];
        }
        else if (poly_id[polygon_index - 1] > 0)
        {
            num_faces += edge_index - polygon_list[polygon_index - 1];
        }
        else
        {
        }

        polygon_corners = new int[edge_index];
        cout << "\n\n" << flush << "edge_index = " << flush << edge_index << "\n\n" << flush;

        cout << "num_corners = " << flush << num_corners << "\n\n" << flush;

        cout << "num_faces = " << flush << num_faces << "\n\n" << flush;
    }

    line2coord(corner_list, edge_index, edge_list, polygon_index, polygon_list, &polygon_corners);

    cout << "VertexNumber = " << flush << vertexNumber << flush << '\n' << flush;
    cout << "EdgeNumber = " << flush << edgeNumber << flush << '\n' << flush;
    cout << "FaceNumber = " << flush << faceNumber << flush << '\n' << flush;
    cout << "ElementNumber = " << flush << elementNumber << flush << '\n' << flush;

    coDoLines *polylines = new coDoLines(linePort->getObjName(), vertexNumber, x_coord, y_coord, z_coord, cornerNumber, corner_list, edgeNumber, line_list);
    linePort->setCurrentObject(polylines);

    float *polygon_data = new float[vertexNumber];
    for (int i = 0; i < vertexNumber; i++)
    {
        polygon_data[i] = 10 * (x_coord[i]);
    }

    coDoFloat *lineData = new coDoFloat(lineDataPort->getObjName(), vertexNumber, polygon_data);
    lineDataPort->setCurrentObject(lineData);

    //boundary polygons should not be showed -> correct polygon list
    int *cleft_list;
    int cleft_size = 0;
    {
        int i;
        for (i = 0; i < faceNumber; i++)
        {
            if (poly_id[i] < 0)
            {
                ++cleft_size;
            }
            else
            {
            }
        }
        cleft_list = new int[cleft_size];
        int offset = faceNumber - cleft_size;
        for (i = 0; i < cleft_size; i++)
        {
            cleft_list[i] = polygon_list[i + offset] - num_faces;
        }
    }
    delete[] polygon_list;

    coDoPolygons *polygons = new coDoPolygons(polyPort->getObjName(), vertexNumber, x_coord, y_coord, z_coord, num_corners, (polygon_corners + num_faces), cleft_size, cleft_list);
    polyPort->setCurrentObject(polygons);

    // set the vertex order for twosided lighting in the renderer
    // 1=clockwise 2=counterclockwise
    // missing vertex order -> no twosided lighting (inner surface not lighted)
    // wrong vertex order -> wrong lighting for surfaces with normals
    polygons->addAttribute("vertexOrder", "2");

    //coordinates
    delete[] x_coord;
    delete[] y_coord;
    delete[] z_coord;

    //lines
    delete[] corner_list;
    delete[] line_list;

    //polygons
    delete[] cleft_list;
    delete[] poly_id;
    delete[] polygon_corners;
    delete[] edge_list;

    //elements

    //data
    delete[] polygon_data;
}

void ReadIWSGeo::scanHeader(char *line, int *nVert, int *nEdge, int *nCorner, int *nFace, int *nElem)
{
    //scan headlines of the file
    while (isspace(*line))
    {
        ++line;
    }

    if (strstr(line, "%%") != NULL)
    {
        if (strstr(line, "VertexNumber") != NULL)
        {
            while (!isdigit(*line))
            {
                ++line;
            }
            if (isWholeNumber(line) == true)
            {
                *nVert = atoi(line);
                //cout << "VertexNumber = " << flush << (*nVert) << flush << '\n' << flush;
            }
            else
            {
                cerr << "Fehler bei VertexNumber\n\n";
                return;
            }
        }

        else if (strstr(line, "EdgeNumber") != NULL)
        {
            while (!isdigit(*line))
            {
                ++line;
            }
            if (isWholeNumber(line) == true)
            {
                *nEdge = atoi(line);
                *nCorner = 2 * (*nEdge);
                //cout << "EdgeNumber = " << flush << (*nEdge) << flush << '\n' << flush;
            }
            else
            {
                cerr << "Fehler bei EdgeNumber\n\n";
                return;
            }
        }

        else if (strstr(line, "FaceNumber") != NULL)
        {
            while (!isdigit(*line))
            {
                ++line;
            }
            if (isWholeNumber(line) == true)
            {
                *nFace = atoi(line);
                //cout << "FaceNumber = " << flush << (*nFace) << flush << '\n' << flush;
            }
            else
            {
                cerr << "Fehler bei FaceNumber\n\n";
                return;
            }
        }

        else if (strstr(line, "ElementNumber") != NULL)
        {
            while (!isdigit(*line))
            {
                ++line;
            }
            if (isWholeNumber(line) == true)
            {
                *nElem = atoi(line);
                //cout << "ElementNumber = " << flush << (*nElem) << flush << '\n' << flush;
            }
            else
            {
                cerr << "Fehler bei ElementNumber\n\n";
                return;
            }
        }
        else
        {
        }
    }
}

char *ReadIWSGeo::testInput(char *line)
{
    char *qualifier = NULL;

    if (strstr(line, "%") != NULL)
    {
        if ((strstr(line, "% Vertices") != NULL) && (strstr(line, "NET") == NULL))
        {
            qualifier = "Vertices";
            return qualifier;
        }

        else if ((strstr(line, "% Edges") != NULL) && (strstr(line, "NET") == NULL))
        {
            qualifier = "Edges";
            return qualifier;
        }

        else if ((strstr(line, "% Faces") != NULL) && (strstr(line, "NET") == NULL))
        {
            qualifier = "Faces";
            return qualifier;
        }

        else if ((strstr(line, "% Elements") != NULL) && (strstr(line, "NET") == NULL))
        {
            qualifier = "Elements";
            return qualifier;
        }

        else
        {
            return NULL;
        }
    }
    else
    {
        return NULL;
    }

    return qualifier;
}

void ReadIWSGeo::allocCoord(char *line, char *key, int nVert, float **cx, float **cy, float **cz)
{
    if ((strstr(line, "Vertices") != NULL) && (strstr(key, "Vertices") != NULL))
    {
        (*cx) = new float[nVert];
        (*cy) = new float[nVert];
        (*cz) = new float[nVert];
        cout << "dimension of coordinate lists = " << flush << nVert << flush << "\n\n" << flush;
    }
    else
    {
    }
}

void ReadIWSGeo::allocLines(char *line, char *key, int nCorner, int nEdge, int **cl, int **ll)
{
    if ((strstr(line, "Edges") != NULL) && (strstr(key, "Edges") != NULL))
    {
        (*ll) = new int[nEdge];
        cout << "dimension of edge list = " << flush << nEdge << flush << "\n\n" << flush;

        (*cl) = new int[nCorner];
        cout << "dimension of corner list = " << flush << nCorner << flush << "\n\n" << flush;
    }
    else
    {
    }
}

void ReadIWSGeo::allocPoly(char *line, char *key, int nFace, int **pl, int **id)
{
    if ((strstr(line, "Faces") != NULL) && (strstr(key, "Faces") != NULL))
    {
        (*pl) = new int[nFace];
        cout << "dimension of polygon list = " << flush << nFace << flush << "\n\n" << flush;

        (*id) = new int[nFace];
        //cout << "dimension of polygon id list = " << flush << nCorner << flush << "\n\n" << flush;
    }
    else
    {
    }
}

int ReadIWSGeo::setPolyId(char *line, char *key)
{
    if ((key != NULL) && (strcmp(key, "Faces") == 0) && (strstr(line, "% Faces") == NULL) && (strstr(line, "$") == NULL))
    {
        while (isspace(*line))
        {
            ++line;
        }
        int variable = 0;
        sscanf(line, "%d", &variable);
        return variable;
    }
    else
    {
        cerr << "error: function should not have been called here!\n\n" << flush;
        return 0;
    }
    return 0;
}

int ReadIWSGeo::setPolygons(char *line, char *key, int *old_size, int *new_size, int polygon_index, int **polygon_list, int edge_index, int **edge_list)
{
    int num_edges = 0;
    int *tmp_edges;
    int *pl = (*polygon_list);
    int *el = (*edge_list);

    if ((key != NULL) && (strcmp(key, "Faces") == 0) && (strstr(line, "% Faces") == NULL) && (strstr(line, "$") == NULL))
    {
        pl[polygon_index] = edge_index;

        while (*line != '\0')
        {
            if (((edge_index + num_edges) < (*old_size)))
            {
                //enough memory
                el[edge_index + num_edges] = atoi(line);
                ++num_edges;

                //skip to next number
                while (isspace(*line))
                {
                    ++line;
                }
                while (!isspace(*line))
                {
                    ++line;
                }
                while (isspace(*line))
                {
                    ++line;
                }
            }
            else
            {
                //not enough memory
                (*new_size) = (*old_size) + CHUNK_SIZE;
                tmp_edges = new int[(*new_size)];
                memcpy(tmp_edges, el, (*old_size) * sizeof(int));
                delete[] el;
                el = tmp_edges;
                (*old_size) = (*new_size);

                el[edge_index + num_edges] = atoi(line);
                ++num_edges;

                //skip to next number
                while (isspace(*line))
                {
                    ++line;
                }
                while (!isspace(*line))
                {
                    ++line;
                }
                while (isspace(*line))
                {
                    ++line;
                }
            }
        }
    }
    else
    {
    }

    return (edge_index + num_edges);
}

void ReadIWSGeo::line2coord(int *corner_list, int edge_index, int *edge_list, int polygon_index, int *polygon_list, int **polygon_corners)
{
    int *cl = (*polygon_corners);

    for (int j = 0; j < polygon_index; j++)
    {
        int cornersOfPolygon = 0;
        int counter = polygon_list[j];

        if (j == (polygon_index - 1))
        {
            cornersOfPolygon = (edge_index - polygon_list[j]);
        }
        else
        {
            cornersOfPolygon = (polygon_list[j + 1] - polygon_list[j]);
        }

        bool start = true;
        bool end = false;
        int *temp_vector = new int[cornersOfPolygon];

        if (cornersOfPolygon < 2)
        {
            cerr << "error: \"polygon\" with " << flush << cornersOfPolygon << " corners found!\n" << flush;
            return;
        }
        else
        {
        }

        int i;
        for (i = 0; i < cornersOfPolygon; i++)
        {
            temp_vector[i] = edge_list[counter + i];
        }

        int first, second;
        for (i = 0; i < cornersOfPolygon; i++)
        {
            if (i == (cornersOfPolygon - 1))
            {
                end = true;
            }
            else
            {
                end = false;
            }

            if (i == 0)
            {
                start = true;
            }
            else
            {
                start = false;
            }

            int ln = temp_vector[i]; //line index
            ln *= 2; //every line has two points
            first = corner_list[ln];
            second = corner_list[ln + 1];

            //cout << "line index: " << flush << ((ln/2)+1) << flush << "  coordinate indices: " << flush << first << flush << ' ' << flush << second << flush << '\n' << flush;

            if (start)
            {
                cl[counter] = first;
                cl[counter + 1] = second;
                start = false;
            }
            else if (end)
            {
                if ((first == cl[counter]) && (second == cl[counter + i]))
                {
                    //cl[counter+i] = second;
                    //break;
                }
                else if ((second == cl[counter]) && (first == cl[counter + i]))
                {
                    //cl[counter+i] = first;
                    //break;
                }
                else
                {
                    //cerr << "error: non-closed polygon found\n" << flush;
                    //return;
                }
            }
            else
            {
                if (first == cl[counter + i])
                {
                    cl[counter + i + 1] = second;
                }
                else if (second == cl[counter + i])
                {
                    cl[counter + i + 1] = first;
                }
                else
                {
                    //cerr << "error: non-closed polygon found\n" << flush;
                    //return;
                }
            }
        }

        delete[] temp_vector;
    }
}

/*****************************************\ 
 *                                       *
 * auxilliary functions string -> number *
 *                                       *
\*****************************************/

//test if we found a whole number
bool ReadIWSGeo::isWholeNumber(char *line)
{
    while (isspace(*line))
    {
        ++line;
    }

    if ((*line == '-') || (*line == '+'))
    {
        int ndigits = 0;
        ++line;
        while (isdigit(*line))
        {
            ++line;
            ++ndigits;
        }
        if (ndigits == 0)
        {
            return false;
        }
        else
        {
            if (isspace(*line) || (*line == '\0'))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }

    else
    {
        int ndigits = 0;
        while (isdigit(*line))
        {
            ++line;
            ++ndigits;
        }
        if (ndigits == 0)
        {
            return false;
        }
        else
        {
            if (isspace(*line) || (*line == '\0'))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    }
    return false;
}

//test, if we found a number.
bool ReadIWSGeo::isNumber(char *line)
{
    while (isspace(*line))
    {
        ++line;
    }

    if ((*line == '-') || (*line == '+'))
    {
        ++line;
    }

    while (isdigit(*line))
    {
        ++line;
    }

    if (isspace(*line) || (*line == '\0'))
    {
        return true;
    }
    else
    {
    }

    if ((*line == '.') || (*line == ','))
    {
        ++line;
        while (isdigit(*line))
        {
            ++line;
        }

        if (isspace(*line) || (*line == '\0'))
        {
            return true;
        }
        else
        {
        }

        if ((*line == 'e') || (*line == 'E'))
        {
            ++line;
            if ((*line == '-') || (*line == '+'))
            {
                ++line;
            }
            else
            {
            }

            if (isdigit(*line))
            {
                while (isdigit(*line))
                {
                    ++line;
                }
                if (isspace(*line) || (*line == '\0'))
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        else
            return false;
    }
    return false;
}

/*************************************\ 
 *                                   *
 * THE DEFINITE BUT NOT ULTIMATE END *
 *                                   *
\*************************************/
