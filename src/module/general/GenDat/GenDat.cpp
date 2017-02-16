/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1994 RUS  **
 **                                                                        **
 ** Description: Application to generate various Datasets                  **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                             Uwe Woessner                               **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  21.07.94  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include <do/coDoStructuredGrid.h>
#include <do/coDoUniformGrid.h>
#include <do/coDoRectilinearGrid.h>
#include <do/coDoData.h>

#include "GenDat.h"

enum
{
    UNIFORM = 0,
    RECTILINEAR,
    RANDOM,
    HALF_CYL,
    FULL_CYL,
    TORUS
};

enum
{
    SINUS = 0,
    RAMP,
    DONTKNOW,
    PIPE
};

enum
{
    OPT1 = 0,
    COLIN,
    OPT3,
    RANDORIENT,
    CIRCULAR,
    EXPANDING,
    OSCILLATE
};

enum
{
    MINUS_ONE_TO_ONE = 0,
    ZERO_TO_SIZE,
    START_TO_END
};

#ifdef _WIN32
#define drand48() ((float)rand() / (float)RAND_MAX)
#endif

GenDat::GenDat(int argc, char *argv[])
    : coModule(argc, argv, "Generate data")
{
    gridOut = addOutputPort("GridOut0", "UniformGrid|RectilinearGrid|StructuredGrid", "Grid");
    scalarOut = addOutputPort("DataOut0", "Float", "Scalar Data");
    vectorOut = addOutputPort("DataOut1", "Vec3", "Vector Data");

    const char *coordTypes[] = {
        "Uniform",
        "Rectilinear",
        "Random",
        "Half cylinder",
        "Full cylinder",
        "Torus"
    };
    coordType = addChoiceParam("Coord_Type", "Coordinate type");
    coordType->setValue(6, coordTypes, 0);

    const char *coordReprs[] = {
        "Uniform",
        "Rectilinear",
        "Structured"
    };
    coordRepr = addChoiceParam("Coord_Representation", "Coordinate representation");
    coordRepr->setValue(3, coordReprs, 0);

    const char *coordRanges[] = {
        "-1 to 1",
        "0 to size",
        "start to end"
    };
    coordRange = addChoiceParam("Coord_Range", "Coordinate range");
    coordRange->setValue(3, coordRanges, 0);

    const char *funcs[] = {
        "Sines",
        "Ramps",
        "Random",
        "Pipe"
    };
    func = addChoiceParam("Function", "Function for scalar values");
    func->setValue(4, funcs, 0);

    const char *orients[] = {
        "Opt1",
        "Colin",
        "Opt3",
        "Random",
        "Circular",
        "Expand"
    };
    orient = addChoiceParam("Orientation", "Function for vector values");
    orient->setValue(6, orients, 0);

    xSize = addIntSliderParam("xSize", "Size in x-direction");
    xSize->setValue(1, 64, 8);

    ySize = addIntSliderParam("ySize", "Size in y-direction");
    ySize->setValue(1, 64, 8);

    zSize = addIntSliderParam("zSize", "Size in z-direction");
    zSize->setValue(1, 64, 8);

    start = addFloatVectorParam("start", "lower left point if coord_range is 'start to end'");
    start->setValue(-1., -1., -1.);

    end = addFloatVectorParam("end", "upper right point if coord_range is 'start to end'");
    end->setValue(-1., -1., -1.);

    timestep = addIntSliderParam("timestep", "Timestep if orientation is 'Colin'");
    end->setValue(0, 20, 0);

    color = addColorParam("color", "Color for grid");
    color->setValue(0., 0., 1., 1.);

    attrName = addStringParam("AttributeName", "name of attribute to attach to object");
    attrValue = addStringParam("AttributeVale", "value of attribute to attach to object");
}

void
Invert(int len, float *ar)
{
    for (int i = 0, j = len - 1; i < j; ++i, --j)
        std::swap(ar[i], ar[j]);
}

int GenDat::compute(const char *)
{
    //
    // ...... Read Parameters ........
    //
    long xSize = this->xSize->getValue();
    long ySize = this->ySize->getValue();
    long zSize = this->zSize->getValue();
    int Coord_Type = coordType->getValue();
    long Timestep = timestep->getValue();
    int Coord_Representation = coordRepr->getValue();
    int Coord_Range = coordRange->getValue();
    int Function = func->getValue();
    int Orientation = orient->getValue();
    int doneV = 0;
    const char *COLOR = "COLOR";
    float startx = start->getValue(0);
    float starty = start->getValue(1);
    float startz = start->getValue(2);
    float endx = end->getValue(0);
    float endy = end->getValue(1);
    float endz = end->getValue(2);
    float T = 0.0, a;
    float rr = color->getValue(0);
    float gg = color->getValue(1);
    float bb = color->getValue(2);
    float aa = color->getValue(3);
    coDoFloat *sdaten = NULL;
    coDoVec3 *vdaten = NULL;
    coDoAbstractStructuredGrid *grid = NULL;

    // fprintf(stderr, "in compute Callback\n");

    Covise::log_message(__LINE__, __FILE__, "Generating Data");

    Covise::get_color_param("color", &rr, &gg, &bb, &aa);

    char rgbtxt[20];
    sprintf(rgbtxt, "#%02x%02x%02x%02x", (int)(rr * 255.), (int)(gg * 255.), (int)(bb * 255.), (int)(aa * 255.));

    T = Timestep / 2.0f;

    // cout << "Type: " << Coord_Type << "Funktion: " << Function << "\n";
    // cout << "Range: " << Coord_Range << "Rep: " << Coord_Representation << "\n";
    // cout << "xSize: " << xSize << "\n";
    // cout << "ySize: " << ySize << "\n";
    // cout << "zSize: " << zSize << "\n";

    if (xSize < 1)
        xSize = 1;
    if (ySize < 1)
        ySize = 1;
    if (zSize < 1)
        zSize = 1;

    float xDiv = (float)((xSize > 1) ? (xSize - 1) : 1);
    float yDiv = (float)((ySize > 1) ? (ySize - 1) : 1);
    float zDiv = (float)((zSize > 1) ? (zSize - 1) : 1);

    switch (Coord_Type)
    {
    case UNIFORM:
        if (Coord_Representation == UNIFORM)
        {
            coDoUniformGrid *griduni = NULL;
            //     fprintf(stderr, "vor new coDoUniformGrid %s\n", Mesh);
            if (Coord_Range == MINUS_ONE_TO_ONE)
                griduni = new coDoUniformGrid(gridOut->getObjName(), (int)xSize, (int)ySize, (int)zSize, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
            else if (Coord_Range == ZERO_TO_SIZE)
                griduni = new coDoUniformGrid(gridOut->getObjName(), (int)xSize, (int)ySize, (int)zSize, 0.0, (float)xSize, 0.0, (float)ySize, 0.0, (float)zSize);
            else
                griduni = new coDoUniformGrid(gridOut->getObjName(), (int)xSize, (int)ySize, (int)zSize, startx, endx, starty, endy, startz, endz);
            griduni->addAttribute(COLOR, rgbtxt);

            griduni->addAttribute("DataObjectName", scalarOut->getObjName());
            //     fprintf(stderr, "nach new coDoUniformGrid\n");
            grid = griduni;
        }
        else if (Coord_Representation == RECTILINEAR)
        {
            coDoRectilinearGrid *gridrect = new coDoRectilinearGrid(gridOut->getObjName(), xSize, ySize, zSize);
            gridrect->addAttribute(COLOR, rgbtxt);
            grid = gridrect;

            float *xKoord, *yKoord, *zKoord;
            gridrect->getAddresses(&xKoord, &yKoord, &zKoord);
            if (Coord_Range == MINUS_ONE_TO_ONE)
            {
                for (int i = 0; i < xSize; i++)
                    xKoord[i] = -1.0f + ((i * 2.0f) / xDiv);
                for (int i = 0; i < ySize; i++)
                    yKoord[i] = -1.0f + ((i * 2.0f) / yDiv);
                for (int i = 0; i < zSize; i++)
                    zKoord[i] = -1.0f + ((i * 2.0f) / zDiv);
            }
            else if (Coord_Range == ZERO_TO_SIZE)
            {
                for (int i = 0; i < xSize; i++)
                    xKoord[i] = (float)i;
                for (int i = 0; i < ySize; i++)
                    yKoord[i] = (float)i;
                for (int i = 0; i < zSize; i++)
                    zKoord[i] = (float)i;
            }
            else
            {
                for (int i = 0; i < xSize; i++)
                    xKoord[i] = startx + ((i * endx - startx) / xDiv);
                for (int i = 0; i < ySize; i++)
                    yKoord[i] = starty + ((i * endy - starty) / yDiv);
                for (int i = 0; i < zSize; i++)
                    zKoord[i] = startz + ((i * endz - startz) / zDiv);
            }
            if (xKoord[0] > xKoord[xSize - 1])
            {
                Invert(xSize, xKoord);
            }
            if (yKoord[0] > yKoord[ySize - 1])
            {
                Invert(ySize, yKoord);
            }
            if (zKoord[0] > zKoord[zSize - 1])
            {
                Invert(zSize, zKoord);
            }
        }
        else // Structured Grid
        {
            coDoStructuredGrid *gridstruct = new coDoStructuredGrid(gridOut->getObjName(), xSize, ySize, zSize);
            grid = gridstruct;
            gridstruct->addAttribute(COLOR, rgbtxt);

#ifdef VECTOR_INTERACTOR_ATTRIBUTE
            char abuf[1000];
            sprintf(abuf, "M%s\n%s\n%s\nend\n%f\n%f\n%f\nstart\n%f\n%f\n%f\n1\n", Covise::get_module(), Covise::get_instance(), Covise::get_host(), endx, endy, endz, startx, starty, startz);
            gridstruct->addAttribute("VECTOR0", abuf);
#endif

            float *Koordx, *Koordy, *Koordz;
            gridstruct->getAddresses(&Koordx, &Koordy, &Koordz);

            if (Coord_Range == MINUS_ONE_TO_ONE)
            {
                for (int x = 0; x < xSize; x++)
                    for (int y = 0; y < ySize; y++)
                        for (int z = 0; z < zSize; z++)
                        {
                            Koordx[x * ySize * zSize + y * zSize + z] = -1.0f + ((x * 2.0f) / xDiv);
                            Koordy[x * ySize * zSize + y * zSize + z] = -1.0f + ((y * 2.0f) / yDiv);
                            Koordz[x * ySize * zSize + y * zSize + z] = -1.0f + ((z * 2.0f) / zDiv);
                        }
            }
            else if (Coord_Range == ZERO_TO_SIZE)
            {
                for (int x = 0; x < xSize; x++)
                    for (int y = 0; y < ySize; y++)
                        for (int z = 0; z < zSize; z++)
                        {
                            Koordx[x * ySize * zSize + y * zSize + z] = (float)x;
                            Koordy[x * ySize * zSize + y * zSize + z] = (float)y;
                            Koordz[x * ySize * zSize + y * zSize + z] = (float)z;
                        }
            }
            else
            {
                for (int x = 0; x < xSize; x++)
                    for (int y = 0; y < ySize; y++)
                        for (int z = 0; z < zSize; z++)
                        {
                            Koordx[x * ySize * zSize + y * zSize + z] = startx + (x * (endx - startx) / xDiv);
                            Koordy[x * ySize * zSize + y * zSize + z] = starty + (y * (endy - starty) / yDiv);
                            Koordz[x * ySize * zSize + y * zSize + z] = startz + (z * (endz - startz) / zDiv);
                        }
            }
            if (Orientation == COLIN)
            {
                doneV = 1;
                // Vektordaten
                vdaten = new coDoVec3(vectorOut->getObjName(), xSize * ySize * zSize);
                float *Datenptrx, *Datenptry, *Datenptrz;
                vdaten->getAddresses(&Datenptrx, &Datenptry, &Datenptrz);

                for (int x = 0; x < xSize; x++)
                    for (int y = 0; y < ySize; y++)
                        for (int z = 0; z < zSize; z++)
                        {
                            a = (Koordx[x * ySize * zSize + y * zSize + z] - T) * (Koordx[x * ySize * zSize + y * zSize + z] - T) + Koordy[x * ySize * zSize + y * zSize + z] * Koordy[x * ySize * zSize + y * zSize + z];
                            if (a == 0.0)
                            {
                                Datenptrx[x * ySize * zSize + y * zSize + z] = 0.0;
                                Datenptry[x * ySize * zSize + y * zSize + z] = 0.0;
                                Datenptrz[x * ySize * zSize + y * zSize + z] = 0.0;
                            }
                            else
                            {
                                Datenptrx[x * ySize * zSize + y * zSize + z] = 1 + Koordy[x * ySize * zSize + y * zSize + z] / a;
                                Datenptry[x * ySize * zSize + y * zSize + z] = -1 * (Koordx[x * ySize * zSize + y * zSize + z] - T) / a;
                                Datenptrz[x * ySize * zSize + y * zSize + z] = 0.0;
                            }
                        }
            }
        }

        break;
    case RECTILINEAR:
        if (Coord_Representation == UNIFORM)
        {
            // error = Covise::update_choice_param("Coord_Representation", RECTILINEAR);
        }
        if (Coord_Representation <= RECTILINEAR)
        {
            coDoRectilinearGrid *gridrect = new coDoRectilinearGrid(gridOut->getObjName(), xSize, ySize, zSize);
            grid = gridrect;
            gridrect->addAttribute(COLOR, rgbtxt);
            float *xKoord, *yKoord, *zKoord;
            gridrect->getAddresses(&xKoord, &yKoord, &zKoord);

            if (Coord_Range == MINUS_ONE_TO_ONE)
            {
                for (int i = 0; i < xSize; i++)
                {
                    if ((2 * i - xSize) < 0)
                    {
                        xKoord[i] = (i - xSize / 2.0f) * (i - xSize / 2.0f) / ((xSize / 2.0f - 1.0f) * (xSize / 2.0f - 1.0f));
                    }
                    else
                    {
                        xKoord[i] = -(i - xSize / 2.0f) * (i - xSize / 2.0f) / ((xSize / 2.0f - 1.0f) * (xSize / 2.0f - 1.0f));
                    }
                    //cerr <<"xKoord[" << i << "]=="  << xKoord[i] << endl;
                }
                for (int i = 0; i < ySize; i++)
                {
                    if ((2 * i - ySize) < 0)
                    {
                        yKoord[i] = (i - ySize / 2.0f) * (i - ySize / 2.0f) / ((ySize / 2.0f - 1.0f) * (ySize / 2.0f - 1.0f));
                    }
                    else
                    {
                        yKoord[i] = -(i - ySize / 2.0f) * (i - ySize / 2.0f) / ((ySize / 2.0f - 1.0f) * (ySize / 2.0f - 1.0f));
                    }
                    //cerr <<"yKoord[" << i << "]=="  << yKoord[i] << endl;
                }
                for (int i = 0; i < zSize; i++)
                {
                    if ((2 * i - zSize) < 0)
                    {
                        zKoord[i] = (i - zSize / 2.0f) * (i - zSize / 2.0f) / ((zSize / 2.0f - 1.0f) * (zSize / 2.0f - 1.0f));
                    }
                    else
                    {
                        zKoord[i] = -(i - zSize / 2.0f) * (i - zSize / 2.0f) / ((zSize / 2.0f - 1.0f) * (zSize / 2.0f - 1.0f));
                    }
                    //cerr <<"zKoord[" << i << "]=="  << zKoord[i] << endl;
                }
            }
            else
            {
                for (int i = 0; i < xSize; i++)
                {
                    xKoord[i] = (float)i * (float)i;
                }
                for (int i = 0; i < ySize; i++)
                {
                    yKoord[i] = (float)i * (float)i;
                }
                for (int i = 0; i < zSize; i++)
                {
                    zKoord[i] = (float)i * (float)i;
                }
            }
            if (xKoord[0] > xKoord[xSize - 1])
            {
                Invert(xSize, xKoord);
            }
            if (yKoord[0] > yKoord[ySize - 1])
            {
                Invert(ySize, yKoord);
            }
            if (zKoord[0] > zKoord[zSize - 1])
            {
                Invert(zSize, zKoord);
            }
        }
        else
        {
            coDoStructuredGrid *gridstruct = new coDoStructuredGrid(gridOut->getObjName(), xSize, ySize, zSize);
            grid = gridstruct;
            gridstruct->addAttribute(COLOR, rgbtxt);
            float *Koordx, *Koordy, *Koordz;
            gridstruct->getAddresses(&Koordx, &Koordy, &Koordz);

            if (Coord_Range == MINUS_ONE_TO_ONE)
            {
                for (int x = 0; x < xSize; x++)
                    for (int y = 0; y < ySize; y++)
                        for (int z = 0; z < zSize; z++)
                        {
                            if ((x - xSize / 2.0) < 0)
                                Koordx[x * ySize * zSize + y * zSize + z] = (float)(pow((x - xSize / 2.0), 2) / pow((float)((xSize - 1) - xSize / 2), 2));
                            else
                                Koordx[x * ySize * zSize + y * zSize + z] = (float)(pow((x - xSize / 2.0), 2) / pow((float)((xSize - 1) - xSize / 2), 2) * -1);
                            if ((y - ySize / 2.0) < 0)
                                Koordy[x * ySize * zSize + y * zSize + z] = (float)(pow((y - ySize / 2.0), 2) / pow((float)((ySize - 1) - ySize / 2), 2));
                            else
                                Koordy[x * ySize * zSize + y * zSize + z] = (float)(pow((y - ySize / 2.0), 2) / pow((float)((ySize - 1) - ySize / 2), 2) * -1);
                            if ((z - xSize / 2.0) < 0)
                                Koordz[x * ySize * zSize + y * zSize + z] = (float)(pow((z - zSize / 2.0), 2) / pow((float)((zSize - 1) - zSize / 2), 2));
                            else
                                Koordz[x * ySize * zSize + y * zSize + z] = (float)(pow((z - zSize / 2.0), 2) / pow((float)((zSize - 1) - zSize / 2), 2) * -1);
                        }
            }
            else
            {
                for (int x = 0; x < xSize; x++)
                    for (int y = 0; y < ySize; y++)
                        for (int z = 0; z < zSize; z++)
                        {
                            Koordx[x * ySize * zSize + y * zSize + z] = pow((float)x, 2);
                            Koordy[x * ySize * zSize + y * zSize + z] = pow((float)y, 2);
                            Koordz[x * ySize * zSize + y * zSize + z] = pow((float)z, 2);
                        }
            }
        }
        break;
    case HALF_CYL:
    {
        coDoStructuredGrid *gridstruct = new coDoStructuredGrid(gridOut->getObjName(), xSize, ySize, zSize);
        grid = gridstruct;
        gridstruct->addAttribute(COLOR, rgbtxt);

        float *Koordx, *Koordy, *Koordz;
        gridstruct->getAddresses(&Koordx, &Koordy, &Koordz);

        if (Coord_Representation <= RECTILINEAR)
        {
            // error = Covise::update_choice_param("Coord_Representation", 3);
        }
        for (int x = 0; x < xSize; x++)
            for (int y = 0; y < ySize; y++)
                for (int z = 0; z < zSize; z++)
                {
                    Koordx[x * ySize * zSize + y * zSize + z] = -1.0f + ((x * 2.0f) / xSize);
                    Koordy[x * ySize * zSize + y * zSize + z] = (float)((float)z / zSize * sin(((float)y / yDiv) * M_PI));
                    Koordz[x * ySize * zSize + y * zSize + z] = (float)((float)z / zSize * cos(((float)y / yDiv) * M_PI));
                }
    }
    break;
    case RANDOM:
    {
        coDoStructuredGrid *gridstruct = new coDoStructuredGrid(gridOut->getObjName(), xSize, ySize, zSize);
        grid = gridstruct;
        gridstruct->addAttribute(COLOR, rgbtxt);

        float *Koordx, *Koordy, *Koordz;
        gridstruct->getAddresses(&Koordx, &Koordy, &Koordz);

        if (Coord_Representation <= RECTILINEAR)
        {
            // error = Covise::update_choice_param("Coord_Representation", 3);
        }
        for (int x = 0; x < xSize; x++)
            for (int y = 0; y < ySize; y++)
                for (int z = 0; z < zSize; z++)
                {
                    Koordx[x * ySize * zSize + y * zSize + z] = drand48();
                    Koordy[x * ySize * zSize + y * zSize + z] = drand48();
                    Koordz[x * ySize * zSize + y * zSize + z] = drand48();
                }
    }
    break;
    case FULL_CYL:
    {
        coDoStructuredGrid *gridstruct = new coDoStructuredGrid(gridOut->getObjName(), xSize, ySize, zSize);
        grid = gridstruct;
        gridstruct->addAttribute(COLOR, rgbtxt);

        float *Koordx, *Koordy, *Koordz;
        gridstruct->getAddresses(&Koordx, &Koordy, &Koordz);

        if (Coord_Representation <= RECTILINEAR)
        {
            // error = Covise::update_choice_param("Coord_Representation", 3);
        }
        if ((Koordx = new float[1 + xSize * ySize * zSize]) == NULL)
        {
            cout << "Could not allocate " << xSize *ySize *zSize * sizeof(float) << " Bytes for Koord\n";
            exit(255);
        }
        if ((Koordy = new float[1 + xSize * ySize * zSize]) == NULL)
        {
            cout << "Could not allocate " << xSize *ySize *zSize * sizeof(float) << " Bytes for Koord\n";
            exit(255);
        }
        if ((Koordz = new float[1 + xSize * ySize * zSize]) == NULL)
        {
            cout << "Could not allocate " << xSize *ySize *zSize * sizeof(float) << " Bytes for Koord\n";
            exit(255);
        }
        for (int x = 0; x < xSize; x++)
            for (int y = 0; y < ySize; y++)
                for (int z = 0; z < zSize; z++)
                {
                    Koordx[x * ySize * zSize + y * zSize + z] = -1.0f + ((x * 2.0f) / xSize);
                    Koordy[x * ySize * zSize + y * zSize + z] = (float)((float)z / zSize * sin(((float)y / (ySize - 1)) * M_PI * 2.0));
                    Koordz[x * ySize * zSize + y * zSize + z] = (float)((float)z / zSize * cos(((float)y / (ySize - 1)) * M_PI * 2.0));
                }
    }
    break;
    case TORUS:
    {
        coDoStructuredGrid *gridstruct = new coDoStructuredGrid(gridOut->getObjName(), xSize, ySize, zSize);
        grid = gridstruct;
        gridstruct->addAttribute(COLOR, rgbtxt);

        float *Koordx, *Koordy, *Koordz;
        gridstruct->getAddresses(&Koordx, &Koordy, &Koordz);

        if (Coord_Representation <= RECTILINEAR)
        {
            // error = Covise::update_choice_param("Coord_Representation", 3);
        }
        for (int x = 0; x < xSize; x++)
            for (int y = 0; y < ySize; y++)
                for (int z = 0; z < zSize; z++)
                {
                    Koordx[x * ySize * zSize + y * zSize + z] = (float)(sin(((float)x / (xSize - 1)) * M_PI * 2.0) * (4 + (cos(((float)z / (zSize - 1)) * M_PI * 2.0) * (float)y / ySize)));
                    Koordy[x * ySize * zSize + y * zSize + z] = (float)(sin(((float)z / (zSize - 1)) * M_PI * 2.0) * (float)y / ySize);
                    Koordz[x * ySize * zSize + y * zSize + z] = (float)(cos(((float)x / (xSize - 1)) * M_PI * 2.0) * (4 + (cos(((float)z / (zSize - 1)) * M_PI * 2.0) * (float)y / ySize)));
                }
    }
    break;
    }

    sdaten = new coDoFloat(scalarOut->getObjName(), xSize * ySize * zSize);
    sdaten->addAttribute("DataObjectName", scalarOut->getObjName());
    {
        float *Datenptr = sdaten->getAddress();
        switch (Function)
        {
        case SINUS:
            for (int x = 0; x < xSize; x++)
                for (int y = 0; y < ySize; y++)
                    for (int z = 0; z < zSize; z++)
                        Datenptr[x * ySize * zSize + y * zSize + z] = sin((float)x) * sin((float)y) * sin((float)z);
            break;
        case RAMP:
            for (int x = 0; x < xSize; x++)
                for (int y = 0; y < ySize; y++)
                    for (int z = 0; z < zSize; z++)
                        Datenptr[x * ySize * zSize + y * zSize + z] = (float)(x % 3 + y % 3 + z % 3);
            break;
        case RANDOM:
            for (int x = 0; x < xSize; x++)
                for (int y = 0; y < ySize; y++)
                    for (int z = 0; z < zSize; z++)
                        Datenptr[x * ySize * zSize + y * zSize + z] = drand48();
            break;
        case PIPE:
            for (int x = 0; x < xSize; x++)
                for (int y = 0; y < ySize; y++)
                    for (int z = 0; z < zSize; z++)
                        Datenptr[x * ySize * zSize + y * zSize + z] = sin((float)x) * sin((float)y);
            break;
        }
    }

    if (!doneV)
    {
        // Vektordaten
        vdaten = new coDoVec3(vectorOut->getObjName(), xSize * ySize * zSize);
        float *Datenptrx, *Datenptry, *Datenptrz;
        vdaten->getAddresses(&Datenptrx, &Datenptry, &Datenptrz);

        switch (Orientation)
        {
        case OPT1:
            for (int x = 0; x < xSize; x++)
                for (int y = 0; y < ySize; y++)
                    for (int z = 0; z < zSize; z++)
                    {
                        Datenptrx[x * ySize * zSize + y * zSize + z] = (float)(sin(((float)x / (float)xSize) * (M_PI / 2.0)));
                        Datenptry[x * ySize * zSize + y * zSize + z] = (float)(sin(((float)y / (float)ySize) * (M_PI / 2.0)));
                        Datenptrz[x * ySize * zSize + y * zSize + z] = (float)(sin(((float)z / (float)zSize) * (M_PI / 2.0)));
                    }
            break;
        case COLIN:
            for (int x = 0; x < xSize; x++)
                for (int y = 0; y < ySize; y++)
                    for (int z = 0; z < zSize; z++)
                    {
                        Datenptrx[x * ySize * zSize + y * zSize + z] = (float)x;
                        Datenptry[x * ySize * zSize + y * zSize + z] = (float)y;
                        Datenptrz[x * ySize * zSize + y * zSize + z] = (float)z;
                    }
            break;
        case OPT3:
            for (int x = 0; x < xSize; x++)
                for (int y = 0; y < ySize; y++)
                    for (int z = 0; z < zSize; z++)
                    {
                        Datenptrx[x * ySize * zSize + y * zSize + z] = (float)x;
                        Datenptry[x * ySize * zSize + y * zSize + z] = (float)z;
                        Datenptrz[x * ySize * zSize + y * zSize + z] = (float)-y;
                    }
            break;
        case RANDORIENT:
            for (int x = 0; x < xSize * ySize * zSize; x++)
            {
                Datenptrx[x] = drand48();
                Datenptry[x] = drand48();
                Datenptrz[x] = drand48();
            }
            break;
        case CIRCULAR:
        {
            float v1[3], v2[3], c[3];

            v2[0] = 0.0;
            v2[1] = 1.0;
            v2[2] = 0.0; // Normale
            c[0] = 0.0;
            c[1] = 0.0;
            c[2] = 0.0; // Mittelpunkt

            for (int x = 0; x < xSize; x++)
                for (int y = 0; y < ySize; y++)
                    for (int z = 0; z < zSize; z++)
                    {
                        float coord[3];
                        grid->getPointCoordinates(x, &coord[0], y, &coord[1], z, &coord[2]);

                        v1[0] = coord[0] - c[0];
                        v1[1] = 0.0;
                        v1[2] = coord[2] - c[2];

                        Datenptrz[x * ySize * zSize + y * zSize + z] = v1[0] * v2[1] - v2[0] * v1[1];
                        Datenptrx[x * ySize * zSize + y * zSize + z] = v1[1] * v2[2] - v2[1] * v1[2];
                        Datenptry[x * ySize * zSize + y * zSize + z] = v1[2] * v2[0] - v2[2] * v1[0];
                    }
        }
        break;
        case EXPANDING:
        {
            float v1[3], v2[3], c[3];

            v2[0] = 0.0;
            v2[1] = 1.0;
            v2[2] = 0.0; // Normale
            c[0] = 0.0;
            c[1] = 0.0;
            c[2] = 0.0; // Mittelpunkt

            for (int x = 0; x < xSize; x++)
                for (int y = 0; y < ySize; y++)
                    for (int z = 0; z < zSize; z++)
                    {
                        float coord[3];
                        grid->getPointCoordinates(x, &coord[0], y, &coord[1], z, &coord[2]);

                        v1[0] = coord[0] - c[0];
                        v1[1] = 0.0;
                        v1[2] = coord[2] - c[2];

                        Datenptrz[x * ySize * zSize + y * zSize + z] = (v1[0] * v2[1] - v2[0] * v1[1]) + v1[2] * 0.2f;
                        Datenptrx[x * ySize * zSize + y * zSize + z] = (v1[1] * v2[2] - v2[1] * v1[2]) + v1[0] * 0.2f;
                        Datenptry[x * ySize * zSize + y * zSize + z] = (v1[2] * v2[0] - v2[2] * v1[0]) + v1[1] * 0.2f;
                    }
        }
        break;
        case OSCILLATE:
        {
        }
        break;
        }
    }

    const char *aname = attrName->getValue();
    const char *aval = attrValue->getValue();
    if (aname && aname[0] && aval)
    {
        if (grid)
            grid->addAttribute(aname, aval);
        if (sdaten)
            sdaten->addAttribute(aname, aval);
        if (vdaten)
            vdaten->addAttribute(aname, aval);
    }

    gridOut->setCurrentObject(grid);
    scalarOut->setCurrentObject(sdaten);
    vectorOut->setCurrentObject(vdaten);

    return CONTINUE_PIPELINE;
}

MODULE_MAIN(IO, GenDat)
