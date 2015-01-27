/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                                   RUS  **
 **                                                                        **
 **                                                                        **
 ** Description:   COVISE Read_MovieBYU application module                 **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Thilo Krueger February,March 1998                                      **
 ** under strong support of the vislab team                                **
 **                                                                        **
 \**************************************************************************/

//////
//////
//////

#include <appl/ApplInterface.h>
#include "ReadMovieBYU.h"
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>

//////
////// we must provide main to init covise
//////

int main(int argc, char *argv[])
{
    // init
    Application *application = new Application(argc, argv);

    // and back to covise
    application->run();

    // done
    return 1;
}

//////
////// computeCallback (do nothing but call our real compute-function)
//////

void Application::computeCallback(void *userData, void *)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(NULL);
}

//////
////// this is our compute-routine
//////

void Application::compute(const char *)
{
    // get parameters of path and filetype (bin fortran ..) datatype (skalar vektor)

    Covise::get_browser_param("gridpath", &gridpath);
    Covise::get_browser_param("datapath", &datapath);

    Covise::get_choice_param("filetype", &filetype);
    Covise::get_choice_param("datatype", &datatype);

    Covise::get_scalar_param("timesteps", &timesteps);
    Covise::get_scalar_param("delta", &delta);

    Covise::get_browser_param("colorpath", &colorpath);

    // do not have an idea to get gridtyp -> only unstructured format

    switch (filetype)
    {
    case 1:
        filetype = _FILE_TYPE_BINARY;
        break;
    case 2:
        filetype = _FILE_TYPE_FORTRAN;
        break;
    case 3:
        filetype = _FILE_TYPE_ASCII;
        break;
    }
    switch (datatype)
    {
    case 1:
        datatype = _FILE_NONE;
        timesteps = 1;
        break;
    case 2:
        datatype = _FILE_SKALAR;
        break;
    case 3:
        datatype = _FILE_VEKTOR;
        break;
    case 4:
        datatype = _FILE_DISPLACE;
        break;
    }

    //  has datapath an entry or not
    if (strcmp(datapath, "~/covise/*") == 0)
    {
        datatype = _FILE_NONE;
    }

    // check if timesteps has a correct number
    if (timesteps < 1)
    {
        timesteps = 1;
        Covise::sendInfo("no valid entry for timesteps, using default");
    }

    //////
    ////// the read-function itself
    //////

    // here declaration of the variables for the void functions

    int i, j, k, n; // counter of course
    int nparts, njoints, npolys, nconnect; // 1.line                         np,nj,npt,ncon
    int *npartslist; // 2.line soll integer list geben npl
    float *ncoordinate; // gridcoords                     coords
    int *inconnect; // polygons                       iconn

    float *skalardata, *vektordata; // Data
    float **displacementdata;

    FILE *grid_fp; // this goes for read in data FILE

    // temp stuff
    char buf[300]; // Read in buffer and Covise output variable
    char *obj_name, *data_name;

    int colorNum;
    char **userColors;

    //////
    ////// Movie BYU has an unstructured grid
    //////

    // first we must open our grid file
    // this one is for ascii files
    // don't forget fortran +-1

    switch (filetype)
    {
    case _FILE_TYPE_ASCII:
    {
        // opens the geometry file
        if ((grid_fp = Covise::fopen(gridpath, "r")) == NULL)
        {
            Covise::sendError("ERROR: Can't open file >> %s given by parameter gridpath", gridpath);
            return;
        }

        //  get first line of geometry
        read_first_geo(grid_fp, &nparts, &njoints, &npolys, &nconnect);

        // show the info in Covise
        Covise::sendInfo("parts: %d  nodes/joints: %d  elements/polygone: %d  connectivity: %d", nparts, njoints, npolys, nconnect);

        // get the second line of geometry
        read_second_geo(grid_fp, nparts, npartslist);

        // read the grid
        read_coords(grid_fp, njoints, ncoordinate);

        // get the elements and polygons list
        read_iconn(grid_fp, nconnect, inconnect);

        // get the possible colorlist
        read_color(colorpath, nparts, userColors, &colorNum);

        switch (datatype)
        {
        case _FILE_NONE: // don't read data
            Covise::sendInfo("Info: no data-file selected");
            break;
        case _FILE_SKALAR:
            // reads skalar or vektordata
            read_skalar(datapath, njoints, skalardata);
            break;
        case _FILE_VEKTOR:
            read_vektor(datapath, njoints, vektordata);
            break;
        case _FILE_DISPLACE: // read in displacements
            read_displace(datapath, njoints, timesteps, delta, displacementdata);
            break;
        }

        // and now we transform our data covise compatible,
        // when we have more than one part we need
        // to split up the data in the several parts !!!

        coDistributedObject **outObjs; // grid Objects
        outObjs = new coDistributedObject *[nparts + 1];
        outObjs[nparts] = NULL;

        coDistributedObject **outObjdata; // data Objects
        outObjdata = new coDistributedObject *[nparts + 1];
        outObjdata[nparts] = NULL;

        coDistributedObject **outTime; // data Objects
        outTime = new coDistributedObject *[timesteps + 1];
        outTime[timesteps] = NULL;

        coDistributedObject **transform; // displacement objects
        transform = new coDistributedObject *[nparts + 1];
        transform[nparts] = NULL;

        coDoUnstructuredGrid *trans_out = NULL;
        coDoUnstructuredGrid *poly_out = NULL;

        obj_name = Covise::get_object_name("poly");
        data_name = Covise::get_object_name("data");

        float *all_x_coord, *all_y_coord, *all_z_coord;
        int *all_vl, *all_pl, *all_tl;

        all_x_coord = new float[njoints]; // first i write the global lists to work with
        all_y_coord = new float[njoints]; // out of these i can make my part lists
        all_z_coord = new float[njoints];
        all_vl = new int[nconnect];
        all_pl = new int[npolys];
        all_tl = new int[npolys];

        for (i = 0; i < njoints; i++)
        {
            all_x_coord[i] = ncoordinate[(i * 3)];
            all_y_coord[i] = ncoordinate[(i * 3) + 1];
            all_z_coord[i] = ncoordinate[(i * 3) + 2];
        }
        n = 0;
        for (i = 0; i < npolys; i++)
        {
            all_pl[i] = n;
            for (; inconnect[n] >= 0; n++) // loop without starting point
                all_vl[n] = inconnect[n] - 1; // fortran substract one
            all_vl[n] = (-inconnect[n]) - 1;
            n++;
        }

        // for unstructered i make a type list

        j = 0;
        for (i = 0; i < npolys; i++)
        {
            if (i == (npolys - 1))
                j = nconnect - all_pl[i];
            else
                j = all_pl[i + 1] - all_pl[i];

            if (j == 1)
                all_tl[i] = TYPE_POINT;
            if (j == 2)
                all_tl[i] = TYPE_BAR;
            if (j == 3)
                all_tl[i] = TYPE_TRIANGLE;
            if (j == 4)
                all_tl[i] = TYPE_QUAD;
            if (j == 5)
                all_tl[i] = TYPE_PRISM;
            if (j == 6)
                all_tl[i] = TYPE_PYRAMID;
            if (j == 8)
                all_tl[i] = TYPE_HEXAEDER;
            j = 0;
        }

        // the global lists are ready

        int begin, end, number, next, listplace; // counter for the overview
        int conn_anz, coord_anz; // connects and coords size
        int *temp_arr, *count_arr; // array for the parts view

        float *x_coord, *y_coord, *z_coord;
        int *vl, *pl, *tl;

        for (k = 0; k < timesteps; k++) // begin of the timesteps/displacement loop
        {

            for (j = 0; j < nparts; j++) // begin of the parts loop
            {
                transform[j] = NULL;
                outObjs[j] = NULL;
                outObjdata[j] = NULL;

                // size of the polygons -> number
                begin = npartslist[j * 2] - 1; // remember fortran -1
                end = npartslist[(j * 2) + 1] - 1; // starts counting on 1
                number = end - begin + 1;

                // size of the connects -> conn_anz is the size of vl
                // conn_anz = 0;
                if (begin + number == npolys)
                    next = nconnect; // nconnect must be a fortran number
                // otherwise +1
                else
                    next = all_pl[end + 1];
                conn_anz = next - all_pl[begin];

                // size of the coords
                temp_arr = new int[njoints];
                count_arr = new int[njoints];
                for (n = 0; n < njoints; n++) // set temp_arr to a zero list
                {
                    temp_arr[n] = 0;
                    count_arr[n] = 0;
                }

                for (n = all_pl[begin]; n < next; n++)
                    temp_arr[all_vl[n]] = 1;

                // temp_arr is my 0110 list to eliminate and counting

                coord_anz = 0;
                for (n = 0; n < njoints; n++)
                    coord_anz += temp_arr[n];

                x_coord = new float[coord_anz];
                y_coord = new float[coord_anz];
                z_coord = new float[coord_anz];

                vl = new int[conn_anz];
                pl = new int[number];
                tl = new int[number];

                // we have now our size identifier
                // and can tell them to covise

                // the coords and list objects are pointed in covise
                // and must be produced out of the globlal lists by
                // elimination of the unneccessary parts
                listplace = 0;
                for (n = 0; n < njoints; n++) // do the part coords
                {
                    if (temp_arr[n] == 1)
                    {
                        x_coord[listplace] = all_x_coord[n];
                        y_coord[listplace] = all_y_coord[n];
                        z_coord[listplace] = all_z_coord[n];
                        count_arr[n] = listplace;
                        listplace += 1;
                    }
                }
                // now my count_arr has changed!  it has entrys at the same place as temp_arr
                // but has the old coordinate
                // it is ready for the vertex list

                for (n = 0; n < conn_anz; n++)
                {
                    vl[n] = count_arr[all_vl[all_pl[begin] + n]];
                }

                // now the polygon list is missing

                listplace = all_pl[begin];
                for (n = 0; n < number; n++)
                {
                    pl[n] = all_pl[begin + n] - listplace;
                }

                // and finally the part type list

                for (n = 0; n < number; n++)
                {
                    tl[n] = all_tl[begin + n];
                }

                // and now comes the data
                // read in data when it is existing
                // and split it up in our parts

                coDoFloat *unstr_s3d_out = NULL;
                coDoVec3 *unstr_v3d_out = NULL;

                switch (datatype)
                {
                case _FILE_NONE: // don't make data
                    if (k == 0) // make the first Polygon set only the first time
                    {
                        sprintf(buf, "%s_%d", obj_name, j);
                        poly_out = new coDoUnstructuredGrid(buf, number, conn_anz, coord_anz, pl, vl, x_coord, y_coord, z_coord, tl);
                        if (poly_out->objectOk())
                        {
                            poly_out->addAttribute("vertexOrder", "2");
                            if (userColors != NULL)
                                poly_out->addAttribute("COLOR", userColors[j % colorNum]);
                            outObjs[j] = poly_out;
                        }
                        else
                            outObjs[j] = NULL;
                    }
                    break;
                case _FILE_SKALAR:
                {
                    if (k == 0)
                    {
                        float *partskalardata;
                        partskalardata = new float[coord_anz];
                        listplace = 0;
                        // do the part coords
                        for (n = 0; n < njoints; n++)
                        {
                            if (temp_arr[n] == 1)
                            {
                                partskalardata[listplace] = skalardata[n];
                                listplace += 1;
                            }
                        }
                        // data name with time and part
                        sprintf(buf, "%s_%d", data_name, j);
                        unstr_s3d_out = new coDoFloat(buf, coord_anz, partskalardata);
                        outObjdata[j] = unstr_s3d_out;
                        cerr << "alocare++ " << j << endl;
                        delete[] partskalardata;
                    }

                    break;
                }
                case _FILE_VEKTOR:
                {
                    if (k == 0)
                    {
                        float *u_data, *v_data, *w_data;

                        u_data = new float[coord_anz];
                        v_data = new float[coord_anz];
                        w_data = new float[coord_anz];

                        listplace = 0;
                        for (i = 0; i < njoints; i++)
                        {
                            if (temp_arr[i] == 1)
                            {
                                u_data[listplace] = vektordata[(i * 3)];
                                v_data[listplace] = vektordata[(i * 3) + 1];
                                w_data[listplace] = vektordata[(i * 3) + 2];
                                listplace += 1;
                            }
                        }

                        // data name with time and part
                        sprintf(buf, "%s_%d", data_name, j);
                        unstr_v3d_out = new coDoVec3(buf, coord_anz, u_data, v_data, w_data);
                        outObjdata[j] = unstr_v3d_out;
                        delete[] u_data;
                        delete[] v_data;
                        delete[] w_data;
                    }
                    break;
                }
                case _FILE_DISPLACE:
                {
                    float *i_data, *j_data, *k_data, *ultra;

                    i_data = new float[coord_anz];
                    j_data = new float[coord_anz];
                    k_data = new float[coord_anz];

                    ultra = displacementdata[k];
                    listplace = 0;

                    for (i = 0; i < njoints; i++) // in this loop the new coords were created
                    { // i think it were absolut displacements
                        if (temp_arr[i] == 1)
                        {
                            i_data[listplace] = ultra[(i * 3)] + all_x_coord[i];
                            j_data[listplace] = ultra[(i * 3) + 1] + all_y_coord[i];
                            k_data[listplace] = ultra[(i * 3) + 2] + all_z_coord[i];
                            listplace += 1;
                        }
                    }

                    sprintf(buf, "%s_%d_%d", data_name, k, j);
                    trans_out = new coDoUnstructuredGrid(buf, number, conn_anz, coord_anz, pl, vl, i_data, j_data, k_data, tl);
                    trans_out->addAttribute("vertexOrder", "2");
                    if (userColors != NULL)
                        trans_out->addAttribute("COLOR", userColors[j % colorNum]);

                    transform[j] = trans_out;

                    if (j == nparts - 1)
                    {
                        sprintf(buf, "%s_%d", data_name, k);
                        coDoSet *Transformset = new coDoSet(buf, transform);

                        outTime[k] = Transformset;
                    }

                    delete[] i_data;
                    delete[] j_data;
                    delete[] k_data;

                    break;
                }

                } // end of switch datatype

                delete[] temp_arr;
                delete[] count_arr;

                delete[] x_coord;
                delete[] y_coord;
                delete[] z_coord;
                delete[] vl;
                delete[] pl;
                delete[] tl;

            } // end of the parts loop
            for (j = 0; j < nparts; j++)
                if (transform[j] != NULL)
                    delete transform[j];

        } // end of the timesteps loop

        delete[] all_vl;
        delete[] all_pl;
        delete[] all_tl;
        delete[] all_x_coord;
        delete[] all_y_coord;
        delete[] all_z_coord;

        // and if existing create data
        switch (datatype)
        {
        case _FILE_NONE: // don't delete data
        {
            // outObjs are the different parts
            new coDoSet(obj_name, outObjs);
            for (i = 0; i < nparts; i++)
                if (outObjs[i] != NULL)
                    delete outObjs[i];

            break;
        }

        case _FILE_SKALAR:
        {
            // finally we can create our data set
            // outObjdata are the different data parts
            new coDoSet(data_name, outObjdata);
            for (i = 0; i < nparts; i++)
                if (outObjdata[i] != NULL)
                    delete outObjdata[i];

            delete[] skalardata;
            break;
        }
        case _FILE_VEKTOR:
        {
            // here we make our vektor data
            // outObjdata is the vektorset
            new coDoSet(data_name, outObjdata);
            for (i = 0; i < nparts; i++)
                if (outObjdata[i] != NULL)
                    delete outObjdata[i];
            delete[] vektordata;
            break;
        }
        case _FILE_DISPLACE:
        {
            // here we make the displacement Polygon set
            coDoSet *outDisplacements = new coDoSet(data_name, outTime);

            sprintf(buf, "1 %ld", timesteps);
            outDisplacements->addAttribute("TIMESTEP", buf);

            for (i = 0; i < timesteps; i++)
                delete outTime[i];

            for (i = 0; i < timesteps; i++)
                delete displacementdata[i];
            delete[] displacementdata;
            break;
        }
        }

        // now follows my delete lines
        delete[] npartslist;
        delete[] ncoordinate;
        delete[] inconnect;

        delete[] outTime;
        delete[] outObjs;
        delete[] outObjdata;
        delete[] transform;
        fclose(grid_fp);

        break;
    } // end of case ascii
    default:
        Covise::sendError("ERROR: selected filetype not supported by MovieBYU");
        return;
    } // end of switch filetype

    if (userColors != NULL)
        delete[] userColors;
} // end of compute

// here comes the read functions for ascii read in
// i need following: read_first_geo, read_second_geo, read_coords, read_iconn
//                   read_skalar, read_vektor

void Application::read_first_geo(FILE *grid_fp, int *np, int *nj, int *npt, int *ncon)
{

    if (fscanf(grid_fp, "%d%d%d%d\n", np, nj, npt, ncon) != 4)
    {
        fprintf(stderr, "fscanf_1 failed in ReadMovieBYU.cpp");
    }
}

void Application::read_second_geo(FILE *grid_fp, int np, int *&npl)
{
    int i, zahl;
    zahl = np * 2;
    npl = new int[zahl];
    for (i = 0; i < zahl; i++)
    {
        if (fscanf(grid_fp, "%d", &(npl[i])) != 1)
        {
            fprintf(stderr, "fscanf_2 failed in ReadMovieBYU.cpp");
        }
    }
    return;
}

void Application::read_coords(FILE *grid_fp, int nj, float *&coords)
{
    int i, zahl;
    zahl = nj * 3;
    coords = new float[zahl];
    for (i = 0; i < zahl; i++)
    {
        if (fscanf(grid_fp, "%f", &(coords[i])) != 1)
        {
            fprintf(stderr, "fscanf_3 failed in ReadMovieBYU.cpp");
        }
    }
    return;
}

void Application::read_iconn(FILE *grid_fp, int ncon, int *&iconn)

{
    int i;
    iconn = new int[ncon];
    for (i = 0; i < ncon; i++)
    {
        if (fscanf(grid_fp, "%d", &(iconn[i])) != 1)
        {
            fprintf(stderr, "fscanf_4 failed in ReadMovieBYU.cpp");
        }
    }
    return;
}

// brauch noch fuer beide Reader eine time sequenz

void Application::read_skalar(char *datapath, int nj, float *&skalar)
{
    FILE *data_fp;
    if ((data_fp = Covise::fopen(datapath, "r")) == NULL)
    {
        Covise::sendError("ERROR: Can't open file >> %s given by parameter datapath", datapath);
        datatype = _FILE_NONE;
        return;
    }

    int i;
    skalar = new float[nj];
    for (i = 0; i < nj; i++)
    {
        if (fscanf(data_fp, "%f", &(skalar[i])) != 1)
        {
            fprintf(stderr, "fscanf_5 failed in ReadMovieBYU.cpp");
        }
    }

    fclose(data_fp);
    return;
}

void Application::read_vektor(char *datapath, int nj, float *&vektor)
{
    FILE *data_fp;

    if ((data_fp = Covise::fopen(datapath, "r")) == NULL)
    {
        Covise::sendError("ERROR: Can't open file >> %s given by parameter datapath", datapath);
        datatype = _FILE_NONE;
        return;
    }

    int i, zahl;
    zahl = 3 * nj;
    vektor = new float[zahl];
    for (i = 0; i < zahl; i++)
    {
        if (fscanf(data_fp, "%f", &(vektor[i])) != 1)
        {
            fprintf(stderr, "fscanf_6 failed in ReadMovieBYU.cpp");
        }
    }

    fclose(data_fp);
    return;
}

void Application::read_displace(char *datapath, int nj, int time, int skip, float **&displaced)
{
    int j, i, d, zahl;

    char partpath[200];
    char mainpath[500];
    char *t;

    FILE *data_fp;

    zahl = 3 * nj;
    displaced = new float *[time];
    /* for(j = 0;j < time;j++)
      displaced[j] = new float[zahl];*/

    strcpy(mainpath, datapath);
    mainpath[strlen(mainpath) - 3] = '\0';
    t = datapath + (strlen(datapath) - 3);
    sscanf(t, "%d", &d);

    for (j = 0; j < time; j++)
    {
        sprintf(partpath, "%s%03d", mainpath, d + (j * skip));
        displaced[j] = new float[nj * 3];
        if ((data_fp = Covise::fopen(partpath, "r")) == NULL)
        {
            Covise::sendError("ERROR: Can't open file >> %s given by parameter datapath", partpath);
            datatype = _FILE_NONE;
            return;
        }
        for (i = 0; i < zahl; i++)
        {
            if (fscanf(data_fp, "%f", &displaced[j][i]) != 1)
            {
                fprintf(stderr, "fscanf_7 failed in ReadMovieBYU.cpp");
            }
        }

        fclose(data_fp);
    }

    return;
}

void Application::read_color(char *colorpath, int np, char **&colorlist, int *colornumber)
{
    FILE *color_fp;
    char buf[300];
    int i, j, counter;

    //  has colorpath an entry or not
    if (strcmp(colorpath, "~/covise/*") == 0)
    {
        Covise::sendInfo("no colorfile is selected, default colors are used");
        colorlist = new char *[3];
        for (j = 0; j < 3; j++)
            colorlist[j] = new char[40];
        colorlist[0] = (char *)"red";
        colorlist[1] = (char *)"green";
        colorlist[2] = (char *)"blue";
        *colornumber = 3;
        return;
    }

    if ((color_fp = Covise::fopen(colorpath, "r")) == NULL)
    {
        Covise::sendError("ERROR: Can't open file >> %s given by parameter colorpath", colorpath);
        return;
    }

    counter = 0;
    colorlist = new char *[np];
    for (j = 0; j < np; j++)
        colorlist[j] = new char[40];

    for (i = 0; i < np; i++)
    {
        if (fscanf(color_fp, "%s", buf) != EOF)
        {
            strcpy(colorlist[i], buf);
            counter++;
        }
    }

    *colornumber = counter;

    fclose(color_fp);
    return;
}
