/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Read module for POLYH data         	                  **
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
 ** Date:  26.07.97  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadPolyh.h"

#include <stdio.h>

void main(int argc, char *argv[])
{

    Application *application = new Application(argc, argv);

    application->run();
}

//
// static stub callback functions calling the real class
// member functions
//

void Application::quitCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->quit(callbackData);
}

void Application::computeCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->compute(callbackData);
}

//
//
//..........................................................................
//
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
}

void Application::compute(void *)
{
    //
    // ...... do work here ........
    //

    // read input parameters and data object name
    int i, j, k, fd;
    char buf[300];
    char buf2[300];
    char dp[400];
    char dpend[100];
    int numt, currt, t, endt, isonum;
    int numtri, numv, numt3;
    float *x_c, *y_c, *z_c;
    int *vl, *ll, byte_cnt;
    Header header;

    Covise::get_browser_param("data_path", &data_Path);
    Covise::get_scalar_param("numt", &numt);
    Covise::get_scalar_param("isonum", &isonum);
    strcpy(dp, data_Path);
    i = strlen(dp) - 1;
    while ((dp[i] < '0') || (dp[i] > '9'))
        i--;
    // dp[i] ist jetzt die letzte Ziffer, alles danach ist Endung
    strcpy(dpend, dp + i + 1); // dpend= Endung;
    dp[i + 1] = 0;
    int numd = 0;
    while ((dp[i] >= '0') && (dp[i] <= '9'))
    {
        i--;
        numd++;
    }
    sscanf(dp + i + 1, "%d", &currt); //currt = Aktueller Zeitschritt
    endt = currt + numt;
    dp[i + 1] = 0; // dp = basename

    Mesh = Covise::get_object_name("mesh");
    Data = Covise::get_object_name("data");

    coDistributedObject **Mesh_sets = new coDistributedObject *[numt + 1];
    Mesh_sets[0] = NULL;
    coDistributedObject **Data_sets = new coDistributedObject *[numt + 1];
    Data_sets[0] = NULL;

    for (t = currt; t < endt; t++)
    {
        sprintf(buf2, "%%s%%0%dd%%s", numd);
        sprintf(buf, buf2, dp, t, dpend);
        fd = Covise::open(buf, O_RDONLY);
        if (fd < 0)
        {
            if (t == currt)
            {
                strcpy(buf2, "ERROR: Can't open file >> ");
                strcat(buf2, buf);
                Covise::sendError(buf2);
                return;
            }
            else
            {
                break;
            }
        }

        /* Read header, if insufficient byte cnt, file is corrupted, so quit */
        byte_cnt = read(fd, &header, sizeof(header));
        if (byte_cnt != sizeof(header))
        {
            printf("header read failed, only got %d bytes\n",
                   byte_cnt);
            exit(0);
        }

        /* Read pertinent fields from header data */
        //num_proc = header.num_processors;
        //num_thresholds = header.num_thresholds;

        /* Depending on header fields, skip some number of bytes to get to
         start of data */
        //vol_cnt = header.vol_cnt;
        //volume_code = header.volume_code;
        //flags = header.flags;

        int header_offset = sizeof(int) * 5;
        if ((header.flags & THRESH_RANGE_FLAG) == THRESH_RANGE_FLAG)
            header_offset += header.num_thresholds * 2 * sizeof(float);
        else
            header_offset += header.num_thresholds * sizeof(float);
        if ((header.flags & DECIMATE_FLAG) == DECIMATE_FLAG)
            header_offset += sizeof(double) * 4 + sizeof(float) * (header.vol_cnt - 1);
        lseek(fd, header_offset, SEEK_SET);

        printf("\t#proc=%d  #thresh=%d  #vol=%d\n",
               header.num_processors, header.num_thresholds, header.vol_cnt);

        int total_numv = 0;
        int total_numt = 0;
        for (j = 0; j < header.num_processors; j++)
        {

            for (k = 0; k < header.num_thresholds; k++)
            {

                read(fd, &numv, sizeof(int));
                if (numv)
                {

                    lseek(fd, (2 * numv * sizeof(Point)), SEEK_CUR);
                    if (header.vol_cnt > 1)
                        lseek(fd, numv * sizeof(float) * (header.vol_cnt - 1), SEEK_CUR);

                    read(fd, &numtri, sizeof(int));

                    lseek(fd, (numtri * 3 * sizeof(int)), SEEK_CUR);
                    if (k == isonum)
                    {
                        total_numv += numv;
                        total_numt += numtri;
                    }
                }
            }
        }

        lseek(fd, header_offset, SEEK_SET);

        printf("\t#vert=%d  #tri=%d\n", total_numv, total_numt);

        if (Mesh != NULL)
        {
            sprintf(buf, "%s_%d", Mesh, t);
            mesh = new coDoPolygons(buf, total_numv, total_numt * 3, total_numt);
            if (mesh->objectOk())
            {
                mesh->getAddresses(&x_c, &y_c, &z_c, &vl, &ll);
                mesh->addAttribute("vertexOrder", "2");
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'mesh' failed");
                return;
            }
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'mesh'");
            return;
        }

        if (Data != 0)
        {
            sprintf(buf, "%s_%d", Data, t);
            data = new coDoVec3(buf, total_numv);
            if (data->objectOk())
            {
                data->getAddresses(&vx_data, &vy_data, &vz_data);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'data' failed");
                return;
            }
        }
        else
        {
            Covise::sendError("ERROR: Object name not correct for 'data'");
            return;
        }
        Point p;
        total_numv = 0;
        total_numt = 0;
        int total_numt3 = 0;
        for (j = 0; j < header.num_processors; j++)
        {

            for (k = 0; k < header.num_thresholds; k++)
            {
                read(fd, &numv, sizeof(int));
                if (numv)
                {
                    if (k == isonum)
                    {

                        for (i = 0; i < numv; i++)
                        {
                            read(fd, &p, sizeof(p));
                            *x_c = p.x;
                            x_c++;
                            *y_c = p.y;
                            y_c++;
                            *z_c = p.z;
                            z_c++;
                        }
                        for (i = 0; i < numv; i++)
                        {
                            read(fd, &p, sizeof(p));
                            *vx_data = p.x;
                            vx_data++;
                            *vy_data = p.y;
                            vy_data++;
                            *vz_data = p.z;
                            vz_data++;
                        }
                        //read(fd,x_c,numv*sizeof(float));
                        //read(fd,y_c,numv*sizeof(float));
                        //read(fd,z_c,numv*sizeof(float));
                        //x_c+=numv;
                        //y_c+=numv;
                        //z_c+=numv;
                        if (header.vol_cnt > 1)
                            lseek(fd, numv * sizeof(float) * (header.vol_cnt - 1), SEEK_CUR);

                        read(fd, &numtri, sizeof(int));
                        numt3 = 3 * numtri;

                        read(fd, vl, numt3 * sizeof(int));
                        for (i = 0; i < numt3; i++)
                        {
                            *vl += total_numv;
                            vl++;
                        }
                        for (i = 0; i < numtri; i++)
                        {
                            *ll = total_numt3 + i * 3;
                            ll++;
                        }

                        total_numv += numv;
                        total_numt3 += numt3;
                        total_numt += numtri;
                    }
                    else
                    {

                        lseek(fd, (2 * numv * sizeof(Point)), SEEK_CUR);
                        if (header.vol_cnt > 1)
                            lseek(fd, numv * sizeof(float) * (header.vol_cnt - 1), SEEK_CUR);

                        read(fd, &numtri, sizeof(int));

                        lseek(fd, (numtri * 3 * sizeof(int)), SEEK_CUR);
                    }
                }
            }
        }

        for (i = 0; Mesh_sets[i]; i++)
            ;
        Mesh_sets[i] = mesh;
        Mesh_sets[i + 1] = NULL;
        for (i = 0; Data_sets[i]; i++)
            ;
        Data_sets[i] = data;
        Data_sets[i + 1] = NULL;
        close(fd);
    }
    coDoSet *Mesh_set = new coDoSet(Mesh, Mesh_sets);
    coDoSet *Data_set = new coDoSet(Data, Data_sets);
    Mesh_set->addAttribute("TIMESTEP", "1 100");
    delete Mesh_sets[0];
    delete[] Mesh_sets;
    for (i = 0; Data_sets[i]; i++)
        delete Data_sets[i];
    delete[] Data_sets;
    delete Mesh_set;
    delete Data_set;
}
