/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Sandia Exo data  	                  **
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
 ** Date:  19.06.97  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadExo.h"

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

void Application::parameterCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->parameter(callbackData);
}

Application::Application(int argc, char *argv[])

{
    file_Path = NULL;
    scal1 = scal2 = scal3 = scal4 = vert1 = 1;
    Covise::set_module_description("Read data from Sandia Exodus format");
    Covise::add_port(OUTPUT_PORT, "mesh", "coDoUnstructuredGrid", "Grid");
    Covise::add_port(OUTPUT_PORT, "sc1", "coDoFloat", "sc1");
    Covise::add_port(OUTPUT_PORT, "sc2", "coDoFloat", "sc2");
    Covise::add_port(OUTPUT_PORT, "sc3", "coDoFloat", "sc3");
    Covise::add_port(OUTPUT_PORT, "sc4", "coDoFloat", "sc4");
    Covise::add_port(OUTPUT_PORT, "v1", "coDoVec3", "v1");
    Covise::add_port(PARIN, "file_path", "Browser", "STL file path");
    Covise::add_port(PARIN, "timestep", "Slider", "Timestep");
    Covise::add_port(PARIN, "scal1", "Choice", "scal1");
    Covise::add_port(PARIN, "scal2", "Choice", "scal2");
    Covise::add_port(PARIN, "scal3", "Choice", "scal3");
    Covise::add_port(PARIN, "scal4", "Choice", "scal4");
    Covise::add_port(PARIN, "vert1", "Choice", "vert1");
    Covise::set_port_default("scal1", "1 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10");
    Covise::set_port_default("scal2", "1 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10");
    Covise::set_port_default("scal3", "1 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10");
    Covise::set_port_default("scal4", "1 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10");
    Covise::set_port_default("vert1", "1 S1 S2 S3 S4 S5 S6 S7 S8 S9 S10");
    Covise::set_port_default("timestep", "0 100 0");
    Covise::set_port_default("file_path", "data/sandia/etch3.exo *.exo*");
    Covise::set_port_immediate("file_path", 1);
    Covise::init(argc, argv);
    Covise::set_quit_callback(Application::quitCallback, this);
    Covise::set_start_callback(Application::computeCallback, this);
    Covise::set_param_callback(Application::parameterCallback, this);
    /*Covise::reply_param_name="file_path";
   Covise::reply_param_type="Browser";
   Covise::no_of_reply_tokens=1;
   Covise::reply_buffer = new char*[Covise::no_of_reply_tokens];
   Covise::reply_buffer[0] = "data/sandia/etch3.exo *.exo*";
   Covise::callParamReplyCallback();*/

    /*
   //send me a Param message with the default value
   sprintf(tb1,"C%s\n%s\n%s\n",Covise::get_module(),Covise::get_instance(),Covise::get_host());
   Covise::set_feedback_info(tb1);
   Covise::send_feedback_message("PARREP","file_path\nBrowser\n1\ndata/sandia/etch3.exo\n*.exo*\n");
   */
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

void Application::parameter(void *)
{

    char *pname;
    char buf[600];
    int i;
    static int first = 1;
    pname = Covise::get_reply_param_name();

    if (strcmp("file_path", pname) == 0)
    {
        timestep = 1;
        Covise::get_reply_browser(&file_Path);
        Covise::getname(buf, file_Path);
        if (buf[0] == '\0')
        {
            strcpy(buf, "ERROR: Can't find file >> ");
            strcat(buf, file_Path);
            Covise::sendError(buf);
            return;
        }

        file_Path = new char[strlen(buf) + 1];
        strcpy(file_Path, buf);
        exoid = ex_open(file_Path, EX_READ, &word_size, &io_word_size, &version);
        if (exoid == -1)
        {
            strcpy(tb1, "ERROR: Can't open file >> ");
            strcat(tb1, file_Path);
            Covise::sendError(tb1);
            return;
        }
        ex_get_init(exoid, database_title, &num_dim,
                    &num_nodes, &num_elem, &num_elem_blks,
                    &num_node_sets, &num_side_sets);

        ex_get_var_param(exoid, "n", &num_nodal_variables);
        ex_get_var_param(exoid, "e", &num_element_variables);
        ex_get_var_param(exoid, "g", &num_global_variables);

        sprintf(buf, "Dataset: %s Version: %f", database_title, version);
        Covise::sendInfo(buf);
        sprintf(buf, "Num_elem: %d Num_Coord: %d Num_blks: %d Num_node_sets: %d Num_side_sets: %d Num_node_vars: %d Num_elem_vars: %d", num_elem, num_nodes,
                num_elem_blks, num_node_sets, num_side_sets, num_nodal_variables, num_element_variables);
        Covise::sendInfo(buf);

        if (num_nodal_variables || num_element_variables)
        {
            var_names = new char *[num_nodal_variables + num_element_variables];
            for (i = 0; i < num_nodal_variables + num_element_variables; i++)
            {
                var_names[i] = new char[MAX_LINE_LENGTH + 1];
            }
            ex_get_var_names(exoid, "n", num_nodal_variables,
                             var_names);
            ex_get_var_names(exoid, "e", num_element_variables,
                             &(var_names[num_nodal_variables]));

            //for( i = 0; i < num_nodal_variables+num_element_variables; i++ ){
            //    strcat(var_names[i],"\n");
            //}
            if (!first)
            {
                Covise::update_choice_param("scal1", num_nodal_variables + num_element_variables, var_names, scal1);
                Covise::update_choice_param("scal2", num_nodal_variables + num_element_variables, var_names, scal2);
                Covise::update_choice_param("scal3", num_nodal_variables + num_element_variables, var_names, scal3);
                Covise::update_choice_param("scal4", num_nodal_variables + num_element_variables, var_names, scal4);
                Covise::update_choice_param("vert1", num_nodal_variables + num_element_variables, var_names, vert1);
            }
            //for( i = 0; i < num_nodal_variables+num_element_variables; i++ ){
            //    var_names[i][strlen(var_names[i])-1]=0;
            //}
        }
        ex_inquire(exoid, EX_INQ_TIME, &num_timesteps, NULL, NULL);
        timestep = 0;
        if (!first)
            Covise::update_slider_param("timestep", 0, num_timesteps - 1, timestep);

        exoid = ex_close(exoid);
    }
    first = 0;
}

void Application::compute(void *)
{
    //
    // ...... do work here ........
    //

    // read input parameters and data object name
    int i, tmpi, min, max;
    char buf[600];
    char BLOCKINFObuf[600];
    char READ_MODULEbuf[600];
    char *Mesh, *Scalar1, *Scalar2, *Scalar3, *Scalar4, *Vertex1;
    coDoUnstructuredGrid *mesh;
    coDoFloat *Scal1;
    coDoFloat *Scal2;
    coDoFloat *Scal3;
    coDoFloat *Scal4;
    coDoVec3 *Vert1;
    float *scdata, *xd, *yd, *zd;

    if (file_Path == NULL)
    {
        Covise::sendError("Please select a file first!\n");
        return;
    }
    Covise::get_slider_param("timestep", &min, &max, &timestep);
    if (timestep < min)
        timestep = min;
    Covise::get_choice_param("scal1", &scal1);
    Covise::get_choice_param("scal2", &scal2);
    Covise::get_choice_param("scal3", &scal3);
    Covise::get_choice_param("scal4", &scal4);
    Covise::get_choice_param("vert1", &vert1);

    Mesh = Covise::get_object_name("mesh");
    Scalar1 = Covise::get_object_name("sc1");
    Scalar2 = Covise::get_object_name("sc2");
    Scalar3 = Covise::get_object_name("sc3");
    Scalar4 = Covise::get_object_name("sc4");
    Vertex1 = Covise::get_object_name("v1");

    Covise::getname(buf, file_Path);
    if (buf[0] == '\0')
    {
        strcpy(buf, "ERROR: Can't find file >> ");
        strcat(buf, file_Path);
        Covise::sendError(buf);
        return;
    }
    //strcpy(buf,file_Path);

    io_word_size = 0;
    word_size = sizeof(float);

    exoid = ex_open(file_Path, EX_READ, &word_size, &io_word_size, &version);
    if (exoid == -1)
    {
        strcpy(tb1, "ERROR: Can't open file >> ");
        strcat(tb1, file_Path);
        Covise::sendError(tb1);
        return;
    }

    //printf( "\nfile id = %d, version = %f\n", exoid, version );
    //printf( "word size = %d,  io word size = %d\n", word_size, io_word_size );

    int *elem_blk_ids = new int[num_elem_blks];
    int *element_types = new int[num_elem_blks];
    int *element_ns = new int[num_elem_blks];
    ex_get_elem_blk_ids(exoid, elem_blk_ids);

    sprintf(BLOCKINFObuf, "%d %d %d", timestep - min, max - min, min);
    sprintf(READ_MODULEbuf, "O%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());

    num_elem = 0;
    num_conn = 0;
    for (i = 0; i < num_elem_blks; i++)
    {
        ex_get_elem_block(exoid, elem_blk_ids[i], ex_buf,
                          element_ns + i, &tmpi, &tmpi);

        element_types[i] = TYPE_HEXAEDER;

        if (!strncmp(ex_buf, "HEX", 3))
        {
            element_types[i] = TYPE_HEXAEDER;
        }
        else if (!strncmp(ex_buf, "SHELL", 5))
        {
            element_types[i] = TYPE_QUAD;
        }
        else if (!strncmp(ex_buf, "TRUSS", 5))
        {
            element_types[i] = TYPE_BAR;
        }
        else if (!strncmp(ex_buf, "SPHERE", 6))
        {
            element_types[i] = TYPE_POINT;
        }
        else if (!strncmp(ex_buf, "TETRA", 5))
        {
            element_types[i] = TYPE_TETRAHEDER;
        }
        else
        {
            sprintf(buf, "Unknown elemtype %s", ex_buf);
            Covise::sendInfo(buf);
        }
        num_conn += UnstructuredGrid_Num_Nodes[element_types[i]] * element_ns[i];
        num_elem += element_ns[i];
    }

    if (Mesh != NULL)
    {
        mesh = new coDoUnstructuredGrid(Mesh, num_elem, num_conn, num_nodes, 1);
        if (mesh->objectOk())
        {
            mesh->getAddresses(&el, &vl, &x_coord, &y_coord, &z_coord);
            mesh->getTypeList(&tl);
            ex_get_coord(exoid, x_coord, y_coord, z_coord);
            num_conn = 0;
            int nc = 0;
            int elpos = 0;
            for (i = 0; i < num_elem_blks; i++)
            {
                ex_get_elem_conn(exoid, elem_blk_ids[i], vl + num_conn);
                nc = UnstructuredGrid_Num_Nodes[element_types[i]] * element_ns[i];
                for (j = 0; j < element_ns[i]; j++)
                {
                    *tl = element_types[i];
                    tl++;
                    *el = elpos;
                    el++;
                    elpos += UnstructuredGrid_Num_Nodes[element_types[i]];
                }
                for (j = 0; j < nc; j++)
                {
                    vl[num_conn + j]--;
                }
                num_conn += nc;
            }
            mesh->addAttribute("BLOCKINFO", BLOCKINFObuf);
            mesh->addAttribute("READ_MODULE", READ_MODULEbuf);
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
    Scal1 = NULL;
    Scal2 = NULL;
    Scal3 = NULL;
    Scal4 = NULL;
    Vert1 = NULL;
    if ((num_nodal_variables + num_element_variables) > 0)
    {
        if (Scalar1 != NULL)
        {
            if (scal1 > num_nodal_variables)
                Scal1 = new coDoFloat(Scalar1, num_elem);
            else
                Scal1 = new coDoFloat(Scalar1, num_nodes);
            if (Scal1->objectOk())
            {
                Scal1->getAddress(&scdata);
                if (scal1 > num_nodal_variables)
                {
                    num_elem = 0;
                    for (i = 0; i < num_elem_blks; i++)
                    {
                        ex_get_elem_var(exoid, timestep + 1, scal1 - num_nodal_variables, elem_blk_ids[i], element_ns[i], scdata + num_elem);
                        num_elem += element_ns[i];
                    }
                }
                else
                    ex_get_nodal_var(exoid, timestep + 1, scal1, num_nodes, scdata);
                Scal1->addAttribute("BLOCKINFO", BLOCKINFObuf);
                Scal1->addAttribute("READ_MODULE", READ_MODULEbuf);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'sc1' failed");
                return;
            }
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'sc1'");
            return;
        }
        if (Scalar2 != NULL)
        {
            if (scal2 > num_nodal_variables)
                Scal2 = new coDoFloat(Scalar2, num_elem);
            else
                Scal2 = new coDoFloat(Scalar2, num_nodes);
            if (Scal2->objectOk())
            {
                Scal2->getAddress(&scdata);
                if (scal2 > num_nodal_variables)
                {
                    num_elem = 0;
                    for (i = 0; i < num_elem_blks; i++)
                    {
                        ex_get_elem_var(exoid, timestep + 1, scal2 - num_nodal_variables, elem_blk_ids[i], element_ns[i], scdata + num_elem);
                        num_elem += element_ns[i];
                    }
                }
                else
                    ex_get_nodal_var(exoid, timestep + 1, scal2, num_nodes, scdata);
                Scal2->addAttribute("BLOCKINFO", BLOCKINFObuf);
                Scal2->addAttribute("READ_MODULE", READ_MODULEbuf);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'sc2' failed");
                return;
            }
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'sc2'");
            return;
        }

        if (Scalar3 != NULL)
        {
            if (scal3 > num_nodal_variables)
                Scal3 = new coDoFloat(Scalar3, num_elem);
            else
                Scal3 = new coDoFloat(Scalar3, num_nodes);
            if (Scal3->objectOk())
            {
                Scal3->getAddress(&scdata);
                if (scal3 > num_nodal_variables)
                {
                    num_elem = 0;
                    for (i = 0; i < num_elem_blks; i++)
                    {
                        ex_get_elem_var(exoid, timestep + 1, scal3 - num_nodal_variables, elem_blk_ids[i], element_ns[i], scdata + num_elem);
                        num_elem += element_ns[i];
                    }
                }
                else
                    ex_get_nodal_var(exoid, timestep + 1, scal3, num_nodes, scdata);
                Scal3->addAttribute("BLOCKINFO", BLOCKINFObuf);
                Scal3->addAttribute("READ_MODULE", READ_MODULEbuf);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'sc2' failed");
                return;
            }
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'sc3'");
            return;
        }

        if (Scalar4 != NULL)
        {
            if (scal4 > num_nodal_variables)
                Scal4 = new coDoFloat(Scalar4, num_elem);
            else
                Scal4 = new coDoFloat(Scalar4, num_nodes);
            if (Scal4->objectOk())
            {
                Scal4->getAddress(&scdata);
                if (scal4 > num_nodal_variables)
                {
                    num_elem = 0;
                    for (i = 0; i < num_elem_blks; i++)
                    {
                        ex_get_elem_var(exoid, timestep + 1, scal4 - num_nodal_variables, elem_blk_ids[i], element_ns[i], scdata + num_elem);
                        num_elem += element_ns[i];
                    }
                }
                else
                    ex_get_nodal_var(exoid, timestep + 1, scal4, num_nodes, scdata);
                Scal4->addAttribute("BLOCKINFO", BLOCKINFObuf);
                Scal4->addAttribute("READ_MODULE", READ_MODULEbuf);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'sc2' failed");
                return;
            }
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'sc4'");
            return;
        }
    }
    Vert1 = NULL;
    if (num_nodal_variables > 2)
    {
        if (Vertex1 != NULL)
        {
            if (vert1 > num_nodal_variables)
                Vert1 = new coDoVec3(Vertex1, num_elem);
            else
                Vert1 = new coDoVec3(Vertex1, num_nodes);
            if (Vert1->objectOk())
            {
                Vert1->getAddresses(&xd, &yd, &zd);
                if (vert1 > num_nodal_variables)
                {
                    num_elem = 0;
                    for (i = 0; i < num_elem_blks; i++)
                    {
                        ex_get_elem_var(exoid, timestep + 1, vert1 - num_nodal_variables, elem_blk_ids[i], element_ns[i], xd + num_elem);
                        ex_get_elem_var(exoid, timestep + 1, vert1 - num_nodal_variables + 1, elem_blk_ids[i], element_ns[i], yd + num_elem);
                        ex_get_elem_var(exoid, timestep + 1, vert1 - num_nodal_variables + 2, elem_blk_ids[i], element_ns[i], zd + num_elem);
                        num_elem += element_ns[i];
                    }
                }
                else
                {
                    ex_get_nodal_var(exoid, timestep + 1, vert1, num_nodes, xd);
                    ex_get_nodal_var(exoid, timestep + 1, vert1 + 1, num_nodes, yd);
                    ex_get_nodal_var(exoid, timestep + 1, vert1 + 2, num_nodes, zd);
                    if (strncasecmp(var_names[vert1 - 1], "DIS", 3) == 0)
                    {
                        for (i = 0; i < num_nodes; i++)
                        {
                            x_coord[i] += xd[i];
                            y_coord[i] += yd[i];
                            z_coord[i] += zd[i];
                        }
                    }
                }
                Vert1->addAttribute("BLOCKINFO", BLOCKINFObuf);
                Vert1->addAttribute("READ_MODULE", READ_MODULEbuf);
            }
            else
            {
                Covise::sendError("ERROR: creation of data object 'v1' failed");
                return;
            }
        }
        else
        {
            Covise::sendError("ERROR: object name not correct for 'v1'");
            return;
        }
    }

    /* for( i = 0; i < num_nodal_variables; i++ ){
   delete[] var_names[i];
    }
    delete[] var_names;*/
    delete[] elem_blk_ids;
    delete[] element_types;
    delete[] element_ns;

    exoid = ex_close(exoid);
    delete mesh;
    delete Scal1;
    delete Scal2;
    delete Scal3;
    delete Scal4;
    delete Vert1;
}
