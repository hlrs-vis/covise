/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description: Read module for fire Files                              **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 ** Author:                                                                **
 **                                                                        **
 **                            Andreas Wierse                              **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 ** Date:  30.07.98  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadFireUniversal.h"
#include "FireUniversalFile.h"
#include <string.h>

//macros
#define ERR0(cond, text, action)     \
    {                                \
        if (cond)                    \
        {                            \
            Covise::sendError(text); \
            {                        \
                action               \
            }                        \
        }                            \
    }

#define ERR1(cond, text, arg1, action) \
    {                                  \
        if (cond)                      \
        {                              \
            sprintf(buf, text, arg1);  \
            Covise::sendError(buf);    \
            {                          \
                action                 \
            }                          \
        }                              \
    }

#define ERR2(cond, text, arg1, arg2, action) \
    {                                        \
        if (cond)                            \
        {                                    \
            sprintf(buf, text, arg1, arg2);  \
            Covise::sendError(buf);          \
            {                                \
                action                       \
            }                                \
        }                                    \
    }
;
static const int NDATA = 3; // number of results data fields

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

void
Application::paramCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->paramChange(callbackData);
}

/*********************************
 *                               *
 *     C O N S T R U C T O R     *
 *                               *
 *********************************/

Application::Application(int argc, char *argv[])
{
    firegridfile = 0L;
    firedatafile = 0L;
    gridfile_name = 0L;
    datafile_name = 0L;
    choicelist = 0L;

    Covise::set_module_description("Read Fire Universal Files");

    // File Name
    Covise::add_port(PARIN, "gridfile_name", "Browser", "Grid File path");
    Covise::set_port_default("gridfile_name", "./*.*");
    Covise::set_port_immediate("gridfile_name", 1);

    Covise::add_port(PARIN, "datafile_name", "Browser", "Data File path");
    Covise::set_port_default("datafile_name", "./*.*");
    Covise::set_port_immediate("datafile_name", 1);

    Covise::add_port(PARIN, "field1", "Choice", "Field to be read");
    Covise::set_port_default("field1", "1 ---");

    Covise::add_port(PARIN, "field2", "Choice", "Field to be read");
    Covise::set_port_default("field2", "1 ---");

    Covise::add_port(PARIN, "field3", "Choice", "Field to be read");
    Covise::set_port_default("field3", "1 ---");

    // Output
    Covise::add_port(OUTPUT_PORT, "mesh", "coDoUnstructuredGrid", "Mesh output");
    Covise::add_port(OUTPUT_PORT, "data1", "coDoVec3 | coDoFloat", "Data Field 1 output");
    Covise::add_port(OUTPUT_PORT, "data2", "coDoVec3 | coDoFloat", "Data Field 2 output");
    Covise::add_port(OUTPUT_PORT, "data3", "coDoVec3 | coDoFloat", "Data Field 3 output");

    // Do the setup
    Covise::init(argc, argv);
    Covise::set_quit_callback(Application::quitCallback, this);
    Covise::set_start_callback(Application::computeCallback, this);
    Covise::set_param_callback(Application::paramCallback, this);

    // Set internal object pointers to Files and Filenames
}

/*******************************
 *                             *
 *     D E S T R U C T O R     *
 *                             *
 *******************************/

Application::~Application()
{
}

void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
}

void Application::paramChange(void *)
{

    char *tmp;
    char *pname, *new_gridfile_name;
    char *new_datafile_name;

    // get watchdir parameter
    pname = Covise::get_reply_param_name();

    if (strcmp("gridfile_name", pname) == 0)
    {
        Covise::get_reply_browser((const char **)&tmp);

        new_gridfile_name = (char *)new char[strlen(tmp) + 1];
        strcpy(new_gridfile_name, tmp);

        if (new_gridfile_name != NULL)
        {
            if (gridfile_name == 0 || strcmp(gridfile_name, new_gridfile_name) != 0)
            {
                delete gridfile_name;
                gridfile_name = new_gridfile_name;
                if (firegridfile != 0)
                    delete firegridfile;
                firegridfile = new FireFile(gridfile_name);

                ReadGrid(DETERMINE_SIZE);
            }
        }
        else
        {
            Covise::sendError("ERROR:file_name is NULL");
        }
    }
    else if (strcmp("datafile_name", pname) == 0)
    {
        Covise::get_reply_browser((const char **)&tmp);

        new_datafile_name = (char *)new char[strlen(tmp) + 1];
        strcpy(new_datafile_name, tmp);

        if (new_datafile_name != NULL)
        {
            if (datafile_name == 0 || strcmp(datafile_name, new_datafile_name) != 0)
            {
                delete datafile_name;
                datafile_name = new_datafile_name;
                if (firedatafile != 0)
                    delete firedatafile;
                firedatafile = new FireFile(datafile_name);

                ResetChoiceList();
                ReadData(DETERMINE_SIZE);
                UpdateChoiceList();
            }
        }
        else
        {
            Covise::sendError("ERROR:file_name is NULL");
        }
    }
}

void Application::compute(void *)
{

    // ======================== Input parameters ======================

    char *new_file_name, *tmp_name, buf[256];
    int i;
    int fieldNo[NDATA];
    char *pname, *new_gridfile_name;
    char *new_datafile_name;

    //   	Covise::get_browser_param("grid_file_name", &tmp_name);
    //	new_file_name = (char*) new char[strlen(tmp_name)+1];
    //	strcpy(new_gridfile_name,tmp_name);

    //   	Covise::get_browser_param("data_file_name", &tmp_name);
    //	new_file_name = (char*) new char[strlen(tmp_name)+1];
    //	strcpy(new_datafile_name,tmp_name);

    //     int ende = 0;
    //     while(!ende)
    //         ;

    for (i = 0; i < NDATA; i++)
    {
        sprintf(buf, "field%i", i + 1);
        Covise::get_choice_param(buf, fieldNo + i);
    }
    no1 = choicelist->get_orig_num(fieldNo[0]);
    no2 = choicelist->get_orig_num(fieldNo[1]);
    no3 = choicelist->get_orig_num(fieldNo[2]);
    //	cout << "Chosen: " << no1 << ", " << no2 << ", " << no3 << endl;

    ReadGrid(READ_DATA);
    ReadData(READ_DATA);
}

int Application::ReadData(int skip)
{
    int cont = 1;
    char buffer[255];
    float *s, *u, *v, *w;
    int field1, field2, field3;

    if (skip == DETERMINE_SIZE)
    {
        int count = 0;
        firedatafile->skip_block();
        firedatafile->skip_block();
        while (cont && count < MAX_NO_OF_DATA_SETS)
        {
            cont = firedatafile->determine_data(no_of_elements,
                                                data_start[count],
                                                &data_name[count],
                                                data_type[count]);
            if (cont)
            {
                sprintf(buffer, "%s", data_name[count]);
                Covise::sendInfo(buffer);
                count++;
            }
            //            cout << "count = " << count << endl;
        }
        if (count == MAX_NO_OF_DATA_SETS)
        {
            sprintf(buffer, "data sets exceed maximum no, allowed: %d", MAX_NO_OF_DATA_SETS);
            Covise::sendInfo(buffer);
        }
        no_of_data_sets = count;
        //        cout << "read " << no_of_data_sets << " data sets\n";
        return no_of_data_sets;
    }
    else
    {
        Covise::get_choice_param("field1", &field1);
        Covise::get_choice_param("field2", &field2);
        Covise::get_choice_param("field3", &field3);

        if (field1 > choicelist->get_num())
            field1 = 1;
        if (field2 > choicelist->get_num())
            field2 = 1;
        if (field3 > choicelist->get_num())
            field3 = 1;
        field1 -= 2;
        field2 -= 2;
        field3 -= 2;

        /*
              cout << "field1 = " << field1 << endl;
              for(int k = 0;k < no_of_data_sets;k++)
               cout << k << ": " << data_name[k] << endl;
      */

        char *Data1 = Covise::get_object_name("data1");
        char *Data2 = Covise::get_object_name("data2");
        char *Data3 = Covise::get_object_name("data3");

        ERR0((Data1 == NULL), "Error getting name 'data1'", return (0);)
        ERR0((Data2 == NULL), "Error getting name 'data2'", return (0);)
        ERR0((Data3 == NULL), "Error getting name 'data3'", return (0);)

        if (data_type[field1] == DATA_SCALAR)
        {
            coDoFloat *data1 = new coDoFloat(Data1, no_of_elements);
            ERR0((data1->objectOk() != 1), "Error creating Fire data1 object", return (0);)
            data1->getAddress(&s);
            firedatafile->read_data(no_of_elements, data_start[field1], s);
            delete data1;
        }
        else
        {
            coDoVec3 *data1 = new coDoVec3(Data1, no_of_elements);
            ERR0((data1->objectOk() != 1), "Error creating Fire data1 object", return (0);)
            data1->getAddresses(&u, &v, &w);
            firedatafile->read_data(no_of_elements, data_start[field1], u, v, w);
            delete data1;
        }

        if (data_type[field2] == DATA_SCALAR)
        {
            coDoFloat *data2 = new coDoFloat(Data2, no_of_elements);
            ERR0((data2->objectOk() != 1), "Error creating Fire data2 object", return (0);)
            data2->getAddress(&s);
            firedatafile->read_data(no_of_elements, data_start[field2], s);
            delete data2;
        }
        else
        {
            coDoVec3 *data2 = new coDoVec3(Data2, no_of_elements);
            ERR0((data2->objectOk() != 1), "Error creating Fire data2 object", return (0);)
            data2->getAddresses(&u, &v, &w);
            firedatafile->read_data(no_of_elements, data_start[field2], u, v, w);
            delete data2;
        }

        if (data_type[field3] == DATA_SCALAR)
        {
            coDoFloat *data3 = new coDoFloat(Data3, no_of_elements);
            ERR0((data3->objectOk() != 1), "Error creating Fire data3 object", return (0);)
            data3->getAddress(&s);
            firedatafile->read_data(no_of_elements, data_start[field3], s);
            delete data3;
        }
        else
        {
            coDoVec3 *data3 = new coDoVec3(Data3, no_of_elements);
            ERR0((data3->objectOk() != 1), "Error creating Fire data3 object", return (0);)
            data3->getAddresses(&u, &v, &w);
            firedatafile->read_data(no_of_elements, data_start[field3], u, v, w);
            delete data3;
        }
    }
}

int Application::ReadGrid(int skip)
{
    char buffer[255];
    int *elem, *conn, *typelist;
    float *x, *y, *z;

    if (skip == DETERMINE_SIZE)
    {
        firegridfile->skip_block();
        firegridfile->read_nodes(no_of_grid_points);
        sprintf(buffer, "Reading %d grid points", no_of_grid_points);
        Covise::sendInfo(buffer);
        firegridfile->read_elements(no_of_elements);
        sprintf(buffer, "Reading %d elements", no_of_elements);
        Covise::sendInfo(buffer);
        return no_of_elements;
    }
    else
    {
        char *Mesh = Covise::get_object_name("mesh");

        ERR0((Mesh == NULL), "Error getting name 'mesh'", return (0);)

        coDoUnstructuredGrid *mesh = new coDoUnstructuredGrid(Mesh,
                                                              no_of_elements, no_of_elements * 8, no_of_grid_points, 1);

        ERR0((mesh->objectOk() != 1), "Error creating Fire mesh object", return (0);)

        mesh->getAddresses(&elem, &conn, &x, &y, &z);

        firegridfile->read_nodes(no_of_grid_points, x, y, z);
        firegridfile->read_elements(no_of_grid_points, elem, conn);

        mesh->getTypeList(&typelist);
        for (int i = 0; i < no_of_elements; i++)
            typelist[i] = TYPE_HEXAEDER;

        delete mesh;

        return no_of_elements;
    }
}

void Application::ResetChoiceList(void)
{

    delete choicelist;

    choicelist = new ChoiceList("---", 0);

    return;
}

void Application::UpdateChoiceList(void)
{
    int i;

    for (i = 0; i < no_of_data_sets; i++)
        choicelist->add(data_name[i], i + 1);

    if (!Covise::in_map_loading())
    {
        Covise::update_choice_param(
            "field1", choicelist->get_num(), (char **)choicelist->get_strings(), 1);
        Covise::update_choice_param(
            "field2", choicelist->get_num(), (char **)choicelist->get_strings(), 1);
        Covise::update_choice_param(
            "field3", choicelist->get_num(), (char **)choicelist->get_strings(), 1);
    }
    return;
}
