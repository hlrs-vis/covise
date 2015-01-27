/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description: Read module for Ihs data         	                  **
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
 ** Date:  17.07.97  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
#include "ReadCth.h"

#include <stdio.h>
#include "plotread.h"

Variable *find_var(PlotList *v_list, Data_Obj *dobj)
{
    List_Node *ptr;
    Variable *var;

    ptr = v_list->front;
    while (ptr != NULL)
    {
        var = (Variable *)ptr->data;
        if (strcmp(var->var_name, dobj->var_name) == 0)
            return (var);
        else
            ptr = ptr->next;
    }
    fprintf(stdout, "Couldn't find a match for data_obj var_name in variable list");

    return ((Variable *)NULL);
}

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
    int i, fd;
    char buf[600];
    char buf2[300];
    char dp[400];
    char dpend[100];
    int numt, currt = 0, t, endt, numvar, startb, numb, currb;
    Plotfile *p;
    List_Node *ptr;
    List_Node *ptr2;
    Timeslice *ts;
    Variable *var;
    Variable *current_var;
    Data_Obj *dobj;
    Struct_Block *sb;
    PlotList *ts_list, *do_list, *v_list;
    mesh = NULL;

    Covise::get_browser_param("data_path", &data_Path);
    Covise::get_scalar_param("numt", &numt);
    Covise::get_scalar_param("numb", &numb);
    Covise::get_scalar_param("currb", &currb);
    Covise::get_scalar_param("varnum", &numvar);
    strcpy(dp, data_Path);
    i = strlen(dp) - 1;
    while ((dp[i] < '0') && (dp[i] > '9'))
        i--;
    // dp[i] ist jetzt die letzte Ziffer, alles danach ist Endung
    strcpy(dpend, dp + i + 1); // dpend= Endung;
    dp[i + 1] = 0;
    while ((dp[i] >= '0') && (dp[i] <= '9'))
        i--;
    sscanf(dp + i + 1, "%d", &startb); //currt = Aktueller Zeitschritt
    endt = currt + numt;
    dp[i + 1] = 0; // dp = basename

    Mesh = Covise::get_object_name("mesh");
    Data = Covise::get_object_name("data");

    coDistributedObject **Mesh_sets = new coDistributedObject *[numt + 1];
    Mesh_sets[0] = NULL;
    coDistributedObject **Data_sets = new coDistributedObject *[numt + 1];
    Data_sets[0] = NULL;

    int alltime = 0;
    FILE *fp;
    sprintf(buf, "%s%d%s", dp, startb + currb, dpend);
    fp = Covise::fopen(buf, "r");
    if (fp == NULL)
    {
        strcpy(buf2, "ERROR: Can't open file >> ");
        strcat(buf2, buf);
        Covise::sendError(buf2);
        return;
    }
    p = Set_Plotfile(fp);
    if (!p)
    {
        strcpy(buf2, "ERROR reading file >> ");
        strcat(buf2, buf);
        Covise::sendError(buf2);
        return;
    }
    sprintf(buf, "  info: %s\n", p->info);
    Covise::sendInfo(buf);
    fprintf(stdout, "  variables:\n");
    fflush(stdout);
    v_list = Load_Variable_List(p->var_list_ptr);
    ptr = v_list->front;
    while (ptr != NULL)
    {
        var = (Variable *)ptr->data;
        fprintf(stdout, "    var_name: %-20s   type: %d   units: %s\n",
                var->var_name, var->type, var->units);
        ptr = ptr->next;
    }

    fprintf(stdout, "\n");
    fflush(stdout);

    ts_list = Load_Timeslice_List(p->timeslice_list_ptr);
    fprintf(stdout, "  timeslices:\n");
    ptr2 = ts_list->front;
    while (ptr2 != NULL)
    {
        ts = (Timeslice *)ptr2->data;
        fprintf(stdout, "    time: %g  cycle: %d  data_obj_list_ptr: %#o\n", ts->time, ts->cycle, ts->data_obj_list_ptr);
        //ts = (Timeslice *) ts_list->front->data;
        do_list = Load_Data_Obj_List(ts->data_obj_list_ptr);
        ptr = do_list->front;

        for (i = 0; i < numvar; i++)
            ptr = ptr->next; // skip numvar datasets

        if (ptr != NULL)
        {
            dobj = (Data_Obj *)ptr->data;
            fprintf(stdout, "    obj: %s  var: %s  obj_ptr: %#o\n", dobj->obj_name, dobj->var_name,
                    dobj->obj_ptr);

            current_var = find_var(v_list, dobj);

            sb = Load_Struct_Block(current_var, dobj);
            if (sb->domain->ndim != 3)
            {
                strcpy(buf, "ERROR: ndim !=3  ");
                Covise::sendError(buf);
                return;
            }
            xdim = sb->domain->size[0];
            ydim = sb->domain->size[1];
            zdim = sb->domain->size[2];
            float *xc, *yc, *zc;
            xc = sb->domain->coords;
            yc = xc + xdim;
            zc = yc + ydim;
            if (Mesh != NULL)
            {
                sprintf(buf, "%s_g_%d", Mesh, alltime);
                mesh = new coDoRectilinearGrid(buf, xdim, ydim, zdim, xc, yc, zc);
                if (mesh->objectOk())
                {
                    sprintf(buf, "%d", alltime);
                    mesh->addAttribute("TIMESTEP", buf);
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
                sprintf(buf, "%s_%d", Data, alltime);
                data = new coDoFloat(buf, xdim, ydim, zdim, (float *)sb->values);
                if (data->objectOk())
                {
                    sprintf(buf, "%d", alltime);
                    data->addAttribute("TIMESTEP", buf);
                    //data->getAddress(&s_data);
                    //read(fd,s_data,xdim*ydim*zdim*sizeof(float));
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
            for (i = 0; Mesh_sets[i]; i++)
                ;
            Mesh_sets[i] = mesh;
            Mesh_sets[i + 1] = NULL;
            for (i = 0; Data_sets[i]; i++)
                ;
            Data_sets[i] = data;
            Data_sets[i + 1] = NULL;

            Free_SB(sb);
        }
        Free_DO_List(do_list);
        alltime++;
        if (alltime >= numt)
            break;

        ptr2 = ptr2->next;
    }

    Close_Plotfile();
    Free_VAR_List(v_list);
    Free_TS_List(ts_list);
    Free_Plotfile(p);
    coDoSet *Mesh_set = new coDoSet(Mesh, Mesh_sets);
    coDoSet *Data_set = new coDoSet(Data, Data_sets);
    Mesh_set->addAttribute("TIMESTEP", "1 100");
    Data_set->addAttribute("TIMESTEP", "1 100");

    sprintf(buf, "%d %d %d", currb, numb, numt);
    Mesh_set->addAttribute("BLOCKINFO", buf);
    sprintf(buf, "B%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
    Mesh_set->addAttribute("READ_MODULE", buf);
    sprintf(buf, "%d %d %d", currb, numb, numt);
    Mesh_set->addAttribute("BLOCKINFO", buf);
    sprintf(buf, "B%s\n%s\n%s\n", Covise::get_module(), Covise::get_instance(), Covise::get_host());
    Mesh_set->addAttribute("READ_MODULE", buf);
    delete Mesh_sets[0];
    delete[] Mesh_sets;
    for (i = 0; Data_sets[i]; i++)
        delete Data_sets[i];
    delete[] Data_sets;
    delete Mesh_set;
    delete Data_set;
    delete mesh;
}
