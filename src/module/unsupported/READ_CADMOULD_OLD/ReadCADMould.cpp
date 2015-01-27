/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)1999 RUS  **
 **   ReadCADMould module for Covise API 2.0                               **
 **                                                                        **
 ** Author:                                                                **
 **                           Ralph Bruckschen                             **
 **                            Vircinity GmbH                              **
 **                             Nobelstr. 15                               **
 **                            70550 Stuttgart                             **
 ** Date:  11.11.99  V0.1                                                  **
\**************************************************************************/

#include <iostream.h>
#include <fstream.h>
#include <strstream.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <netdb.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>

#include "ReadCADMould.h"

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Constructor : This will set up module port structure
/// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
ReadSAI::ReadSAI()
{

    // declare the name of our module
    set_module_description("ReadSAI reads in CADMould files");

    //klappt nicht :-( ????
    meshfile = addFileBrowserParam("Mesh_file", "select Meshfile");
    meshfile->setValue("data/meshfile", "*");
    datafile = addFileBrowserParam("Data_file", "select Datafile");
    datafile->setValue("data/datafile", "*");

    // add an output port for this type
    Grid_Out_Port = addOutputPort("Grid out", "Set_Polygons", "Grid Output Port");
    Line_Out_Port = addOutputPort("Lines out", "Set_Lines", "Lines Output Port");
    Animated_Grid_Out_Port = addOutputPort("Animated Polygons out", "Set_Polygons", "Animated Grid Output Port");

    Fill_Level_Data_Out_Port = addOutputPort("Fill Level Data out", "Set_Float", "Fill Level Data Output Port");
    Time_Data_Out_Port = addOutputPort("Time Data out", "Set_Float", "Time Data Output Port");
    Animated_Data_Out_Port = addOutputPort("Animated Data out", "Set_Float", "Animated Time Data Output Port");

    no_of_steps = addInt32Param("Timesteps", "No. of steps for fill animation");
    no_of_steps->setValue(25);

    dataset = NULL;
    // and that's all ... no init() or anything else ... that's done in the lib
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Compute callback: Called when the module is executed
// ++++
// ++++  NEVER use input/output ports or distributed objects anywhere
// ++++        else than inside this function
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int ReadSAI::compute()

{
    char buf[255];

    const char *meshfilename = meshfile->getValue();
    const char *datafilename = datafile->getValue();

    Covise::getname(buf, meshfilename);

    //        const char *meshfilename="/mnt/ext/pr/vir32402/rwb/sai/tr01.txt"; //meshfile->getValue();
    //        const char *datafilename="/mnt/ext/pr/vir32402/rwb/sai/tr01test.txt"; //datafile->getValue();
    //if(!dataset)
    //{
    dataset = new CadmouldData;
    //fprintf(stderr, "mesh: %s\ndata: %s\n", meshfilename, datafilename);
    if (dataset->load(meshfilename, datafilename) == FAIL)
    {
        sendError("Error reading files");
        delete dataset;
        dataset = NULL;
        return STOP_PIPELINE;
    }
    //}
    if (dataset)
    {
        int *vertexlist = new int[3 * dataset->no_elements];
        int *polygonlist = new int[dataset->no_elements];
        int i, j;
        int vert_count = 0;

        for (i = 0; i < dataset->no_elements; i++)
        {
            polygonlist[i] = vert_count;
            if (dataset->elem[2][i] == -1)
            {
                vertexlist[vert_count] = dataset->elem[0][i] - 1;
                vert_count++;
                vertexlist[vert_count] = dataset->elem[1][i] - 1;
                vert_count++;

                // add 3rd point same as 2nd if bar element
                vertexlist[vert_count] = dataset->elem[1][i] - 1;
                vert_count++;
            }
            else
            {
                vertexlist[vert_count] = dataset->elem[0][i] - 1;
                vert_count++;
                vertexlist[vert_count] = dataset->elem[1][i] - 1;
                vert_count++;
                vertexlist[vert_count] = dataset->elem[2][i] - 1;
                vert_count++;
            }
        }
        //fprintf(stderr,"vertices: %d points: %d elem: %d\n",vert_count,
        //        dataset->no_points, dataset->no_elements);

        int timesteps = no_of_steps->getValue();

        if (timesteps < 1)
        {
            Covise::sendWarning("Number of time steps not valid, set to default.");
            no_of_steps->setValue(25);
            timesteps = 25;
        }

        const char *poly_name = Grid_Out_Port->getObjName();
        const char *line_name = Line_Out_Port->getObjName();
        const char *animated_poly_name = Animated_Grid_Out_Port->getObjName();
        const char *fill_level_data_name = Fill_Level_Data_Out_Port->getObjName();
        const char *time_data_name = Time_Data_Out_Port->getObjName();
        const char *animated_data_name = Animated_Data_Out_Port->getObjName();

        coDoFloat *fill_level_data
            = new coDoFloat(fill_level_data_name,
                            dataset->no_points, dataset->value);

        coDoFloat *time_data
            = new coDoFloat(time_data_name,
                            dataset->no_points, dataset->fill_time);

        coDoPolygons *polys = new coDoPolygons(poly_name, dataset->no_points,
                                               dataset->points[0], dataset->points[1], dataset->points[2],
                                               vert_count, vertexlist, dataset->no_elements, polygonlist);
        polys->addAttribute("vertexOrder", "1");

        coDoLines *lines = new coDoLines(line_name, dataset->no_points,
                                         dataset->points[0], dataset->points[1], dataset->points[2],
                                         vert_count, vertexlist, dataset->no_elements, polygonlist);
        lines->addAttribute("vertexOrder", "1");

        coDistributedObject **animated_poly_list = new coDistributedObject *[timesteps + 1];
        coDistributedObject **animated_data_list = new coDistributedObject *[timesteps + 1];

        float curr_time;
        float max_time = 0.0;

        for (j = 0; j < dataset->no_points; j++)
            if (dataset->fill_time[j] > max_time)
                max_time = dataset->fill_time[j];

        //cerr << "maximum time: " << max_time << endl;
        for (i = 0; i < timesteps; i++)
        {
            float *fill_time;
            curr_time = i * max_time / (timesteps - 1.0);
            //cerr << " curr_time: " << curr_time << endl;
            animated_poly_list[i] = (coDistributedObject *)polys;
            polys->incRefCount();
            sprintf(buf, "%s_%d", animated_data_name, i);

            coDoFloat *sdata = new coDoFloat(buf, dataset->no_elements);
            animated_data_list[i] = sdata;
            sdata->getAddress(&fill_time);
            for (j = 0; j < dataset->no_elements; j++)
            {
                if ((j < dataset->no_elements - 1
                     && polygonlist[j + 1] - polygonlist[j] == 3)
                    || (j == dataset->no_elements - 1
                        && vert_count - polygonlist[j] == 3))
                {
                    int *start_vertex = &vertexlist[polygonlist[j]];
                    fill_time[j] = -2.0;
                    for (int k = 0; k < 3; k++)
                    {
                        if (dataset->fill_time[start_vertex[k]] > fill_time[j])
                        {
                            fill_time[j] = dataset->fill_time[start_vertex[k]];
                        }
                    }
                }
                if (fill_time[j] > curr_time)
                {
                    fill_time[j] = max_time;
                }
            }
        }
        animated_poly_list[i] = NULL;
        animated_data_list[i] = NULL;

        coDoSet *animated_polys = new coDoSet(animated_poly_name, animated_poly_list);
        coDoSet *animated_data = new coDoSet(animated_data_name, animated_data_list);
        sprintf(buf, "%d %d", 0, timesteps - 1);

        animated_polys->addAttribute("TIMESTEP", buf);
        animated_data->addAttribute("TIMESTEP", buf);

        // tell the output port that this is his object
        //cerr << "putting objects\n" ;

        // add species attributes
        animated_data->addAttribute("SPECIES", "Time");
        fill_level_data->addAttribute("SPECIES", "Level");
        time_data->addAttribute("SPECIES", "Time");

        // assign to ports
        Grid_Out_Port->setCurrentObject(polys);
        Line_Out_Port->setCurrentObject(lines);
        Animated_Grid_Out_Port->setCurrentObject(animated_polys);
        Fill_Level_Data_Out_Port->setCurrentObject(fill_level_data);
        Time_Data_Out_Port->setCurrentObject(time_data);
    }
    return CONTINUE_PIPELINE;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  Parameter callback: This one is called whenever an immediate
// ++++                      mode parameter is changed, but NOT for
// ++++                      non-immediate ports
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ReadSAI::param(const char *name)
{
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  postInst() is called once after we contacted Covise, but before
// ++++             getting into the main loop
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void ReadSAI::postInst()
{
    //cerr << "after Contruction" << endl;
}

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++++
// ++++  What's left to do for the Main program:
// ++++                                    create the module and start it
// ++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

int main(int argc, char *argv[])

{
    // create the module
    //cerr << "Constructor: ReadSAI *application = new ReadSAI;" << endl;
    ReadSAI *application = new ReadSAI;

    // this call leaves with exit(), so we ...
    //cerr << "Init + Main-Loop : application->start(argc,argv);" << endl;
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
