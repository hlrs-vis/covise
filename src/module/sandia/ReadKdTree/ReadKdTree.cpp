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
** Date:  17.03.95  V1.0                                                  **
\**************************************************************************/

//#include "ApplInterface.h"
#include "ReadKdTree.h"

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
    FILE *input_file;
    kd_tree_t kd_tree;
    int compressed_flag;
    int i;
    char buf[500];

    // read input parameters and data object name

    Covise::get_browser_param("filename", &filename);

    KDTree = Covise::get_object_name("KdTree");

    if ((input_file = Covise::fopen(filename, "r")) == NULL)
    {
        strcpy(buf, "ERROR: Can't open file >> ");
        strcat(buf, filename);
        Covise::sendError(buf);
        return;
    }

    fread(&kd_tree.size, sizeof(int), 1, input_file);
    fread(&compressed_flag, sizeof(int), 1, input_file);
    if (compressed_flag == COMPRESSED_FLAG)
    {
        kd_tree.compressed = 1;
        kd_tree.root = (compressed_kd_node *)malloc(kd_tree.size * sizeof(compressed_kd_node));
        fread(kd_tree.root, sizeof(compressed_kd_node), kd_tree.size, input_file);
    }
    else
    {
        Covise::sendError("Sorry, can't read non compressed KdTrees");
        return;
    }
    fclose(input_file);

    if (KDTree != NULL)
    {
        KdTree = new DO_KdTree(KDTree, kd_tree.size);
        if (KdTree->objectOk())
        {
            char *mm;
            int *idx;
            KdTree->get_adresses(&mm, &idx);
            for (i = 0; i < kd_tree.size; i++)
            {
                *mm = kd_tree.root[i].min_max_indexes;
                *idx = kd_tree.root[i].index;
                mm++;
                idx++;
            }
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

    delete KdTree;
    free(kd_tree.root);
}
