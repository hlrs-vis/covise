/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _READKDTREE_H
#define _READKDTREE_H
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

#include "ApplInterface.h"
#include "covise_kd_tree.h"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#define SET_MIN(k, min_i) ((k).min_max_indexes |= ((min_i) << 4))
#define SET_MAX(k, max_i) ((k).min_max_indexes |= (max_i))
#define COMPRESSED_FLAG 0xDDDD

typedef struct compressed_kd_node_str
{
    unsigned char min_max_indexes;
    int index;
} compressed_kd_node;

/* A structure defining each node in the tree */
typedef struct kd_node_str
{
    float min, max; /* span holds the min and */
    /* max distances of a cell */
    int index; /* index into the cell array */
} kd_node;

typedef struct kd_node_2_str
{
    float span[2]; /* span holds the min and */
    /* max distances of a cell */
    int index; /* index into the cell array */
} kd_node_2;

/* A structure defining the whole tree */
typedef struct kd_tree_str
{
    compressed_kd_node *root;
    int size,
        compressed; /* Is the kd_tree compressed? */
} kd_tree_t;

class Application
{

private:
    //  member functions
    void compute(void *callbackData);
    void quit(void *callbackData);

    //  Static callback stubs
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    //  Parameter names
    char *filename;
    char *KDTree;

    //  Local data

    //  Shared memory data
    DO_KdTree *KdTree;

public:
    Application(int argc, char *argv[])

    {

        KdTree = 0L;
        Covise::set_module_description("Read KD Tree");
        Covise::add_port(OUTPUT_PORT, "KdTree", "DO_KdTree", "KD Tree");
        Covise::add_port(PARIN, "filename", "Browser", "Data file path");
        Covise::set_port_default("filename", "data/sandia/test.kd *.kd*");
        Covise::init(argc, argv);
        Covise::set_quit_callback(Application::quitCallback, this);
        Covise::set_start_callback(Application::computeCallback, this);
    }

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};

#endif // _READIHS_H
