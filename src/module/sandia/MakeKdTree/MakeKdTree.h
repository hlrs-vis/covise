/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPLICATION_H
#define _APPLICATION_H

/**************************************************************************\
**                                                           (C)1994 RUS  **
**                                                                        **
** Description:  COVISE ColorMap application module                       **
**                                                                        **
**                                                                        **
**                                                                        **
**                             (C) 1994                                   **
**                Computer Center University of Stuttgart                 **
**                            Allmandring 30                              **
**                            70550 Stuttgart                             **
**                                                                        **
**                                                                        **
** Author:  R.Lang, D.Rantzau                                             **
**                                                                        **
**                                                                        **
** Date:  18.05.94  V1.0                                                  **
\**************************************************************************/

#include "ApplInterface.h"
#include "covise_kd_tree.h"
#include <stdlib.h>

#define MAKE_KD_MIN 0
#define MAKE_KD_MAX 1

#define GET_KD_MIN(a) (((a).min_max_indexes) >> 4)
#define GET_KD_MAX(a) (((a).min_max_indexes) & 0xF)

#define DATA_MIN(kd) (s_in[cl[el[kd.index] + GET_KD_MIN(kd)]])
#define DATA_MAX(kd) (s_in[cl[el[kd.index] + GET_KD_MAX(kd)]])

#define MAKE_KD_SET_MIN(k, min_i) ((k).min_max_indexes |= ((min_i) << 4))
#define MAKE_KD_SET_MAX(k, max_i) ((k).min_max_indexes |= (max_i))

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

/* A structure defining the whole tree */
typedef struct kd_tree_str
{
    kd_node *root;
    int size,
        compressed; /* Is the kd_tree compressed? */
} kd_tree_t;

class Application
{

private:
    // callback stub functions
    //
    static void computeCallback(void *userData, void *callbackData);
    static void paramCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);

    /******
    GLOBAL VARIABLES
******/

    int silent;
    float global_max, global_min;
    float *s_in;
    int *el, *cl, *tl;
    int numelem, numconn, numcoord;

    // private member functions
    //
    void compute(void *callbackData);
    void param(void *callbackData);
    void quit(void *callbackData);

    void find_min_max(int element, int *min, int *max);
    void swap_nodes(compressed_kd_node *arg1, compressed_kd_node *arg2);
    void partition_around_median_min(compressed_kd_node *data, int size);
    void partition_around_median_max(compressed_kd_node *data, int size);
    void check_median_min(compressed_kd_node *kd_tree, int size);
    void check_median_max(compressed_kd_node *kd_tree, int size);
    void build_kd_tree(compressed_kd_node *kd_tree, int num_cells, int criterion);
    void check_kd_tree(compressed_kd_node *kd_tree, int kd_tree_size, int criterion);
    kd_tree_t make_kd_tree();

    void handle_objects(coDistributedObject *grid, coDistributedObject *data, char *Outname, coDistributedObject **set_out = NULL);
    int calctype; //choice value (which output to generate)
public:
    Application(int argc, char *argv[])
    {
        silent = 0;
        Covise::set_module_description("Calculate KD Tree from Unstructured Grid and Scalar Data");
        Covise::add_port(INPUT_PORT, "grid", "Set_UnstructuredGrid", "Unstructured Grid");
        Covise::add_port(INPUT_PORT, "data", "Set_Float|Set_Float", "Scalar data for KD Tree");
        Covise::add_port(OUTPUT_PORT, "KdTree", "Set_KdTree", "KdTree");
        Covise::init(argc, argv);
        Covise::set_start_callback(Application::computeCallback, this);
        Covise::set_quit_callback(Application::quitCallback, this);
        Covise::set_param_callback(Application::paramCallback, this);
    }

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};

#endif // _APPLICATION_H
