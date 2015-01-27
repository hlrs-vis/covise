/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
**                                                           (C)1994 RUS  **
**                                                                        **
** Description:  COVISE ColorMap application module                       **
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
#include "MakeKdTree.h"

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

void Application::paramCallback(void *userData, void *callbackData)
{
    Application *thisApp = (Application *)userData;
    thisApp->param(callbackData);
}
//
//
//..........................................................................
//
//

//======================================================================
// Called before module exits
//======================================================================
void Application::quit(void *)
{
    //
    // ...... delete your data here .....
    //
    Covise::log_message(__LINE__, __FILE__, "Quitting now");
}

//======================================================================
// Computation routine (called when PARAM message arrrives)
//======================================================================
void Application::param(void *)
{
}

//======================================================================
// Computation routine (called when START message arrrives)
//======================================================================
void Application::compute(void *)
{

    coDistributedObject *data;
    coDistributedObject *grid;
    char *name;
    silent = 1;

    //	get input data object names
    name = Covise::get_object_name("grid");
    if (name == 0L)
    {
        Covise::sendError("ERROR: Object name not correct for 'Data'");
        return;
    }
    //	retrieve object from shared memeory
    grid = new coDistributedObject(name);

    //	get input data object names
    name = Covise::get_object_name("data");
    if (name == 0L)
    {
        Covise::sendError("ERROR: Object name not correct for 'energy'");
        return;
    }
    //	retrieve object from shared memeory
    data = new coDistributedObject(name);

    //	get output data object names
    name = Covise::get_object_name("KdTree");

    handle_objects(grid->createUnknown(), data->createUnknown(), name);
}

void Application::handle_objects(coDistributedObject *grid, coDistributedObject *data, char *Outname, coDistributedObject **set_out)
{

    coDoSet *D_set;
    coDistributedObject **set_objs;
    int i, set_num_elem, npoint;

    coDistributedObject **grid_objs;
    coDistributedObject **data_objs;
    coDoUnstructuredGrid *u_grid;
    coDoFloat *u_data = NULL;
    coDoFloat *s_data = NULL;
    coDistributedObject *data_out;
    DO_KdTree *KdTree;
    char buf[500];
    char *dataType;
    char *min_max;
    int *index;
    int sx, sy, sz;
    float *x_in, *y_in, *z_in;

    if (grid != 0L)
    {
        dataType = grid->getType();
        if (strcmp(dataType, "UNSGRD") == 0)
        {
            u_grid = (coDoUnstructuredGrid *)grid;
            u_grid->getGridSize(&numelem, &numconn, &numcoord);
            u_grid->get_adresses(&el, &cl, &x_in, &y_in, &z_in);
            u_grid->getTypeList(&tl);

            if (numelem == 0)
                Covise::sendWarning("WARNING: Data object 'Grid' is empty");
        }

        else if (strcmp(dataType, "SETELE") == 0)
        {
            grid_objs = ((coDoSet *)grid)->getAllElements(&set_num_elem);
            data_objs = ((coDoSet *)data)->getAllElements(&set_num_elem);
            set_objs = new coDistributedObject *[set_num_elem];
            set_objs[0] = NULL;
            for (i = 0; i < set_num_elem; i++)
            {
                sprintf(buf, "%s_%d", Outname, i);
                handle_objects(grid_objs[i], data_objs[i], buf, set_objs);
            }
            D_set = new coDoSet(Outname, set_objs);
            if (grid->getAttribute("TIMESTEP"))
            {
                D_set->addAttribute("TIMESTEP", "1 16");
            }
            if (set_out)
            {
                for (i = 0; set_out[i]; i++)
                    ;

                set_out[i] = D_set;
                set_out[i + 1] = NULL;
            }
            else
                delete D_set;
            delete ((coDoSet *)grid);
            delete ((coDoSet *)data);
            for (i = 0; set_objs[i]; i++)
                delete set_objs[i];
            delete[] set_objs;
            return;
        }

        else
        {
            Covise::sendError("ERROR: Data object 'Grid' has wrong data type");
            return;
        }
    }
    else
    {
#ifndef TOLERANT
        Covise::sendError("ERROR: Data object 'Grid' can't be accessed in shared memory");
#endif
        return;
    }
    if (grid != 0L)
    {
        dataType = data->getType();
        if (strcmp(dataType, "STRSDT") == 0)
        {
            s_data = (coDoFloat *)data;
            s_data->getGridSize(&sx, &sy, &sz);
            npoint = sx * sy * sz;
            s_data->get_adress(&s_in);

            if (npoint == 0)
                Covise::sendWarning("WARNING: Data object 'Data' is empty");
        }
        else if (strcmp(dataType, "USTSDT") == 0)
        {
            u_data = (coDoFloat *)data;
            npoint = u_data->getNumPoints();
            u_data->get_adress(&s_in);

            if (npoint == 0)
                Covise::sendWarning("WARNING: Data object 'Data' is empty");
        }
        else
        {
            Covise::sendError("ERROR: Data object 'Data' has wrong data type");
            return;
        }
    }
    //
    //      generate the output data objects
    //
    data_out = KdTree = new DO_KdTree(Outname, numelem);
    if (!KdTree->objectOk())
    {
        Covise::sendError("ERROR: creation of output object failed");
        return;
    }
    KdTree->get_adresses(&min_max, &index);
    kd_tree_t kd = make_kd_tree();
    compressed_kd_node *kt;
    kt = (compressed_kd_node *)kd.root;

    for (i = 0; i < numelem; i++)
    {
        min_max[i] = (kt[i]).min_max_indexes;
        index[i] = (kt[i]).index;
    }
    delete u_data;
    delete s_data;
    delete u_grid;

    //
    //      add objects to set
    //
    if (set_out)
    {
        for (i = 0; set_out[i]; i++)
            ;

        set_out[i] = data_out;
        set_out[i + 1] = NULL;
    }
    else
        delete data_out;
}

/******
    FIND_MIN_MAX_VOLUME
******/
/* Returns the minimum & maximum norms of a cell*/
void Application::find_min_max(int element, int *min, int *max)
{
    int i; /* Counter variable */
    int t_max, t_min; /* Temp variables to keep  */
    /* track of the minimum and */
    /* maximum data each cell */
    float temp; /* Holder for the norm of current vertex */

    t_max = 0;
    t_min = 0;
    for (i = 0; i < UnstructuredGrid_Num_Nodes[tl[element]]; i++)
    {
        temp = s_in[cl[el[element] + i]];
        if (s_in[cl[el[element] + t_max]] < temp)
            t_max = i;
        if (s_in[cl[el[element] + t_min]] > temp)
            t_min = i;
    }

    if (s_in[cl[el[element] + t_max]] > global_max)
        global_max = s_in[cl[el[element] + t_max]];
    if (s_in[cl[el[element] + t_min]] < global_min)
        global_min = s_in[cl[el[element] + t_min]];

    *max = t_max;
    *min = t_min;
} /* End of find_min_max */

/******
    SWAP
******/
/* Util function to swap two nodes. Used by partition. */
void Application::swap_nodes(compressed_kd_node *arg1, compressed_kd_node *arg2)
{
    compressed_kd_node temp;
    temp = *arg2;
    *arg2 = *arg1;
    *arg1 = temp;
}

/******
    PARTITION_AROUND_MEDIAN

    This takes an array and partitions it around the true median element,
    such that all the elements below the median element are less than the
    value of the median element, and all the elements above the median
    element are greater than the value of the median element.
    
    This non-recursive algorithm is adapted from Robert Sedgewick's
    "Algorithms in C++," page 128.
    
******/
/* Two versions, one for min, one for max */
void Application::partition_around_median_min(compressed_kd_node *data, int size)
{
    int i, j, left, right;
    int median = size / 2;
    compressed_kd_node cell;

    left = 0;
    right = size - 1;
    if (!silent)
        printf("entering parition_min(), size = %d\n", size);

    while (right > left)
    {
        cell = data[right];
        i = left - 1;
        j = right;

        for (;;)
        {
            do
            {
                ++i;
            } while (DATA_MIN(data[i]) < DATA_MIN(cell));
            do
            {
                --j;
            } while (DATA_MIN(data[j]) > DATA_MIN(cell) && (j > left));
            if (i >= j)
                break;
            swap_nodes(&data[i], &data[j]);
        }
        swap_nodes(&data[i], &data[right]);
        if (i >= median)
            right = i - 1;
        if (i <= median)
            left = i + 1;
    }
}

void Application::partition_around_median_max(compressed_kd_node *data, int size)
{
    register int i, j, left, right;
    int median = size / 2;
    compressed_kd_node cell;

    if (!silent)
        printf("entering parition_max(), size = %d\n", size);

    left = 0;
    right = size - 1;

    while (right > left)
    {
        cell = data[right];
        i = left - 1;
        j = right;
        /* i++; j--; */

        for (;;)
        {
            do
            {
                i++;
            } while (DATA_MAX(data[i]) < DATA_MAX(cell));
            do
            {
                j--;
            } while (DATA_MAX(data[j]) > DATA_MAX(cell) && (j > left));
            if (i >= j)
                break;
            swap_nodes(&data[i], &data[j]);
        }
        swap_nodes(&data[i], &data[right]);
        if (i >= median)
            right = i - 1;
        if (i <= median)
            left = i + 1;
    }
}

/******
    CHECK_MEDIAN
******/
/* Quicky check to see if partition worked */
/* Two version: one for min, one for max */

void Application::check_median_min(compressed_kd_node *kd_tree, int size)
{
    int i; /* Counters */
    compressed_kd_node median = kd_tree[size / 2];
    float median_min = DATA_MIN(median);

    for (i = 0; i < size / 2; i++)
        if (median_min < DATA_MIN(kd_tree[i]))
            printf("Median incorrect: size = %d, i = %d: %f is > %f at %d!!! (criterion min)\n",
                   size, i, DATA_MIN(kd_tree[i]),
                   DATA_MIN(median), size / 2);
    for (; i < size; i++)
        if (median_min > DATA_MIN(kd_tree[i]))
            printf("Median incorrect: size = %d, i = %d: %f is < %f at %d!!! (criterion min)\n",
                   size, i, DATA_MIN(kd_tree[i]),
                   DATA_MIN(median), size / 2);
}
void Application::check_median_max(compressed_kd_node *kd_tree, int size)
{
    int i; /* Counters */
    compressed_kd_node median = kd_tree[size / 2];
    float median_max = DATA_MAX(median);

    for (i = 0; i < size / 2; i++)
        if (median_max < DATA_MAX(kd_tree[i]))
            printf("Median incorrect: size = %d, i = %d: %f is > %f at %d!!! (criterion max)\n",
                   size, i, DATA_MAX(kd_tree[i]),
                   DATA_MAX(median), size / 2);
    for (; i < size; i++)
        if (median_max > DATA_MAX(kd_tree[i]))
            printf("Median incorrect: size = %d, i = %d: %f is < %f at %d!!! (criterion max)\n",
                   size, i, DATA_MAX(kd_tree[i]),
                   DATA_MAX(median), size / 2);
}

/******
    BUILD_KD_TREE
******/
/* This procedure builds a binary tree so that all of the leaves under a node's 
** right child are great than that node, and all the leaves under the left child
** are smaller.  Also, the node is the median of all of the leaves.  However, the
** tree alternates the variables that it sorts with, so one layer will be organized
** by the minimum, and the next by the maximum.
*/
void Application::build_kd_tree(compressed_kd_node *kd_tree, int num_cells, int criterion)
{
    if (!silent)
        printf("Entering build_kd_tree(), num_cells = %d\n", num_cells);
    if (num_cells > 1)
    {
        if (criterion == MAKE_KD_MIN)
            partition_around_median_min(kd_tree, num_cells);
        else if (criterion == MAKE_KD_MAX)
            partition_around_median_max(kd_tree, num_cells);

        /* Build the tree for both halves */
        criterion = !criterion;
        build_kd_tree(kd_tree, num_cells / 2, criterion);
        /* Second half does _not_ include median (num_cells/2) */
        build_kd_tree(kd_tree + num_cells / 2 + 1,
                      num_cells - num_cells / 2 - 1, criterion);
    }
    if (!silent)
        printf("Exiting build_kd_tree()\n");
} /* End of build_kd_tree */

/******
    CHECK_KD_TREE
******/
void Application::check_kd_tree(compressed_kd_node *kd_tree, int kd_tree_size, int criterion)
{
    compressed_kd_node *cur_node = kd_tree + kd_tree_size / 2;

    if (!silent)
        printf("Checking tree, size = %d\n", kd_tree_size);
    if (kd_tree_size > 0)
    {
        if (criterion == MAKE_KD_MIN)
            check_median_min(kd_tree, kd_tree_size);
        else if (criterion == MAKE_KD_MAX)
            check_median_max(kd_tree, kd_tree_size);

        criterion = !criterion;
        check_kd_tree(kd_tree, cur_node - kd_tree, criterion);
        check_kd_tree(cur_node + 1, kd_tree_size - (cur_node - kd_tree) - 1,
                      criterion);
    }
}

/******
    MAKE_KD_TREE
******/
/* This procedure creates and organizes a kd_tree. See build_kd_tree for 
   structural details */
kd_tree_t Application::make_kd_tree()
{
    kd_tree_t kd_tree;
    compressed_kd_node *kt; /* Holds the tree. Each */
    /* elemnt has a min, a max */
    /* and an index */
    int min_index, max_index; /* Index of min, max, vertex */
    int i; /* Counter variable */

    kd_tree.size = numelem;
    kd_tree.root = (kd_node *)malloc(kd_tree.size * sizeof(compressed_kd_node)); /* Allocate space */
    kt = (compressed_kd_node *)kd_tree.root;
    global_max = -MAXFLOAT;
    global_min = MAXFLOAT;

    /* Find the span of each cell */
    for (i = 0; i < numelem; i++)
    {
        find_min_max(i,
                     &min_index,
                     &max_index);
        MAKE_KD_SET_MIN(kt[i], min_index);
        MAKE_KD_SET_MAX(kt[i], max_index);
        MAKE_KD_SET_MIN(kt[i], min_index);
        MAKE_KD_SET_MAX(kt[i], max_index);

        kt[i].index = i;
    }

    printf("Maximum = %f, Minimum = %f\n", global_max, global_min);

    build_kd_tree((compressed_kd_node *)kd_tree.root, kd_tree.size, MAKE_KD_MIN);
    check_kd_tree((compressed_kd_node *)kd_tree.root, kd_tree.size, MAKE_KD_MIN);

    return kd_tree;
}
