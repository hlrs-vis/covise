/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _APPLICATION_H
#define _APPLICATION_H
/**************************************************************************\ 
 **                                                           (C)1995 RUS  **
 **                                                                        **
 ** Description:  COVISE Cell_Reduce application module                    **
 **                                                                        **
 **                                                                        **
 **                             (C) 1995                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Oliver Heck                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  16.05.95  V1.0                                                  **
\**************************************************************************/

#include <appl/ApplInterface.h>
using namespace covise;
#include <util/coviseCompat.h>

#define DATA_NONE 0
#define DATA_S 1
#define DATA_V 2
#define DATA_S_E 3
#define DATA_V_E 4

class Application
{

private:
    // callback stub functions
    //
    static void computeCallback(void *userData, void *callbackData);
    static void quitCallback(void *userData, void *callbackData);
    static void paramCallback(bool inMapLoading, void *userData, void *callbackData);
    // private member functions
    //

    int MEMORY_OPTIMIZED;
    float x_center, y_center, z_center;

    int norm_check(int v1, int v2, int v3, int v4 = -1);

    //  adjacency vars
    int cuc_count;
    int *cuc, *cuc_pos;
    float angle, n2x, n2y, n2z;

    int numelem, numelem_o, numconn, numcoord;
    int doDoubleCheck;
    int *el, *cl, *tl;

    float *x_in, *y_in, *z_in;
    float *u_in, *v_in, *w_in;
    int num_vert, DataType;
    int num_bar; // number of bar elements
    float *x_out, *y_out, *z_out;
    float *u_out, *v_out, *w_out;
    int num_conn, *conn_list, *conn_tag, num_elem, *elem_list, i_i;
    int *elemMap; // attach elements in line list to
    // grid element list
    int lnum_vert;
    float *lx_out, *ly_out, *lz_out, tresh;
    int lnum_conn, *lconn_list, lnum_elem, *lelem_list, i_l;
    float *lu_out, *lv_out, *lw_out;

    coDoUnstructuredGrid *tmp_grid;
    coDoLines *Lines;
    coDoPolygons *Polygons;
    coDoFloat *SOut;
    coDoVec3 *VOut;

    coDoFloat *SlinesOut;
    coDoVec3 *VlinesOut;

    coDoFloat *USIn;
    coDoVec3 *UVIn;
    coDoFloat *SIn;
    coDoVec3 *VIn;
    // Title of the module
    char *d_title;
    const char *getTitle()
    {
        return d_title;
    };
    void setTitle(const char *title);
    // initial Title of the module
    char *d_init_title;

    void compute(void *callbackData);

    void doModule(coDistributedObject *meshIn,
                  coDistributedObject *dataIn,
                  char *meshOutName,
                  char *dataOutName,
                  char *linesOutName,
                  char *ldataOutName,
                  coDistributedObject **meshOut,
                  coDistributedObject **dataOut,
                  coDistributedObject **linesOut,
                  coDistributedObject **ldataOut,
                  int masterLevel);

    void extractDouble();
    void quit(void *callbackData);
    void surface();
    void lines();
    int test(int, int, int);
    int add_vertex(int v);
    int ladd_vertex(int v);

public:
    Application(int argc, char *argv[]);

    void run()
    {
        Covise::main_loop();
    }

    ~Application()
    {
    }
};
#endif // _APPLICATION_H
