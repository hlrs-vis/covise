/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TRI_POLYGON_H
#define TRI_POLYGON_H
/**************************************************************************\ 
 **                                                           (C)1997 RUS  **
 **                                                                        **
 ** Description:  Source Code for the triangulation of simple polygons     **
 **               without holes (Header file)                              **
 **                                                                        **
 **                                                                        **
 **                             (C) 1997                                   **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30a                             **
 **                            70550 Stuttgart                             **
 **                                                                        **
 **                                                                        **
 ** Author:  Karin Frank                                                   **
 **                                                                        **
 **                                                                        **
 ** Date:  September 1997  V1.0                                            **
\**************************************************************************/
#include <util/coviseCompat.h>
#include "My_Struct.h"
#include "PQ.h"

class TriPolygon
{
    friend class PolygonWithMidpoint;
    friend class MonotonePolygon;

public:
    enum
    {
        max_edges = 100
    };

private:
    typedef struct
    {
        int total;
        int to[max_edges];
        int used[max_edges];
        int tri[max_edges];
    } Diagonal;

    PQ<Vertex_with_coords> *heap;
    Tree *status;
    Vertex_with_coords *vertex_list;
    Diagonal *diag_list;

    int num_vert;
    int num_diag;

    double compute_inner_angle(int v1, int v, int v2);
    double IncreasesInteriorAnglePi(int v1, int v, int v2);
    inline int is_above(int v, int v1);
    inline void insert_diagonal(int v1, int v2);
    int detect_monotone_polygon(int &num, int *v);
    int get_next_diagonal(int &v, int &i);

    int HandleVertex(int v, double sweep);
    int HandleStartVertex(int v, double sweep);
    int HandleEndVertex(int v, double sweep);
    int HandleSplitVertex(int v, double sweep);
    int HandleMergeVertex(int v, double sweep);
    int HandleRegularVertex(int v, double sweep);

    int MakeMonotone();
    int HandleMonotone(int num, int *v_list, int &num_tri, int (*tri)[3], int optimize);

public:
    TriPolygon(){};
    TriPolygon(int num_v, float (*vertices)[2]);
    virtual ~TriPolygon();

    virtual int TriangulatePolygon(int (*triangles)[3], int optimize);
};

class MonotonePolygon : public TriPolygon
{
private:
    double Maximal_y_coordinate;
    int which_max;
    Stack *chain;

    inline int get_diag_index(int v, int v1);
    int detect_triangles(int (*triangles)[3]);
    double compute_angle_weight(int v0, int v1, int v2);
    void optimize_triangulation(int (*triangles)[3]);

public:
    MonotonePolygon(){};
    MonotonePolygon(int num_v, double (*vertices)[2]);
    ~MonotonePolygon();

    int TriangulatePolygon(int (*triangles)[3], int optimize);
};
#endif
