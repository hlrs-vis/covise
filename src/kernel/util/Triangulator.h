/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS   Triangulator
//
// Description: This class provides a triangulation algorithm
//
//
// Initial version: 11.12.2002 (CS)
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// All Rights Reserved.
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//
// $Id: Triangulator.h,v 1.4 2002/12/17 13:43:10 cs_te Exp $
//
#ifndef __TRIANGULATOR_INCLUDED
#define __TRIANGULATOR_INCLUDED

#include <math.h>

namespace covise
{

bool FP_EQUAL(double s, double t);
int math_logstar_n(int n);
static const double C_EPS = 1.0e-20; /* tolerance value: Used for making */
/* all decisions about collinearity or */
/* left/right of segment. Decrease */
/* this value if the input points are */
/* spaced very close together */

class _point_t;

typedef class _point_t
{
private:
public:
    double x;
    double CROSS_SINE(const _point_t &v1)
    {
        return (x * (v1).y - (v1).x * y);
    }
    double y;
    _point_t(double xx, double yy)
    {
        this->x = xx;
        this->y = yy;
    }
    _point_t()
    {
        this->x = 0.0;
        this->y = 0.0;
    }
    double LENGTH()
    {
        return (sqrt(x * x + y * y));
    }
    bool operator>(const _point_t &v1)
    {
        if (y > v1.y + C_EPS)
            return true;
        if (y < v1.y - C_EPS)
            return false;
        return (x > v1.x);
    }
    friend _point_t operator-(const _point_t &v0, const _point_t &v1)
    {
        return _point_t(v0.x - v1.x, v0.y - v1.y);
    }
    bool operator==(const _point_t &v1)
    {
        return (FP_EQUAL(y, v1.y) && FP_EQUAL(x, v1.x));
    }
    bool operator>=(const _point_t &v1)
    {
        if (y > v1.y + C_EPS)
            return true;
        if (y < v1.y - C_EPS)
            return false;
        return (x >= v1.x);
    }
    bool operator<(const _point_t &v1)
    {
        if (y < v1.y - C_EPS)
            return true;
        if (y > v1.y + C_EPS)
            return false;
        return (x < v1.x);
    }
    void setmax(const _point_t &v0, const _point_t &v1);
    void setmin(const _point_t &v0, const _point_t &v1);

} point_t, vector_t;

/* Segment attributes */

typedef struct
{
    point_t v0, v1; /* two endpoints */
    bool is_inserted; /* inserted in trapezoidation yet ? */
    int root0, root1; /* root nodes in Q */
    int next; /* Next logical segment */
    int prev; /* Previous segment */
} segment_t;

/* Trapezoid attributes */

typedef struct
{
    int lseg, rseg; /* two adjoining segments */
    point_t hi, lo; /* max/min y-values */
    int u0, u1;
    int d0, d1;
    int sink; /* pointer to corresponding in Q */
    int usave, uside; /* I forgot what this means */
    int state;
} trap_t;

/* Node attributes for every node in the query structure */

typedef class _node_t
{
public:
    _node_t()
    {
        yval = point_t(0.0, 0.0);
    }
    int nodetype; /* Y-node or S-node */
    int segnum;
    point_t yval;
    int trnum;
    int parent; /* doubly linked DAG */
    int left, right; /* children */
} node_t;

typedef struct
{
    int vnum;
    int next; /* Circularly linked list  */
    int prev; /* describing the monotone */
    int marked; /* polygon */
} monchain_t;

typedef struct
{
    point_t pt;
    int vnext[4]; /* next vertices for the 4 chains */
    int vpos[4]; /* position of v in the 4 chains */
    int nextfree;
} vertexchain_t;

static const int FIRSTPT = 1; /* checking whether pt. is inserted */
static const int LASTPT = 2;
static const int T_X = 1;
static const int T_Y = 2;
static const int T_SINK = 3;
#undef INFINITY
static const int INFINITY = 0x40000000;
static const int S_LEFT = 1; /* for merge-direction */
static const int S_RIGHT = 2;
static const int ST_VALID = 1; /* for trapezium state */
static const int ST_INVALID = 2;
static const int SP_SIMPLE_LRUP = 1; /* for splitting trapezoids */
static const int SP_SIMPLE_LRDN = 2;
static const int SP_2UP_2DN = 3;
static const int SP_2UP_LEFT = 4;
static const int SP_2UP_RIGHT = 5;
static const int SP_2DN_LEFT = 6;
static const int SP_2DN_RIGHT = 7;
static const int SP_NOSPLIT = -1;
static const int TR_FROM_UP = 1; /* for traverse-direction */
static const int TR_FROM_DN = 2;
static const int TRI_LHS = 1;
static const int TRI_RHS = 2;

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define CROSS(v0, v1, v2) (((v1).x - (v0).x) * ((v2).y - (v0).y) - ((v1).y - (v0).y) * ((v2).x - (v0).x))

#define DOT(v0, v1) ((v0).x * (v1).x + (v0).y * (v1).y)

class Triangulator
{
private:
    node_t *qs;
    trap_t *tr;
    segment_t *seg;
    monchain_t *mchain;
    vertexchain_t *vert; /* chain init. information. This */
    /* is used to decide which */
    /* monotone polygon to split if */
    /* there are several other */
    /* polygons touching at the same */
    /* vertex  */
    int *mon; /* contains position of any vertex in */
    /* the monotone chain for the polygon */
    int *visited;
    int choose_idx;
    int *permute;
    int chain_idx, op_idx, mon_idx;
    int q_idx;
    int tr_idx;
    int _segsize; /* max# of segments. Determines how */
    /* many points can be specified as */
    /* input. If your datasets have large */
    /* number of points, increase this */
    /* value accordingly. */
    int newnode(); /* Return a new node to be added into the query tree */
    int newtrap(); /* Return a free trapezoid */
    int init_query_structure(int segnum);
    int getSegSize()
    {
        return _segsize;
    }
    int getQSize() /* maximum table sizes */
    {
        return 4 * _segsize;
    }
    int getTRSize() /* max# trapezoids */
    {
        return 8 * _segsize;
    }
    bool is_left_of(int segnum, point_t &v);
    bool inserted(int segnum, int whichpt);
    int locate_endpoint(point_t &v, point_t &vo, int r);
    int merge_trapezoids(int segnum, int tfirst, int tlast, int side);
    int add_segment(int segnum);
    int find_new_roots(int segnum);
    int construct_trapezoids(int nseg);
    int inside_polygon(trap_t &t);
    /* return a new mon structure from the table */
    int newmon();
    /* return a new chain element from the table */
    int new_chain_element();
    double get_angle(const point_t &vp0, const point_t &vpnext, const point_t &vp1);
    int get_vertex_positions(int v0, int v1, int &ip, int &iq);
    int make_new_monotone_poly(int mcur, int v0, int v1);
    int monotonate_trapezoids(int n);
    int traverse_polygon(int mcur, int trnum, int from, int dir);
    int triangulate_monotone_polygons(int nvert, int nmonpoly, int op[][3]);
    int triangulate_single_polygon(int nvert, int posmax, int side, int op[][3]);
    int choose_segment();
    void initialize(int n);
    void generate_random_ordering(int n);
    bool isCCW(int size, double (*vertices)[2]);
    void revert(int size, double (*vertices)[2]);
    void makeCCW(int size, double (*vertices)[2]);
    void makeCW(int size, double (*vertices)[2]);

public:
    bool _gotError;
    Triangulator(int size);
    ~Triangulator();
    int triangulate_polygon(int ncontours, int cntr[], double (*vertices)[2], int (*triangles)[3]);
    static int getTriangles(int numVert, float *x, float *y, int (*triangles)[3]);
};
}
#endif
//
// History:
//
// $Log: Triangulator.h,v $
// Revision 1.4  2002/12/17 13:43:10  cs_te
// -
//
// Revision 1.3  2002/12/17 13:34:48  ralf
// adapted for Windows
//
// Revision 1.2  2002/12/16 14:16:31  cs_te
// -
//
// Revision 1.1  2002/12/12 11:59:24  cs_te
// initial version
//
//
