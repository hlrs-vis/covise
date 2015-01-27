/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*RICARDO SQA =========================================================
* Status        : UNASSURED
* Module Name   : v2e
* Subject       : Vectis Phase 5 POST to Ensight Gold convertor
* Language      : ANSI C
* Requires      : RUtil 
* Documentation : README.html
* Filename      : v2e.c
* Author        : Adam Sampson (ASam)
* Creation Date : June 2000
* Last Modified : $Date: $
* Version       : $Revision: $
*======================================================================

   Phase 5 POST file to Ensight Gold converter
   ASam, May-June 2000

*/

/* ------------------- Configuration settings ------------------ */

/* Work as a standalone program (rather than functions to link into
   Vectis). This is unlikely to produce anything useful at the moment; 
   it's more to indicate which bits of code are specific to POST
   files. */
#define STANDALONE_CONVERTOR

/* Check that the correct amount of data has been read from Fortran
   records in POST files. */
#define CHECK_RECORD_ENDS

/* Generate "nsided" polygons for patches (rather than trias and
   quads).  This doesn't work very well with Ensight at the moment,
   because it doesn't support per-element variables on nsideds. Once
   Ensight gets proper nsided support, turning this option on will
   produce smaller, more accurate output files (for patches). */
/* #define GENERATE_NSIDED */

/* Show a progress meter while converting scalar cells. Unfortunately,
   the place where a progress meter is really needed is inside the
   qsort(3) call in sort_node_table; that is where the bulk of the
   user time of this program is spent. */
#define CELL_PROGRESS_METER

/* Show a progress meter while reading the input file. The meter is
   actually updated whenever a Fortran record header is read. */
#define INPUT_PROGRESS_METER

/* Warn if a face polygon or patch with less than 3 nodes is
   encountered. */
#define WARN_NUM_OF_NODES

/* ------------------ Machine dependencies ------------------ */

/* Byteswap data read from POST files; this is defined in the
   Makefile on a per-platform basis. */
/*#define NEED_TO_SWAP */

/* --- Typedefs for Fortran types --- */

/* A Fortran record header. */
typedef unsigned int fortran_header;
/* An INTEGER. */
typedef int fortran_integer;
/* A REAL*4. */
typedef float fortran_real4;
/* A CHARACTER. */
typedef char fortran_character;

/* Define BOOL, TRUE and FALSE if we don't already have them. */
#ifndef TRUE
typedef char BOOL;
#define FALSE 0
#define TRUE 1
#endif

/* ---------------------- Header files -------------------- */

/* Header files */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#ifndef WIN32
#include <unistd.h>
#endif

#include "RUtil.h"

#include "v2e_macros.h"
#include "v2e_utility.h"
#include "v2e_util.h"

/* ------ temp. COVISE data structures and routines to write *.covise files -------*/
#include "CoCTypes.h"
#include "CoCUtil.h"

CoC_UNSGRD CoUNSGRD;
/* translations for indicees */
CoC_Polyed_UNSGRD polyIdx;
CoC_Polyed_POLYGN patchIdx;

extern const int TYPE_HEXAGON;
extern const int TYPE_HEXAEDER;
extern const int TYPE_PRISM;
extern const int TYPE_PYRAMID;
extern const int TYPE_TETRAHEDER;
extern const int TYPE_QUAD;
extern const int TYPE_TRIANGLE;
extern const int TYPE_BAR;
extern const int TYPE_NONE;
extern const int TYPE_POINT;

FILE *covGeoF;

/* ---------------------- Globals ------------------- */

/* More Fortran types */
typedef fortran_integer *fortran_integer_array;
typedef fortran_real4 *fortran_real4_array;
/* 2D arrays are stored as 1D arrays. */
typedef fortran_real4 *fortran_real4_2darray;
typedef fortran_character *fortran_character_array;

/* Vectis general variables. These would presumably be declared in a
   header somewhere if this was working as a Vectis writer. */
fortran_integer ni, nj, nk;
fortran_real4 xmin, xmax, ymin, ymax, zmin, zmax;
fortran_real4_array xndim, yndim, zndim;
/* xgrid, ygrid, zgrid are calculated when needed; the use of
   [xyz]grid in the POST file is obsolete. */
fortran_integer icube, jcube, kcube;
fortran_integer ncells, nts, nbwss, nbess, nbsss, nbnss, nblss, nbhss;
fortran_integer_array iglobe, jglobe, kglobe;
fortran_integer_array ilpack, itpack;
fortran_integer_array ils, ile, jls, jle, kls, kle;
fortran_integer_array itypew, itypee, itypes, itypen, itypel, itypeh;
fortran_real4_array voln;
fortran_integer ncellu, ntu, ncellv, ntv, ncellw, ntw, nfpadr;
fortran_real4 iafactor, jafactor, kafactor;
fortran_real4_array areau, areav, areaw;
fortran_integer_array lwus, leus, lsvs, lnvs, llws, lhws, nfpol, lbfpol, lfpol;
fortran_integer nbpatch, nbound, nnode, nnodref;
fortran_integer_array ncpactual, mpatch, ltype, nodspp, lbnod, nodlist;

/* Variables declared per result set. Note that TIME is called
   steptime, to avoid colliding with the Unix system call of the same
   name. */
fortran_real4 pref, steptime, cangle;
fortran_integer ndrops;
fortran_real4_array p, den, t, mach, ps1, te, ed;
fortran_real4_2darray velcent;
fortran_real4_array ua, va, wa, amfu, amox, ampr, amin, amw5, amw6, amw7;
fortran_real4_array combpro, react_rate, ignprob, unt, bt, nox, sootmf, sootcon, radt, gasrads, droprads, absc, scats, fmsource;
fortran_real4_array uapatch, vapatch, wapatch, tpatch, tflpatch, gpatch;
fortran_real4_array taupatch, yppatch;
fortran_real4_array filmt, filmthick, filmu, filmv, filmw, radheatflux, nearwallv;
fortran_real4_array linkper, linkcellT, linkpatchT, linkheat, linkhtc;
fortran_real4_2darray pts;
/*
 fortran_real4_array voln, areau, areav, areaw;
 */

/* Droplet variables. These are initialised to NULL because we need to 
   check if they have already been allocated later. */
fortran_real4_array xdrop = NULL, ydrop = NULL, zdrop = NULL;
fortran_real4_array udrop = NULL, vdrop = NULL, wdrop = NULL;
fortran_real4_array dendr = NULL, tdrop = NULL, ddrop = NULL;
fortran_integer_array count = NULL, dstat = NULL, dhol = NULL;
fortran_integer_array ncdrop = NULL;

/* A constant for each variable. */
enum
{
    v_pref = 0,
    v_steptime,
    v_cangle,
    v_ndrops,
    v_p,
    v_den,
    v_t,
    v_mach,
    v_ps1,
    v_te,
    v_ed,
    v_velcent,
    v_combpro,
    v_react_rate,
    v_ignprob,
    v_unt,
    v_bt,
    v_nox,
    v_sootmf,
    v_sootcon,
    v_radt,
    v_gasrads,
    v_droprads,
    v_absc,
    v_scats,
    v_fmsource,
    v_ua,
    v_va,
    v_wa,
    v_amfu,
    v_amox,
    v_ampr,
    v_amin,
    v_amw5,
    v_amw6,
    v_amw7,
    v_uapatch,
    v_vapatch,
    v_wapatch,
    v_tpatch,
    v_tflpatch,
    v_gpatch,
    v_taupatch,
    v_yppatch,
    v_lppatch,
    v_lctpatch,
    v_lptpatch,
    v_lhpatch,
    v_lhtcpatch,
    v_filmt,
    v_filmthick,
    v_filmu,
    v_filmv,
    v_filmw,
    v_radheatflux,
    v_nearwallv,
    v_pts,
    v_voln,
    v_areau,
    v_areav,
    v_areaw,
    v_xdrop,
    v_ydrop,
    v_zdrop,
    v_udrop,
    v_vdrop,
    v_wdrop,
    v_dendr,
    v_tdrop,
    v_ddrop,
    v_count,
    v_dhol,
    v_ncdrop,
    num_varidents
} varident;

/* Array indicating whether the variable should appear in the output file. 
   This is keyed by the enumeration above. 
   If this code is integrated into Vectis, this array can be set from
   (or replaced with) the individual variables used. */
BOOL have_var[num_varidents],
    new_geometry = FALSE,
    any_new_geometry = FALSE,
    write_ascii = FALSE,
    write_binary = TRUE,
    write_boundaries = FALSE;

/* The Ensight part numbers to use. */
enum
{
    part_patches = 1,
    part_face_polygons,
    part_scalar_cells,
    part_droplets,
    part_boundaries
} partnumbers;

/* The details of a result set, as a linked list. */
struct _resultset
{
    fortran_real4 time;
    struct _resultset *_next;
};
typedef struct _resultset resultset;

/* The first and last result sets. */
resultset *first_result = NULL, *last_result;

/* A count of how many result sets we've encountered so far. */
int numresultsets = 0;

/* A count of how many distinct variables we've encountered
   so far. */
int numvariables = 0;

/* Counts of what types of element have been generated from
   a particular scalar cell in this timestep. */
int *numtetra, *numpyra, *numhexa;

int totaltetra, totalpyra, totalhexa;

/* The types of variable to put in the output files. */
enum
{
    evt_real_ncells = 0,
    evt_vector_ncells2d,
    evt_real_nbpatch,
    evt_u,
    evt_v,
    evt_vectorw_nbpatch,
    evt_vectorw_ndrops,
    evt_real_ndrops,
    evt_int_ndrops
} ensightvartype;

/* Information about an output variable that we've encountered. */
struct _ensightvardata
{
    int number;
};
typedef struct _ensightvardata ensightvardata;

/* Information about output variables that we might encounter. 
   The "data" field in this can be written to. */
struct _ensightvar
{
    int ident;
    char name[80];
    char type;
    void *var; /* pointer to the relevant variable */
    ensightvardata *data;
} ensightvars[] = {
    { v_p, "Cell Pressure", evt_real_ncells, &p, NULL },
    { v_den, "Cell Density", evt_real_ncells, &den, NULL },
    { v_t, "Cell Temperature", evt_real_ncells, &t, NULL },
    { v_mach, "Cell Mach Number", evt_real_ncells, &mach, NULL },
    { v_ps1, "Cell Passive Scalar", evt_real_ncells, &ps1, NULL },
    { v_te, "Cell Turb. Energy", evt_real_ncells, &te, NULL },
    { v_ed, "Cell Turb. Dissipation", evt_real_ncells, &ed, NULL },
    { v_velcent, "Cell Velocity", evt_vector_ncells2d, &velcent, NULL },
    { v_combpro, "Cell Combustion Progress", evt_real_ncells, &combpro, NULL },
    { v_react_rate, "Cell Reaction Rate", evt_real_ncells, &react_rate, NULL },
    { v_ignprob, "Cell Ignition Probability", evt_real_ncells, &ignprob, NULL },
    { v_unt, "Cell Unburned Temperature", evt_real_ncells, &unt, NULL },
    { v_bt, "Cell Burned Temperature", evt_real_ncells, &bt, NULL },
    { v_nox, "Cell NOx Mass Fraction", evt_real_ncells, &nox, NULL },
    { v_sootmf, "Cell Soot Mass Fraction", evt_real_ncells, &sootmf, NULL },
    { v_sootcon, "Cell Soot Concentration", evt_real_ncells, &sootcon, NULL },
    { v_radt, "Cell Radiation Temperature", evt_real_ncells, &radt, NULL },
    { v_gasrads, "Cell Gas Radiation Source", evt_real_ncells, &gasrads, NULL },
    { v_droprads, "Cell Droplet Radiation Source", evt_real_ncells, &droprads, NULL },
    { v_absc, "Cell Absorption Coefficient", evt_real_ncells, &absc, NULL },
    { v_scats, "Cell Scattering Coefficient", evt_real_ncells, &scats, NULL },
    { v_fmsource, "Cell FM Source", evt_real_ncells, &fmsource, NULL },
    /* Note that these can appear under two different names within a POST file. */
    { v_amfu, "Cell Species 1", evt_real_ncells, &amfu, NULL },
    { v_amox, "Cell Species 2", evt_real_ncells, &amox, NULL },
    { v_ampr, "Cell Species 3", evt_real_ncells, &ampr, NULL },
    { v_amin, "Cell Species 4", evt_real_ncells, &amin, NULL },
    { v_amw5, "Cell Species 5", evt_real_ncells, &amw5, NULL },
    { v_amw6, "Cell Species 6", evt_real_ncells, &amw6, NULL },
    { v_amw7, "Cell Species 7", evt_real_ncells, &amw7, NULL },
    /* ------------------------------------------------------------------------ */
    { v_uapatch, "", evt_u, &uapatch, NULL },
    { v_vapatch, "", evt_v, &vapatch, NULL },
    { v_wapatch, "Patch Velocity", evt_vectorw_nbpatch, &wapatch, NULL },
    { v_tpatch, "Patch Temperature", evt_real_nbpatch, &tpatch, NULL },
    { v_tflpatch, "Patch Fluid Temp.", evt_real_nbpatch, &tflpatch, NULL },
    { v_gpatch, "Patch HTC", evt_real_nbpatch, &gpatch, NULL },
    { v_taupatch, "Patch Shear", evt_real_nbpatch, &taupatch, NULL },
    { v_yppatch, "Patch Distance", evt_real_nbpatch, &yppatch, NULL },
    { v_filmt, "Patch Film Temperature", evt_real_nbpatch, &filmt, NULL },
    { v_filmthick, "Patch Film Thickness", evt_real_nbpatch, &filmthick, NULL },
    { v_filmu, "Patch Film U Velocity", evt_real_nbpatch, &filmu, NULL },
    { v_filmv, "Patch Film V Velocity", evt_real_nbpatch, &filmv, NULL },
    { v_filmw, "Patch Film W Velocity", evt_real_nbpatch, &filmw, NULL },
    { v_radheatflux, "Patch Radiation Heat Flux", evt_real_nbpatch, &radheatflux, NULL },
    { v_nearwallv, "Patch Near Wall Velocity", evt_real_nbpatch, &nearwallv, NULL },
    { v_lppatch, "Patch Linked Percentage", evt_real_nbpatch, &linkper, NULL },
    { v_lctpatch, "Patch Link Cell T", evt_real_nbpatch, &linkcellT, NULL },
    { v_lptpatch, "Patch Link Patch T", evt_real_nbpatch, &linkpatchT, NULL },
    { v_lhpatch, "Patch Link Heat Flux", evt_real_nbpatch, &linkheat, NULL },
    { v_lhtcpatch, "Patch Link HTC", evt_real_nbpatch, &linkhtc, NULL },
    { v_udrop, "", evt_u, &udrop, NULL },
    { v_vdrop, "", evt_v, &vdrop, NULL },
    { v_wdrop, "Droplet Velocity", evt_vectorw_ndrops, &wdrop, NULL },
    { v_dendr, "Droplet Density", evt_real_ndrops, &dendr, NULL },
    { v_tdrop, "Droplet Temperature", evt_real_ndrops, &tdrop, NULL },
    { v_ddrop, "Droplet Diameter", evt_real_ndrops, &ddrop, NULL },
    { v_count, "Droplet Count", evt_int_ndrops, &count, NULL },
    /*
   { v_dstat, "Droplet Status", evt_int_ndrops, &dstat, NULL },
   */
    { v_dhol, "Droplet Hole", evt_int_ndrops, &dhol, NULL },
    { v_ncdrop, "Droplet Cell Number", evt_int_ndrops, &ncdrop, NULL }
};
#define NUMENSIGHTVARS (sizeof(ensightvars) / sizeof(struct _ensightvar))

/* The backlink arrays (see write_scalar_cells for description). */
int *backlink, *backlink_index, *backlink_count;

/* A quick way to square a number. */
#define SQUARE(X) ((X) * (X))

/* message string */
char message[MAXLINE];

/* ------------------- Utility functions -------------------- */

/* Append the entire contents of file from onto the end of file to. 
   This lets us open up a file with mode "w+" and use it as temporary
   storage (rather like an M4 diversion). APPEND_BUFSIZE defines the
   size of the temporary buffer used while copying. */
#define APPEND_BUFSIZE 512 * 1024
void
append_file(FILE *to, FILE *from)
{
    long l;
    char buf[APPEND_BUFSIZE];

    fseek(from, 0, SEEK_SET);
    fseek(to, 0, SEEK_END);
    while (!feof(from))
    {
        l = fread(buf, sizeof(char), APPEND_BUFSIZE, from);
        fwrite(buf, sizeof(char), l, to);
    }
}

/* ------------------- Node cache -------------------- */

/* This is called _3dnode rather than _node, because RUtil declares
   an _node struct. */
struct _3dnode
{
    /* The coordinates of the node. */
    float x;
    float y;
    float z;

    /* A unique per-node ID that can be used to locate nodes in
     the node table. */
    int id;

    /* If >= 0: the index into the pts array from which this node came.
     If < 0:  -(1 + cellnum) where cellnum is the number of a cell
     of which this node is the centre. */
    int nodenum;
};
typedef struct _3dnode node;

/* --- The node table --- */

/* Node table global variables. node_table is a pointer to an array of
   nodes; node_table_size is the current size of the array;
   node_table_count is the number of items that are actually being
   used in the array; NODE_TABLE_BLOCK_SIZE is the granularity with
   which to allocate new blocks of memory for the array. need_to_write 
   is TRUE for nodes which should be written to the output file;
   node_table_to_write is the count of items in the node table which
   have need_to_write set. */
node *node_table = NULL;
int node_table_size, node_table_count, node_table_to_write;
BOOL *need_to_write = NULL;
#define NODE_TABLE_BLOCK_SIZE 1000

/* rewrite_table maps node_table node IDs to node IDs in the Ensight
   output file. */
int *rewrite_table = NULL;

#include "v2e.h"

/* ---------------------------------------------------------- */

/* Clear the node table, and allocate the first block. */
void
clear_node_table()
{
    free_node_table();

    node_table_size = NODE_TABLE_BLOCK_SIZE;
    node_table = (node *)RU_allocMem(sizeof(node) * node_table_size, "node table");
    node_table_count = 0;
}

/* Clear the node table, and allocate the first block. */
void
free_node_table()
{
    if (node_table)
        FREE(node_table);
    if (rewrite_table)
        FREE(rewrite_table);
    if (need_to_write)
        FREE(need_to_write);
}

/* ---------------------------------------------------------- */

/* Add a node to the node table. */
int
add_node(int nodenum, float x, float y, float z)
{

    /* Grow the node_table if we need to. */
    if ((node_table_count + 1) > node_table_size)
    {
        node_table_size += NODE_TABLE_BLOCK_SIZE;
        node_table = (node *)RU_reallocMem(node_table,
                                           sizeof(node) * node_table_size);
        if (!node_table)
            covise_message(ENSIGHT_FATAL_ERROR, "out of memory (realloc)");
    }

    node_table[node_table_count].x = x;
    node_table[node_table_count].y = y;
    node_table[node_table_count].z = z;
    node_table[node_table_count].nodenum = nodenum;
    node_table[node_table_count].id = node_table_count;

    return node_table_count++;
}

/* ---------------------------------------------------------- */

/* Add a node to the node table given its index into the lfpol or
   nodlist array. ispatch should be less than zero for patches. */
int
add_node_from_face(int ispatch, int lindex)
{
    int index;

    index = (ispatch < 0) ? (nodlist[lindex] - 1) : (lfpol[lindex] - 1);

    return add_node(index, pts[3 * index], pts[3 * index + 1], pts[3 * index + 2]);
}

/* ---------------------------------------------------------- */

/* Comparison function for nodes, compatible with qsort. */
int
node_comparison_function(const void *a, const void *b)
{
    if (((node *)a)->x < ((node *)b)->x)
        return -1;
    if (((node *)a)->x == ((node *)b)->x)
    {
        if (((node *)a)->y < ((node *)b)->y)
            return -1;
        if (((node *)a)->y == ((node *)b)->y)
        {
            if (((node *)a)->z < ((node *)b)->z)
                return -1;
            if (((node *)a)->z == ((node *)b)->z)
                return 0;
            return 1;
        }
        return 1;
    }
    return 1;
}

/* ---------------------------------------------------------- */

/* Sort the node table and fill in the rewrite_table. */
void
sort_node_table()
{
    int i;

    covise_message(ENSIGHT_INFO, "Calculating scalar cell nodes \n");

    /* Sort the node table. */
    sprintf(message, "sorting (%d nodes) \n", node_table_count);
    covise_message(ENSIGHT_INFO, message);

    qsort(node_table, node_table_count, sizeof(node), node_comparison_function);

    /* Allocate the rewrite and need_to_write tables. */
    rewrite_table = (int *)RU_allocMem(sizeof(int) * node_table_count, "rewrite_table");
    need_to_write = (BOOL *)RU_allocMem(sizeof(BOOL) * node_table_count, "need_to_write");

    /* We always need to write the first node. node_table_to_write is
     set to 1 because Ensight nodes need to be numbered from 1. */
    rewrite_table[node_table[0].id] = 1;
    need_to_write[0] = TRUE;
    node_table_to_write = 1;

    /* Iterate through the rest of the node table, filling in the
     rewrite table. */
    covise_message(ENSIGHT_INFO, "analysing\n");
    for (i = 1; i < node_table_count; i++)
    {

        if (node_table[i].x == node_table[i - 1].x
            && node_table[i].y == node_table[i - 1].y
            && node_table[i].z == node_table[i - 1].z)
        {

            /* The node was the same as the last one; we don't need
	 to write it out. */
            need_to_write[i] = FALSE;
        }
        else
        {

            /* This is a new node; we need to write it out and assign
	 it a new ID number. */
            need_to_write[i] = TRUE;
            node_table_to_write++;
        }

        /* Put an entry in the rewrite table. */
        rewrite_table[node_table[i].id] = node_table_to_write;
    }
}

/* ---------------------------------------------------------- */

/* Find a reasonable guess at a centre point for the cell by taking
   the mean of all the nodes around the cell. This is not a good
   method, as cells which have a lot of nodes at one point on their
   boundary will end up with a centre that isn't very centred.  To
   mitigate this, this does not currently include the patches in the
   average. It also fails on concave cells. */
BOOL
calculate_cell_centre(int cell, node *centre)
{
    int j, k, avgnum = 0;

    centre->x = centre->y = centre->z = 0.0;
    for (j = 0; j < backlink_count[cell]; j++)
    {
        int pol;

        /* Find the polygon number for this face. */
        pol = backlink[backlink_index[cell] + j];

        /* If this is a cell face polygon rather than a patch... */
        if (pol >= 0)
        {

            /* Iterate through the nodes on the polygon. */
            for (k = 0; k < nfpol[pol]; k++)
            {
                int pos;

                /* ... and find the node number for this node. */
                pos = (lfpol[lbfpol[pol] + k] - 1) * 3;

                /* Add the node to the running total. */
                centre->x += pts[pos];
                centre->y += pts[pos + 1];
                centre->z += pts[pos + 2];

                /* And increment the count of nodes. */
                avgnum++;
            }
        }
        else
        {
            node ncentre;
            int patch;

            /* This is a patch, so we need to find the centre of the patch
	 and add that to the total. */

            /* Clear the patch total. */
            ncentre.x = ncentre.y = ncentre.z = 0.0;

            /* Find the number of the patch. */
            patch = (-pol) - 1;

            /* Iterate through the nodes on the patch. */
            for (k = 0; k < nodspp[patch]; k++)
            {
                int nn;

                /* Add this node's coordinates onto the patch total. */
                nn = (nodlist[lbnod[patch] + k] - 1) * 3;
                ncentre.x += pts[nn];
                ncentre.y += pts[nn + 1];
                ncentre.z += pts[nn + 2];
            }

            /* Calculate the average and add that to the running total. */
            centre->x += ncentre.x / nodspp[patch];
            centre->y += ncentre.y / nodspp[patch];
            centre->z += ncentre.z / nodspp[patch];

            /* And increment the count. */
            avgnum++;
        }
    }

    /* If we didn't consider any nodes, then return FALSE to indicate
     that we failed. This means that the cell we were asked to find
     the centre of didn't actually have any nodes. */
    if (!avgnum)
        return FALSE;

    /* Divide the coordinates by the number of nodes counted to get
     the mean. */
    centre->x /= avgnum;
    centre->y /= avgnum;
    centre->z /= avgnum;

    return TRUE;
}

/* ------------------------------------------------------------ */

/* Update the coordinates in the node table from the pts array. 
   This only updates the nodes which would be written. */
void
update_node_table_coordinates()
{
    int i, nn;

    for (i = 0; i < node_table_count; i++)
    {
        if (!need_to_write[i])
            continue;

        nn = node_table[i].nodenum;
        if (nn < 0)
        {

            /* This node is the centre of a cell. Recalculate it. We don't
	 check to see if calculate_cell_centre failed, because the
	 entry would never have been put into the table if it failed
	 originally. */
            calculate_cell_centre((-nn) - 1, &node_table[i]);
        }
        else
        {

            /* This node came from the pts array. */
            node_table[i].x = pts[3 * nn];
            node_table[i].y = pts[3 * nn + 1];
            node_table[i].z = pts[3 * nn + 2];
        }
    }
}

/* ----------------- Ensight output functions ----------------- */

#ifndef GENERATE_NSIDED
/* Macros to calculate how many tria3s and quad4s a given patch
   will turn into. These are used later when writing out per-patch
   result variables. It's important that these stay in sync
   with the algorithm used in write_patches(). */
#define NUMTRIAS(X) (nodspp[X] % 2)
#define NUMQUADS(X) (nodspp[X] / 2 - 1)
#endif

/* ---------------------------------------------------------- */

/* Write a part for the patches to the output file. This is split
   out into a seperate function so that it can be used both when
   writing static geometry and when writing moving boundaries. */
void
write_patches(FILE *of)
{
    int i, j, num_tria, num_quad;
    FILE *fq, *ft;
    char message[MAXLINE];

    sprintf(message, "Writing %d patches \n", nbpatch);
    covise_message(ENSIGHT_INFO, message);

    /* Write out the part header */
    fprintf(of, "part\n");
    fprintf(of, "%10d\n", part_patches);
    fprintf(of, "All patches\n");

    /* Write out the nodes */
    fprintf(of, "coordinates\n");
    fprintf(of, "%10d\n", nnode);
    for (i = 0; i < 3; i++)
    { /* loop over x, y, z */
        for (j = 0; j < nnode; j++)
        { /* loop over nodes */
            fprintf(of, "%12.5e\n", pts[j * 3 + i]);
        }
    }

#ifdef GENERATE_NSIDED
    /* Write out the element type and the element IDs */
    fprintf(of, "nsided\n");
    fprintf(of, "%10d\n", nbpatch);
    for (i = 0; i < nbpatch; i++)
        fprintf(of, "%10d\n", i);

    /* Write out the number of nodes per element */
    for (i = 0; i < nbpatch; i++)
        fprintf(of, "%10d\n", nodspp[i]);

    /* Now write out the nodes */
    for (i = 0; i < nbpatch; i++)
    { /* loop over patches */
        for (j = 0; j < nodspp[i]; j++)
        { /* loop over patch nodes */
            fprintf(of, "%10d", nodlist[lbnod[i] + j]);
        }
        fprintf(of, "\n");
    }
#else
    /* Open temporary files to write to. We need to do this, because
     we can't write out the element definition until we know how many
     trias and quads we've generated. */
    fq = open_file("_quad", "w+");
    ft = open_file("_tria", "w+");

    /* Keep track of how many of each type are generated. */
    num_tria = num_quad = 0;

    /* Now iterate through the patches, writing them out to the temporary
     files as we go. This algorithm is similar to that used by the original
     Ensight reader; it assumes that all patches have more than two nodes. 
     If the algorithm is modified, it will also be necessary to modify the
     NUMTRIAS and NUMQUADS macros above to match it. */
    for (i = 0; i < nbpatch; i++)
    {
        int rem;
        fortran_integer *first, *cur;

        first = &nodlist[lbnod[i]];
        cur = first;
        cur++;

        /* The remaining number of nodes. */
        rem = nodspp[i];

        /* Reduce the number of nodes by twos by creating quad4s. */
        while (rem > 4)
        {
            ++num_quad;
            fprintf(fq, "%10d", *first);
            fprintf(fq, "%10d", *cur++);
            fprintf(fq, "%10d", *cur++);
            fprintf(fq, "%10d\n", *cur);
            rem -= 2;
        }

        /* Now create a final tria3 or quad4, depending upon
       how many nodes remain to be converted. */
        if (rem == 3)
        {
            ++num_tria;
            fprintf(ft, "%10d", *first);
            fprintf(ft, "%10d", *cur++);
            fprintf(ft, "%10d\n", *cur);
        }
        else
        { /* rem == 4 */
            ++num_quad;
            fprintf(fq, "%10d", *first);
            fprintf(fq, "%10d", *cur++);
            fprintf(fq, "%10d", *cur++);
            fprintf(fq, "%10d\n", *cur);
        }
    }

    /* Write out the quads */
    fprintf(of, "quad4\n%10d\n", num_quad);
    for (i = 0; i < num_quad; i++)
        fprintf(of, "%10d\n", i);
    append_file(of, fq);

    /* Write out the triangles */
    fprintf(of, "tria3\n%10d\n", num_tria);
    for (i = 0; i < num_tria; i++)
        fprintf(of, "%10d\n", i);
    append_file(of, ft);

    /* Close and delete the temporary files. */
    close_file(fq);
    close_file(ft);
    unlink_ensight_file("_quad");
    unlink_ensight_file("_tria");

#endif /* GENERATE_NSIDED */
}

/* ---------------------------------------------------------- */

/* Write a part for the patches to the output file. This is split
   out into a seperate function so that it can be used both when
   writing static geometry and when writing moving boundaries. */
void
write_binary_patches()
{

    FILE *coFco, *coFpo, *coF;

    char buffer[80], message[MAXLINE];
    char value[64];
    int i, j, num_tria, num_quad;
    float *x, *y, *z;

    int *polyArr;
    int polyCnt;
    int numPoly;
    int coPos[4];
    int coPosR[4];
    int ii;

    int verts;
    int triCnt, quadCnt;
    int *triIdx;
    int *quadIdx;
    int coAttrSize;
    int coNumAttr;
    int numCo;

    char coRTval[256];
    char coRTattr[80];

    sprintf(message, "Writing %d patches \n", nbpatch);
    covise_message(ENSIGHT_INFO, message);

    x = (float *)RU_allocMem(nnode * sizeof(float), "x");
    y = (float *)RU_allocMem(nnode * sizeof(float), "y");
    z = (float *)RU_allocMem(nnode * sizeof(float), "z");

    numCo = 0;
    for (j = 0; j < nnode; j++)
    { /* loop over nodes */
        x[j] = pts[j * 3];
        y[j] = pts[j * 3 + 1];
        z[j] = pts[j * 3 + 2];
        numCo++;
    }

    /* Keep track of how many of each type are generated. */
    num_tria = num_quad = 0;

    /* we use temp. files for the COVISE polygon list and the corner list */

    coFpo = fopen("_cov_poly", "wb+");
    coFco = fopen("_cov_corn", "wb+");

    /* Now iterate through the patches, writing them out to the temporary
     files as we go. This algorithm is similar to that used by the original
     Ensight reader; it assumes that all patches have more than two nodes. 
     If the algorithm is modified, it will also be necessary to modify the
     NUMTRIAS and NUMQUADS macros above to match it. */
    polyCnt = 0;
    numPoly = 0;
    fwrite(&polyCnt, sizeof(int), 1, coFpo);

    for (i = 0; i < nbpatch; i++)
    {
        int rem;
        fortran_integer *first, *cur;

        first = &nodlist[lbnod[i]];
        cur = first;
        cur++;

        /* The remaining number of nodes. */
        rem = nodspp[i];

        /* Reduce the number of nodes by twos by creating quad4s. */
        while (rem > 4)
        {
            ++num_quad;
            coPos[0] = *first;

            coPos[1] = *(cur++);

            coPos[2] = *(cur++);

            coPos[3] = *cur;

            rem -= 2;
            polyCnt += 4;
            numPoly++;
            fwrite(&polyCnt, sizeof(int), 1, coFpo);

            for (ii = 0; ii < 4; ++ii)
            {
                coPosR[3 - ii] = coPos[ii] - 1;
            }

            fwrite(coPosR, sizeof(int), 4, coFco);
        }

        /* Now create a final tria3 or quad4, depending upon
	 how many nodes remain to be converted. */
        if (rem == 3)
        {
            ++num_tria;

            coPos[0] = *first;

            coPos[1] = *(cur++);

            coPos[2] = *cur;

            polyCnt += 3;
            numPoly++;
            fwrite(&polyCnt, sizeof(int), 1, coFpo);

            for (ii = 0; ii < 3; ++ii)
            {
                coPosR[2 - ii] = coPos[ii] - 1;
            }
            fwrite(coPosR, sizeof(int), 3, coFco);
        }
        else
        { /* rem == 4 */
            ++num_quad;

            coPos[0] = *first;

            coPos[1] = *(cur++);

            coPos[2] = *(cur++);

            coPos[3] = *cur;

            polyCnt += 4;
            numPoly++;
            fwrite(&polyCnt, sizeof(int), 1, coFpo);

            for (ii = 0; ii < 4; ++ii)
            {
                coPosR[3 - ii] = coPos[ii] - 1;
            }
            fwrite(coPosR, sizeof(int), 4, coFco);
        }
    }

    /* write out COVISE file */
    coF = fopen("_v2c_patches", "ab+");

    strcpy(buffer, "POLYGN");
    fwrite(buffer, sizeof(char), 6, coF);
    fwrite(&numPoly, sizeof(int), 1, coF);
    fwrite(&polyCnt, sizeof(int), 1, coF);
    fwrite(&numCo, sizeof(int), 1, coF);

    /* 
     we have appended numPoly + 1 positions to the temp., file 
     but we want only the first numPoly ones  
  */
    polyArr = (int *)RU_allocMem(numPoly * sizeof(int), "polyArr");
    fseek(coFpo, 0, SEEK_SET);
    fread(polyArr, numPoly * sizeof(int), 1, coFpo);

    /*
    we use the availabkle polyeder table to generate an 
    index translation
  */

    triCnt = 0;
    quadCnt = 0;

    triIdx = (int *)RU_allocMem(num_tria * sizeof(int), "triIdx");
    quadIdx = (int *)RU_allocMem(num_quad * sizeof(int), "triIdx");

    for (ii = 0; ii < numPoly; ++ii)
    {

        if (ii < numPoly - 1)
        {
            verts = polyArr[ii + 1] - polyArr[ii];
        }
        else
        {
            verts = polyCnt - polyArr[ii];
        }

        switch (verts)
        {
        case 3:
            triIdx[triCnt] = ii;
            triCnt++;
            break;

        case 4:
            quadIdx[quadCnt] = ii;
            quadCnt++;
            break;
        default:
            fprintf(stderr, "   Something went wrong in the polygon list\n");
            break;
        };
    }

    patchIdx.numTri = triCnt;
    patchIdx.numQuad = quadCnt;

    if (patchIdx.triIdx)
        FREE(patchIdx.triIdx);
    patchIdx.triIdx = triIdx;

    if (patchIdx.quadIdx)
        FREE(patchIdx.quadIdx);
    patchIdx.quadIdx = quadIdx;

    fwrite(polyArr, sizeof(int), numPoly, coF);
    FREE(polyArr);

    append_file(coF, coFco);

    fwrite(x, sizeof(float), numCo, coF);
    fwrite(y, sizeof(float), numCo, coF);
    fwrite(z, sizeof(float), numCo, coF);

    /* add vertex order and REALTIME attribute */
    strcpy(buffer, "vertexOrder");
    sprintf(value, "%d", 2);

    strcpy(coRTattr, "REALTIME");
    sprintf(coRTval, "%30.15f", steptime);

    coAttrSize = sizeof(int) + 2 + strlen(buffer) + strlen(value);
    coAttrSize += 2 + strlen(coRTattr) + strlen(coRTval);

    coNumAttr = 2;

    fwrite(&coAttrSize, sizeof(int), 1, coF);
    fwrite(&coNumAttr, sizeof(int), 1, coF);
    fwrite(buffer, sizeof(char), strlen(buffer) + 1, coF);
    fwrite(value, sizeof(char), strlen(value) + 1, coF);
    fwrite(coRTattr, sizeof(char), strlen(coRTattr) + 1, coF);
    fwrite(coRTval, sizeof(char), strlen(coRTval) + 1, coF);

    fprintf(stderr, "           ....wrote COVISE patches\n");

    FREE(x);
    FREE(y);
    FREE(z);

    close_file(coF);
    close_file(coFco);
    close_file(coFpo);
    unlink("_cov_poly");
    unlink("_cov_corn");
}

/* ---------------------------------------------------------- */

void
write_binary_boundaries()
{

    char buffer[80], message[MAXLINE];
    int i, l;
    int bdy, patch, node;
    int num_tria, num_quad;
    float *x, *y, *z;
    int *numpatchs, *numnodes, *node_used;
    int part_number = part_boundaries;

    FILE *coFco, *coFpo, *coF;

    int *polyArr;
    int polyCnt;
    int numPoly;
    int coPos[4];
    int coPosR[4];
    int ii;

    int coAttrSize;
    int coNumAttr;
    int numCo;
    int bdCnt;

    char coRTval[256];
    char coRTattr[80];
    char coPRTval[256];
    char coPRTattr[80];
    char value[64];

    sprintf(message, "Writing %d patches on %d boundaries \n", nbpatch, nbound);
    covise_message(ENSIGHT_INFO, message);

    /* temporary boundary arrays */
    numpatchs = (int *)RU_allocMem(nbound * sizeof(int), "numpatchs");
    for (i = 0; i < nbound; i++)
        numpatchs[i] = 0;
    numnodes = (int *)RU_allocMem(nbound * sizeof(int), "numnodes");
    for (i = 0; i < nbound; i++)
        numnodes[i] = 0;
    node_used = (int *)RU_allocMem(nnode * sizeof(int), "node_used");

    /* all node coordinates */

    x = (float *)RU_allocMem(nnode * sizeof(float), "x");
    y = (float *)RU_allocMem(nnode * sizeof(float), "y");
    z = (float *)RU_allocMem(nnode * sizeof(float), "z");

    /* number of patches on each boundary */
    for (i = 0; i < nbpatch; i++)
        numpatchs[mpatch[i] - 1]++;

    /* open COVISE tmp file */
    coF = fopen("_v2c_boundaries", "ab+");

    bdCnt = 0;
    for (bdy = 0; bdy < nbound; bdy++)
    {
        if (numpatchs[bdy] > 0)
        {
            bdCnt++;
        }
    }

    writeCoSetHeader(coF, bdCnt);

    bdCnt = 0;
    for (bdy = 0; bdy < nbound; bdy++)
    {

        if (numpatchs[bdy] > 0)
        {

            bdCnt++;
            sprintf(message, "Writing %d patches on boundary %d \n", numpatchs[bdy], bdy + 1);
            covise_message(ENSIGHT_INFO, message);

            part_number++;

            sprintf(message, "Boundary %d Patches", bdy + 1);
            strcpy(buffer, message);

            /* Set the array of nodes used to zero */
            for (node = 0; node < nnode; node++)
                node_used[node] = 0;
            /* numnodes[bdy] = 0; */

            /* Open temporary files to write to. We need to do this, because
	       we can't write out the element definition until we know how many
	       trias and quads we've generated. */

            /* Keep track of how many of each type are generated. */
            num_tria = num_quad = 0;

            /* numnodes[bdy] = nnode; */

            coFpo = fopen("_cov_poly", "wb+");
            coFco = fopen("_cov_corn", "wb+");

            polyCnt = 0;
            numPoly = 0;
            fwrite(&polyCnt, sizeof(int), 1, coFpo);

            for (patch = 0; patch < nbpatch; patch++)
            { /* loop over patches */

                if (mpatch[patch] == bdy + 1)
                { /* patch on boundary */

                    for (l = 0; l < nodspp[patch]; l++)
                    { /* loop over the nodes on the patch */
                        fortran_integer node_number;

                        node_number = nodlist[lbnod[patch] + l]; /* get the node number */

                        node_used[node_number - 1] = 1; /* affect 1 each time the node is used, 0 default */
                    }
                }
            }

            numnodes[bdy] = 0;
            for (i = 0; i < nnode; i++)
            {

                if (node_used[i] == 1)
                {
                    numnodes[bdy]++;
                    node_used[i] = numnodes[bdy];
                    x[numnodes[bdy] - 1] = pts[i * 3];
                    y[numnodes[bdy] - 1] = pts[i * 3 + 1];
                    z[numnodes[bdy] - 1] = pts[i * 3 + 2];
                }
            }

            numCo = 1 + numnodes[bdy];

            for (patch = 0; patch < nbpatch; patch++)
            { /* loop over patches */

                int rem;
                fortran_integer *first, *cur;

                if (mpatch[patch] == bdy + 1)
                { /* patch on boundary */

                    first = &nodlist[lbnod[patch]];
                    cur = first;
                    cur++;

                    /* The remaining number of nodes. */
                    rem = nodspp[patch];

                    /* Reduce the number of nodes by twos by creating quad4s. */
                    while (rem > 4)
                    {
                        ++num_quad;
                        rem -= 2;

                        coPos[0] = node_used[*first - 1];

                        coPos[1] = node_used[*cur - 1];
                        cur++;

                        coPos[2] = node_used[*cur - 1];
                        cur++;

                        coPos[3] = node_used[*cur - 1];

                        polyCnt += 4;
                        numPoly++;
                        fwrite(&polyCnt, sizeof(int), 1, coFpo);

                        for (ii = 0; ii < 4; ++ii)
                        {
                            coPosR[3 - ii] = coPos[ii] - 1;
                            if (coPos[ii] == 0)
                            {
                                fprintf(stderr, "    iq  CORNER %d   %d\n", coPos[ii] - 1, ii);
                            }
                        }

                        fwrite(coPosR, sizeof(int), 4, coFco);
                    }

                    /* Now create a final tria3 or quad4, depending upon
		       how many nodes remain to be converted. */
                    if (rem == 3)
                    {
                        ++num_tria;

                        coPos[0] = node_used[*first - 1];

                        coPos[1] = node_used[*cur - 1];
                        cur++;

                        coPos[2] = node_used[*cur - 1];

                        polyCnt += 3;
                        numPoly++;
                        fwrite(&polyCnt, sizeof(int), 1, coFpo);

                        for (ii = 0; ii < 3; ++ii)
                        {
                            if (coPos[ii] == 0)
                            {
                                fprintf(stderr, "    ft  CORNER %d   %d\n", coPos[ii] - 1, ii);
                            }
                            coPosR[2 - ii] = coPos[ii] - 1;
                        }
                        fwrite(coPosR, sizeof(int), 3, coFco);
                    }

                    else
                    { /* rem == 4 */
                        ++num_quad;

                        coPos[0] = node_used[*first - 1];

                        coPos[1] = node_used[*cur - 1];
                        cur++;

                        coPos[2] = node_used[*cur - 1];
                        cur++;

                        coPos[3] = node_used[*cur - 1];

                        polyCnt += 4;
                        numPoly++;
                        fwrite(&polyCnt, sizeof(int), 1, coFpo);

                        for (ii = 0; ii < 4; ++ii)
                        {
                            coPosR[3 - ii] = coPos[ii] - 1;
                            if (coPos[ii] == 0)
                            {
                                fprintf(stderr, "    fq  CORNER %d   %d\n", coPos[ii] - 1, ii);
                            }
                        }

                        fwrite(coPosR, sizeof(int), 4, coFco);
                    }

                } /* mpatch == bdy+1 */

            } /* nbpatch */

            sprintf(message, "Writing %d nodes on boundary %d\n", numnodes[bdy], bdy + 1);
            covise_message(ENSIGHT_INFO, message);

            /* write out COVISE file */

            strcpy(buffer, "POLYGN");
            fwrite(buffer, sizeof(char), 6, coF);
            fwrite(&numPoly, sizeof(int), 1, coF);
            fwrite(&polyCnt, sizeof(int), 1, coF);
            fwrite(&numCo, sizeof(int), 1, coF);

            /* 
	       we have appended numPoly + 1 positions to the temp., file 
	       but we want only the first numPoly ones  
	    */
            polyArr = (int *)RU_allocMem(numPoly * sizeof(int), "polyArr");
            fseek(coFpo, 0, SEEK_SET);
            fread(polyArr, numPoly * sizeof(int), 1, coFpo);

            fwrite(polyArr, sizeof(int), numPoly, coF);
            FREE(polyArr);

            append_file(coF, coFco);

            fwrite(x, sizeof(float), numCo, coF);
            fwrite(y, sizeof(float), numCo, coF);
            fwrite(z, sizeof(float), numCo, coF);

            /* add vertex order and REALTIME attribute */
            strcpy(buffer, "vertexOrder");
            sprintf(value, "%d", 2);

            strcpy(coPRTattr, "PART");
            sprintf(coPRTval, "%d", bdCnt);

            coAttrSize = sizeof(int) + 2 + strlen(buffer) + strlen(value);
            coAttrSize += 2 + strlen(coPRTattr) + strlen(coPRTval);

            coNumAttr = 2;

            fwrite(&coAttrSize, sizeof(int), 1, coF);
            fwrite(&coNumAttr, sizeof(int), 1, coF);

            fwrite(buffer, sizeof(char), strlen(buffer) + 1, coF);
            fwrite(value, sizeof(char), strlen(value) + 1, coF);
            fwrite(coPRTattr, sizeof(char), strlen(coPRTattr) + 1, coF);
            fwrite(coPRTval, sizeof(char), strlen(coPRTval) + 1, coF);

            close_file(coFco);
            close_file(coFpo);
            unlink("_cov_poly");
            unlink("_cov_corn");

        } /* numpatchs > 0 */

    } /* nbound */

    strcpy(coRTattr, "REALTIME");
    sprintf(coRTval, "%30.15f", steptime);

    coAttrSize = sizeof(int) + 2 + strlen(coRTattr) + strlen(coRTval);

    coNumAttr = 1;

    fwrite(&coAttrSize, sizeof(int), 1, coF);
    fwrite(&coNumAttr, sizeof(int), 1, coF);

    fwrite(coRTattr, sizeof(char), strlen(coRTattr) + 1, coF);
    fwrite(coRTval, sizeof(char), strlen(coRTval) + 1, coF);

    close_file(coF);

    FREE(x);
    FREE(y);
    FREE(z);

    FREE(numpatchs);
    FREE(numnodes);
    FREE(node_used);
}

/* ---------------------------------------------------------- */

/* Write out the scalar cell connectivity based upon the cell face polygons. */

void
calculate_scalar_cell_connectivity(void)
{

    int *position, buf[8];
    int i, j, k, l, pt, numpoly, step;
    double max_delta;
    FILE *fe;

    covise_message(ENSIGHT_INFO, "Calculating scalar cell connectivity\n");

    /* Allocate the numX arrays if they haven't yet been allocated. */
    if (numtetra)
    {
        FREE(numtetra);
        FREE(numpyra);
        FREE(numhexa);
    }
    if (!numtetra)
    {
        numtetra = (int *)RU_allocMem(ncells * sizeof(int), "numtetra");
        numpyra = (int *)RU_allocMem(ncells * sizeof(int), "numpyra");
        numhexa = (int *)RU_allocMem(ncells * sizeof(int), "numhexa");
    }

    /* Calculate the maximum number of faces that can exist. */
    numpoly = 2 * (ntu + ntv + ntw) + nbpatch;

    /* Generate the backlink array, which serves the inverse function to
     the LWUS (etc.) arrays.

     backlink is an array of face numbers.  backlink_index is an array
     of indexes into the backlink array; backlink_count is a count of
     how many faces there are in that direction for each cell.  This
     means that if you have a cell N, there are backlink_count[N]
     faces, which are stored in the backlink array starting at
     position backlink_index[N]. 

     Positive values in backlink indicate cell face polygons; negative
     numbers indicate patches. Note that the patch numbers are 1-based,
     whereas the cell face numbers are 0-based; this avoids a collision
     at zero. */
    if (backlink)
    {
        FREE(backlink);
        FREE(backlink_index);
        FREE(backlink_count);
    }
    backlink = (int *)RU_allocMem(numpoly * sizeof(int), "backlink");
    backlink_index = (int *)RU_allocMem(nts * sizeof(int), "backlink_index");
    backlink_count = (int *)RU_allocMem(nts * sizeof(int), "backlink_count");

    /* Clear the count array for each cell. */
    for (i = 0; i < nts; i++)
        backlink_count[i] = 0;

    /* position is a temporary array, used to hold the position in the
     backlink array for each cell into which the next face encountered
     will be written. */
    position = (int *)RU_allocMem(nts * sizeof(int), "position");

    /* Count the number of faces for each cell. */
    for (i = 0; i < ntu; i++)
    {
        ++backlink_count[lwus[i] - 1];
        ++backlink_count[leus[i] - 1];
    }
    for (i = 0; i < ntv; i++)
    {
        ++backlink_count[lnvs[i] - 1];
        ++backlink_count[lsvs[i] - 1];
    }
    for (i = 0; i < ntw; i++)
    {
        ++backlink_count[llws[i] - 1];
        ++backlink_count[lhws[i] - 1];
    }
    for (i = 0; i < nbpatch; i++)
    {
        ++backlink_count[ncpactual[i] - 1];
    }

    /* Work out the positions in backlink for each cell based on the
     counts. */
    pt = 0;
    for (i = 0; i < nts; i++)
    {
        position[i] = pt;
        backlink_index[i] = pt;
        pt += backlink_count[i];
    }

    /* Now fill the backlink arrays with the face numbers matching each
     cell. */
    for (i = 0; i < ntu; i++)
        backlink[position[lwus[i] - 1]++] = i;
    for (i = 0; i < ntu; i++)
        backlink[position[leus[i] - 1]++] = i;
    for (i = 0; i < ntv; i++)
        backlink[position[lnvs[i] - 1]++] = ntu + i;
    for (i = 0; i < ntv; i++)
        backlink[position[lsvs[i] - 1]++] = ntu + i;
    for (i = 0; i < ntw; i++)
        backlink[position[llws[i] - 1]++] = ntu + ntv + i;
    for (i = 0; i < ntw; i++)
        backlink[position[lhws[i] - 1]++] = ntu + ntv + i;
    for (i = 0; i < nbpatch; i++)
        backlink[position[ncpactual[i] - 1]++] = -(i + 1);

    /* Clear the counts of generated elements. */
    totaltetra = totalpyra = totalhexa = 0;

    /* Clear the node table.*/
    clear_node_table();

#ifdef CELL_PROGRESS_METER
    /* Decide on what a reasonable step for the progress meter is. */
    step = ncells / 100;
    if (step > 1000)
        step = 1000;
#endif

    /* Open a temporary file to write element definitions to. */
    fe = open_file("_elements", "w+");

    /* Precalculate the maximum distance possible in the model. */
    max_delta = sqrt(SQUARE(xmax - xmin) + SQUARE(ymax - ymin)
                     + SQUARE(zmax - zmin));

    /* Iterate through the cells, generating appropriate Ensight elements. */
    for (i = 0; i < ncells; i++)
    {
        node centre;
        int centrenum, numfaces = 0;
        BOOL can_make_hexa = TRUE;

#ifdef CELL_PROGRESS_METER
        if (!(i % step))
        {
            fprintf(stderr, "%d / %d   %d%%\r", i, ncells, (100 * i) / ncells);
        }
#endif

        /* Clear the counts of what types of element were generated. */
        numtetra[i] = numpyra[i] = numhexa[i] = 0;

        /* Check to see if we can generate a hexahedron. Count the number
       of faces, checking to see if any are patches. */
        for (j = 0; j < backlink_count[i]; j++)
        {
            numfaces++;
            if (backlink[backlink_index[i] + j] < 0)
                can_make_hexa = FALSE;
        }

        /* If there is not any patch and more than 6 faces, generating an 
       hexa should still be possible. */
        /* if (numfaces > 6) can_make_hexa = FALSE; */

        if (can_make_hexa)
        {
            node corner[8], mini, maxi;
            double delta[8];
            BOOL initialised;

            /* We can generate a hexahedron.
   
	 We know that the cell consists entirely of face polygons, so
	 we can just iterate around the points on those to find the
	 bounds of the hexahedron to generate. However, in order to be 
	 able to handle changing geometry later, we need to match the
	 corners of the hexahedron to existing points in the pts
	 array. The plan of attack is therefore to generate all the
	 points we need using a bounding box, and then find the
	 closest corresponding points from the original faces. */

            /* Initialise the distances to the longest distance in the world 
	 (corner-to-corner). */
            for (j = 0; j < 8; j++)
            {
                delta[j] = max_delta;
                corner[j].nodenum = 0;
            }

            /* Find the bounding box. */
            initialised = FALSE;
            for (j = 0; j < backlink_count[i]; j++)
            {
                int pol;

                pol = backlink[backlink_index[i] + j];
                for (k = 0; k < nfpol[pol]; k++)
                {
                    int pos;

                    pos = (lfpol[lbfpol[pol] + k] - 1) * 3;

                    if (!initialised)
                    {
                        mini.x = maxi.x = pts[pos];
                        mini.y = maxi.y = pts[pos + 1];
                        mini.z = maxi.z = pts[pos + 2];
                        initialised = TRUE;
                    }
                    else
                    {
                        if (pts[pos] < mini.x)
                            mini.x = pts[pos];
                        if (pts[pos] > maxi.x)
                            maxi.x = pts[pos];
                        if (pts[pos + 1] < mini.y)
                            mini.y = pts[pos + 1];
                        if (pts[pos + 1] > maxi.y)
                            maxi.y = pts[pos + 1];
                        if (pts[pos + 2] < mini.z)
                            mini.z = pts[pos + 2];
                        if (pts[pos + 2] > maxi.z)
                            maxi.z = pts[pos + 2];
                    }
                }
            }

            /* Now generate the corners. */
            corner[0].x = mini.x;
            corner[0].y = mini.y;
            corner[0].z = maxi.z;
            corner[1].x = maxi.x;
            corner[1].y = mini.y;
            corner[1].z = maxi.z;
            corner[2].x = maxi.x;
            corner[2].y = mini.y;
            corner[2].z = mini.z;
            corner[3].x = mini.x;
            corner[3].y = mini.y;
            corner[3].z = mini.z;
            corner[4].x = mini.x;
            corner[4].y = maxi.y;
            corner[4].z = maxi.z;
            corner[5].x = maxi.x;
            corner[5].y = maxi.y;
            corner[5].z = maxi.z;
            corner[6].x = maxi.x;
            corner[6].y = maxi.y;
            corner[6].z = mini.z;
            corner[7].x = mini.x;
            corner[7].y = maxi.y;
            corner[7].z = mini.z;

            /* Iterate through all the nodes on the model, checking to see
	 how close each one is to the corner nodes. */
            for (j = 0; j < backlink_count[i]; j++)
            {
                int pol;

                /* Calculate the number of the current node. */
                pol = backlink[backlink_index[i] + j];

                for (k = 0; k < nfpol[pol]; k++)
                {
                    int pos;
                    double distance;

                    /* Calculate the location of the current node in pts. */
                    pos = lfpol[lbfpol[pol] + k] - 1;

                    for (l = 0; l < 8; l++)
                    {

                        /* Calculate the distance between the current node and the 
	       current corner. */
                        distance = sqrt(SQUARE(pts[pos * 3] - corner[l].x)
                                        + SQUARE(pts[pos * 3 + 1] - corner[l].y)
                                        + SQUARE(pts[pos * 3 + 2] - corner[l].z));

                        /* If it's less than the existing delta, then mark this
	       node in as the corresponding node for that corner. */
                        if (distance < delta[l])
                        {
                            delta[l] = distance;
                            corner[l].nodenum = pos;
                        }
                    }
                }
            }

            /* Finally, we can write out an element definition, allocating
	 nodes as we go. */
            fputc('h', fe);
            for (l = 0; l < 8; l++)
            {
                int num;

                /* Allocate the node, using the coordinates and node number of 
	   the closest node we found that matches it. */
                num = corner[l].nodenum;
                buf[l] = add_node(num, pts[num * 3], pts[num * 3 + 1], pts[num * 3 + 2]);
            }

            /* Write it out to the file. */
            fwrite(buf, sizeof(int), 8, fe);

            numhexa[i]++;
            totalhexa++;
        }
        else
        { /* (!can_make_hexa) */

            /* We can't generate a hexahedron; we need to split up the
	 cell. We check the return value from calculate_cell_centre,
	 because occasionally Vectis generates a cell which consists
	 entirely of faces with no nodes, in which case
	 calculate_cell_centre will return FALSE and we don't need to
	 generate any elements. */
            if (calculate_cell_centre(i, &centre))
            {

                /* If calculate_cell_centre succeeded, then we can go ahead
	   and add the node.  We pass -(i + 1) as the node number to
	   add_node, which indicates that the node represents the
	   centre of cell i. */
                centrenum = add_node(-(i + 1), centre.x, centre.y, centre.z);

                /* Now iterate through the faces, generating appropriate
	   elements.  This code is fairly ugly because it needs to
	   deal with both patches and cell face polygons. */
                for (j = 0; j < backlink_count[i]; j++)
                {
                    int remain, pol, node, patch;

                    /* Find the polygon number for this face. */
                    pol = backlink[backlink_index[i] + j];

                    /* If this is a patch, calculate the patch index. */
                    patch = (-pol) - 1;

                    /* Get the count of nodes remaining to be dealt with. */
                    remain = (pol < 0) ? (nodspp[patch]) : (nfpol[pol]);

                    /* If the face has less than 3 nodes, don't bother with it. */
                    if (remain < 3)
                    {
#ifdef WARN_NUM_OF_NODES
                        if (remain)
                        {
                            sprintf(message, "warning: polygon/patch %d on cell %d "
                                             "(num %d) only has %d nodes\n",
                                    j, i, pol, remain);
                            covise_message(ENSIGHT_WARNING, message);
                        }
#endif
                        continue;
                    }

                    /* Find the first node number. */
                    node = (pol < 0) ? (lbnod[patch]) : (lbfpol[pol]);
                    buf[0] = add_node_from_face(pol, node);
                    node++;

                    /* Generate pyras until we have less than 5 nodes left. */
                    while (remain > 4)
                    {
                        buf[1] = add_node_from_face(pol, node++);
                        buf[2] = add_node_from_face(pol, node++);
                        buf[3] = add_node_from_face(pol, node);
                        buf[4] = centrenum;

                        fputc('p', fe);
                        fwrite(buf, sizeof(int), 5, fe);

                        remain -= 2;
                        ++numpyra[i];
                        ++totalpyra;
                    }

                    /* Generate a final tetra or pyra. */
                    if (remain == 3)
                    {
                        buf[1] = add_node_from_face(pol, node++);
                        buf[2] = add_node_from_face(pol, node);
                        buf[3] = centrenum;

                        fputc('t', fe);
                        fwrite(buf, sizeof(int), 4, fe);

                        ++numtetra[i];
                        ++totaltetra;
                    }
                    else
                    { /* remain == 4 */

                        buf[1] = add_node_from_face(pol, node++);
                        buf[2] = add_node_from_face(pol, node++);
                        buf[3] = add_node_from_face(pol, node);
                        buf[4] = centrenum;

                        fputc('p', fe);
                        fwrite(buf, sizeof(int), 5, fe);

                        ++numpyra[i];
                        ++totalpyra;
                    }
                }
            }
        }
    }

    /* Sort the node table and fill in the rewrite array. */
    sort_node_table();

    /* Generate node connectivity */
    if (write_ascii)
    {
        FILE *f;
        f = open_file("_cellconn", "w");

        generate_node_connectivity(f, fe);

        close_file(f);
    }

    /* Generate node connectivity */
    if (write_binary)
    {
        FILE *f;
        f = open_file("_cellconn", "wb");

        generate_binary_node_connectivity(f, fe);

        close_file(f);
    }

    close_file(fe);
    unlink_ensight_file("_elements");

    FREE(position);
}

/* ---------------------------------------------------------- */

void
generate_node_connectivity(FILE *fo, FILE *fe)
{
    FILE *fp, *ft, *fh;
    int i;

    covise_message(ENSIGHT_INFO, "Generating node connectivity\n");

    /* Open temporary files to write the generated elements to. */
    ft = open_file("_tetra", "w+");
    fp = open_file("_pyra", "w+");
    fh = open_file("_hexa", "w+");

    /* Rewind the elements file. */
    fseek(fe, 0, SEEK_SET);

    /* Convert elements from the elements file and the rewrite table. */
    for (i = 0; i < (totaltetra + totalpyra + totalhexa); i++)
    {
        int buf[8], type;

        /* Get the type of the element. */
        type = fgetc(fe);

        /* Read the node numbers and write them out. */
        switch (type)
        {
        case 't':
            /* A tetrahedron. */
            fread(buf, sizeof(int), 4, fe);
            fprintf(ft, "%10d%10d%10d%10d\n",
                    rewrite_table[buf[0]],
                    rewrite_table[buf[1]],
                    rewrite_table[buf[2]],
                    rewrite_table[buf[3]]);
            break;

        case 'p':
            /* A pyramid. */
            fread(buf, sizeof(int), 5, fe);
            fprintf(fp, "%10d%10d%10d%10d%10d\n",
                    rewrite_table[buf[0]],
                    rewrite_table[buf[1]],
                    rewrite_table[buf[2]],
                    rewrite_table[buf[3]],
                    rewrite_table[buf[4]]);
            break;

        case 'h':
            /* A hexahedron. */
            fread(buf, sizeof(int), 8, fe);
            fprintf(fh, "%10d%10d%10d%10d%10d%10d%10d%10d\n",
                    rewrite_table[buf[0]],
                    rewrite_table[buf[1]],
                    rewrite_table[buf[2]],
                    rewrite_table[buf[3]],
                    rewrite_table[buf[4]],
                    rewrite_table[buf[5]],
                    rewrite_table[buf[6]],
                    rewrite_table[buf[7]]);
            break;

        default:
            covise_message(ENSIGHT_FATAL_ERROR, "internal error: unknown element type encountered in _elements\n");
        }
    }

    /* Write out the tetras. */
    fprintf(fo, "tetra4\n%10d\n", totaltetra);
    for (i = 0; i < totaltetra; i++)
        fprintf(fo, "%10d\n", i);
    append_file(fo, ft);

    /* Write out the pyras. */
    fprintf(fo, "pyramid5\n%10d\n", totalpyra);
    for (i = 0; i < totalpyra; i++)
        fprintf(fo, "%10d\n", i);
    append_file(fo, fp);

    /* Write out the hexas. */
    fprintf(fo, "hexa8\n%10d\n", totalhexa);
    for (i = 0; i < totalhexa; i++)
        fprintf(fo, "%10d\n", i);
    append_file(fo, fh);

    /* Close and unlink the temporary files. */
    close_file(ft);
    unlink_ensight_file("_tetra");
    close_file(fp);
    unlink_ensight_file("_pyra");
    close_file(fh);
    unlink_ensight_file("_hexa");
}
/* ---------------------------------------------------------- */

void
generate_binary_node_connectivity(FILE *fo, FILE *fe)
{
    char buffer[80];
    FILE *fp, *ft, *fh;
    int i, j, *id;

    int numElem;
    int numConn;
    int elemCnt;
    int lastElem;
    int connLCnt;
    int *type_l;
    int *elem_l;
    int *conn_l;

    int cntPyra, cntTetra, cntHexa;
    int *pyraTrans;
    int *tetraTrans;
    int *hexaTrans;

    covise_message(ENSIGHT_INFO, "Generating node connectivity\n");

    /* Open temporary files to write the generated elements to. */
    ft = open_file("_tetrab", "wb+");
    fp = open_file("_pyrab", "wb+");
    fh = open_file("_hexab", "wb+");

    /* Rewind the elements file. */
    fseek(fe, 0, SEEK_SET);

    /*------- generate UNSGR lists ----------*/
    numElem = totaltetra + totalpyra + totalhexa;
    numConn = 4 * totaltetra + 5 * totalpyra + 8 * totalhexa;

    type_l = (int *)RU_allocMem(numElem * sizeof(int), "TYPE_L");
    elem_l = (int *)RU_allocMem(numElem * sizeof(int), "ELEM_L");
    conn_l = (int *)RU_allocMem(numConn * sizeof(int), "CONN_L");

    pyraTrans = (int *)RU_allocMem(totalpyra * sizeof(int), "PYRATRANS");
    hexaTrans = (int *)RU_allocMem(totalhexa * sizeof(int), "HEXATRANS");
    tetraTrans = (int *)RU_allocMem(totaltetra * sizeof(int), "TETRATRANS");

    elem_l[0] = 0;
    lastElem = 0;
    connLCnt = 0;
    elemCnt = 0;

    cntPyra = 0;
    cntHexa = 0;
    cntTetra = 0;

    /* Convert elements from the elements file and the rewrite table. */
    for (i = 0; i < (totaltetra + totalpyra + totalhexa); i++)
    {
        int buf[8], type;
        int tetra[4], pyra[5], hexa[8];
        int notLast;

        notLast = (i + 1 != numElem);

        /* Get the type of the element. */
        type = fgetc(fe);

        /* Read the node numbers and write them out. */
        switch (type)
        {
        case 't':
            /* A tetrahedron. */
            fread(buf, sizeof(int), 4, fe);
            for (j = 0; j < 4; j++)
            {
                tetra[j] = rewrite_table[buf[j]];
            }
            fwrite(tetra, sizeof(int), 4, ft);

            /* we don't want any degenerated polyhedra */
            if (checkPolyeder(tetra, 4) == 4)
            {
                type_l[elemCnt] = TYPE_TETRAHEDER;
                tetraTrans[cntTetra] = elemCnt;
                elemCnt++;
                if (notLast)
                {
                    lastElem += 4;
                    elem_l[elemCnt] = lastElem;
                }

                for (j = 0; j < 4; j++)
                {
                    conn_l[connLCnt] = tetra[j] - 1;
                    connLCnt++;
                }
            }
            else
            {
                tetraTrans[cntTetra] = -1;
            }
            cntTetra++;

            break;

        case 'p':
            /* A pyramid. */
            fread(buf, sizeof(int), 5, fe);
            for (j = 0; j < 5; j++)
            {
                pyra[j] = rewrite_table[buf[j]];
            }
            fwrite(pyra, sizeof(int), 5, fp);

            /* we don't want totally degenerated polyhedra */
            if (checkPolyeder(pyra, 5) == 5)
            {
                type_l[elemCnt] = TYPE_PYRAMID;
                pyraTrans[cntPyra] = elemCnt;
                elemCnt++;
                if (notLast)
                {
                    lastElem += 5;
                    elem_l[elemCnt] = lastElem;
                }

                for (j = 0; j < 5; j++)
                {
                    int pidx;
                    switch (j)
                    {
                    case 0:
                        pidx = 1;
                        break;
                    case 1:
                        pidx = 0;
                        break;
                    case 2:
                        pidx = 3;
                        break;
                    case 3:
                        pidx = 2;
                        break;
                    case 4:
                        pidx = 4;
                        break;
                    }
                    conn_l[connLCnt] = pyra[pidx] - 1;
                    connLCnt++;
                }
            }
            else
            {
                pyraTrans[cntPyra] = -1;
            }
            cntPyra++;
            break;

        case 'h':
            /* A hexahedron. */
            fread(buf, sizeof(int), 8, fe);
            for (j = 0; j < 8; j++)
            {
                hexa[j] = rewrite_table[buf[j]];
            }
            fwrite(hexa, sizeof(int), 8, fh);

            /* we don't want totally degenerated polyhedra */
            if (checkPolyeder(hexa, 8) == 8)
            {
                type_l[elemCnt] = TYPE_HEXAEDER;
                hexaTrans[cntHexa] = elemCnt;
                elemCnt++;
                if (notLast)
                {
                    lastElem += 8;
                    elem_l[elemCnt] = lastElem;
                }

                for (j = 0; j < 8; j++)
                {
                    conn_l[connLCnt] = hexa[j] - 1;
                    connLCnt++;
                }
            }
            else
            {
                hexaTrans[cntHexa] = -1;
            }
            cntHexa++;
            break;

        default:
            covise_message(ENSIGHT_FATAL_ERROR, "internal error: unknown element type encountered in _elements\n");
        }
    }

    fprintf(stderr, "        .....generate_binary_node_connectivity -fill CoUNSGRD\n");

    if (CoUNSGRD.elem_l)
    {
        FREE(CoUNSGRD.elem_l);
    }
    CoUNSGRD.numElem = elemCnt;
    CoUNSGRD.elem_l = elem_l;

    if (CoUNSGRD.type_l)
    {
        FREE(CoUNSGRD.type_l);
    }
    CoUNSGRD.type_l = type_l;

    if (CoUNSGRD.conn_l)
    {
        FREE(CoUNSGRD.conn_l)
    }
    CoUNSGRD.numConn = connLCnt;
    CoUNSGRD.conn_l = conn_l;

    polyIdx.numPyra = cntPyra;
    polyIdx.numHexa = cntHexa;
    polyIdx.numTetra = cntTetra;

    if (polyIdx.pyraIdx)
        FREE(polyIdx.pyraIdx);
    polyIdx.pyraIdx = pyraTrans;

    if (polyIdx.hexaIdx)
        FREE(polyIdx.hexaIdx);
    polyIdx.hexaIdx = hexaTrans;

    if (polyIdx.tetraIdx)
        FREE(polyIdx.tetraIdx);
    polyIdx.tetraIdx = tetraTrans;

    /* Write out the tetras. */
    strcpy(buffer, "tetra4");
    fwrite(buffer, sizeof(char), 80, fo);

    fwrite(&totaltetra, sizeof(int), 1, fo);

    id = (int *)RU_allocMem(totaltetra * sizeof(int), "id");

    for (i = 0; i < totaltetra; i++)
        id[i] = i;

    if (fwrite(id, sizeof(int), totaltetra, fo) != totaltetra)
        covise_message(ENSIGHT_WARNING, "fwrite error for tetra4\n");

    FREE(id);

    append_file(fo, ft);

    /* Write out the pyras. */
    strcpy(buffer, "pyramid5");
    fwrite(buffer, sizeof(char), 80, fo);

    fwrite(&totalpyra, sizeof(int), 1, fo);

    id = (int *)RU_allocMem(totalpyra * sizeof(int), "id");

    for (i = 0; i < totalpyra; i++)
        id[i] = i;

    if (fwrite(id, sizeof(int), totalpyra, fo) != totalpyra)
        covise_message(ENSIGHT_WARNING, "fwrite error for pyramid8\n");

    FREE(id);

    append_file(fo, fp);

    /* Write out the hexas. */
    strcpy(buffer, "hexa8");
    fwrite(buffer, sizeof(char), 80, fo);

    fwrite(&totalhexa, sizeof(int), 1, fo);

    id = (int *)RU_allocMem(totalhexa * sizeof(int), "id");

    for (i = 0; i < totalhexa; i++)
        id[i] = i;

    if (fwrite(id, sizeof(int), totalhexa, fo) != totalhexa)
        covise_message(ENSIGHT_WARNING, "fwrite error for hexa8\n");

    FREE(id);

    append_file(fo, fh);

    sprintf(message, "Writing %d tetras, %d pyras and %d hexas \n", totaltetra, totalpyra, totalhexa);
    covise_message(ENSIGHT_INFO, message);

    /* Close and unlink the temporary files. */
    close_file(ft);
    unlink_ensight_file("_tetrab");
    close_file(fp);
    unlink_ensight_file("_pyrab");
    close_file(fh);
    unlink_ensight_file("_hexab");
}

/* ------------------------------------------------------------- */

/* Write out the scalar cells part for a timestep given the
   scalar cells connectivity file. */
void
write_scalar_cells(FILE *of, FILE *fc)
{
    int i;

    covise_message(ENSIGHT_INFO, "Writing scalar cells\n");

    /* Write out the part header. */
    fprintf(of, "part\n");
    fprintf(of, "%10d\n", part_scalar_cells);
    fprintf(of, "Scalar cells\n");

    /* Fill in the coordinates in the node table from the latest
     pts array. */
    update_node_table_coordinates();

    /* Write out the coordinates from the node table to the output
     file. */
    fprintf(of, "coordinates\n");
    fprintf(of, "%10d\n", node_table_to_write);

    sprintf(message, "Writing %d nodes of %d total\n",
            node_table_to_write, node_table_count);
    covise_message(ENSIGHT_INFO, message);

    for (i = 0; i < node_table_count; i++)
    {
        if (need_to_write[i])
        {
            fprintf(of, "%12.5e\n", node_table[i].x);
        }
    }
    for (i = 0; i < node_table_count; i++)
    {
        if (need_to_write[i])
        {
            fprintf(of, "%12.5e\n", node_table[i].y);
        }
    }
    for (i = 0; i < node_table_count; i++)
    {
        if (need_to_write[i])
        {
            fprintf(of, "%12.5e\n", node_table[i].z);
        }
    }

    /* Copy the element definitions from the connectivity file. */
    append_file(of, fc);
}

/* ---------------------------------------------------------- */

/* Write out the scalar cells part for a timestep given the
   scalar cells connectivity file. */
void
write_binary_scalar_cells()
{

    int count, i;

    float *xco, *yco, *zco;

    covise_message(ENSIGHT_INFO, "Writing binary scalar cells\n");

    /* Fill in the coordinates in the node table from the latest
     pts array. */
    update_node_table_coordinates();

    xco = (float *)RU_allocMem((size_t)(sizeof(float) * node_table_to_write), "node table x");
    yco = (float *)RU_allocMem((size_t)(sizeof(float) * node_table_to_write), "node table y");
    zco = (float *)RU_allocMem((size_t)(sizeof(float) * node_table_to_write), "node table z");

    count = 0;

    for (i = 0; i < node_table_count; i++)
    {
        if (need_to_write[i])
        {
            xco[count] = node_table[i].x;
            yco[count] = node_table[i].y;
            zco[count] = node_table[i].z;
            count++;
        }
    }

    fprintf(stderr, "          ....setting CoUNSGRD node_table_to_write: %d   count %d \n", node_table_to_write, count);

    CoUNSGRD.numCoords = count;

    if (CoUNSGRD.x_coord)
        FREE(CoUNSGRD.x_coord);
    CoUNSGRD.x_coord = xco;

    if (CoUNSGRD.y_coord)
        FREE(CoUNSGRD.y_coord);
    CoUNSGRD.y_coord = yco;

    if (CoUNSGRD.z_coord)
        FREE(CoUNSGRD.z_coord);
    CoUNSGRD.z_coord = zco;
}

/* ---------------------------------------------------------- */

/* Write out the droplet elements. */
void
write_droplets(FILE *of)
{
    int i, nactive;
    char message[MAXLINE];

    /* Check whether the model actually contains any droplets,
     and give up now if it doesn't. */
    if ((!have_var[v_ndrops]) || !ndrops)
        return;

    sprintf(message, "Writing %d droplets \n", ndrops);
    covise_message(ENSIGHT_INFO, message);

    /* Write out the part header. */
    fprintf(of, "part\n");
    fprintf(of, "%10d\n", part_droplets);
    fprintf(of, "Droplets\n");

    /* Count the number of active droplets. */
    nactive = 0;
    for (i = 0; i < ndrops; i++)
        if (!dstat[i])
            nactive++;

    /* Write out the droplet locations for all the active droplets. */
    fprintf(of, "coordinates\n");
    fprintf(of, "%10d\n", nactive);
    for (i = 0; i < ndrops; i++)
        if (!dstat[i])
            fprintf(of, "%12.5e\n", xdrop[i]);
    for (i = 0; i < ndrops; i++)
        if (!dstat[i])
            fprintf(of, "%12.5e\n", ydrop[i]);
    for (i = 0; i < ndrops; i++)
        if (!dstat[i])
            fprintf(of, "%12.5e\n", zdrop[i]);

    /* Write out the element header. */
    fprintf(of, "point\n");
    fprintf(of, "%10d\n", nactive);

    /* Write out the element IDs. */
    for (i = 0; i < nactive; i++)
        fprintf(of, "%10d\n", i + 1);

    /* Write out the element definitions. */
    for (i = 0; i < nactive; i++)
        fprintf(of, "%10d\n", i + 1);
}

/* ---------------------------------------------------------- */

/* Write out the droplet elements. */
void
write_binary_droplets()
{

    FILE *coF;

    char buffer[80];
    int i, nactive;
    int active_drop_count;
    float *x, *y, *z;

    int coAttrSize;
    int coNumAttr;

    char coRTval[256];
    char coRTattr[80];

    covise_message(ENSIGHT_INFO, "Writing binary droplets\n");

    /* Check whether the model actually contains any droplets,
     and give up now if it doesn't. */
    if ((!have_var[v_ndrops]) || !ndrops)
        return;

    /* write out COVISE file */
    coF = fopen("_v2c_drops", "ab+");

    strcpy(buffer, "POINTS");
    fwrite(buffer, sizeof(char), 6, coF);

    /* Count the number of active droplets. */
    nactive = 0;
    for (i = 0; i < ndrops; i++)
        if (!dstat[i])
            nactive++;

    x = (float *)RU_allocMem(sizeof(float) * nactive, "x drop");
    y = (float *)RU_allocMem(sizeof(float) * nactive, "y drop");
    z = (float *)RU_allocMem(sizeof(float) * nactive, "z drop");

    active_drop_count = 0;

    for (i = 0; i < ndrops; i++)
    {
        if (!dstat[i])
        {
            x[active_drop_count] = xdrop[i];
            y[active_drop_count] = ydrop[i];
            z[active_drop_count] = zdrop[i];
            active_drop_count++;
        }
    }

    fwrite(&nactive, sizeof(int), 1, coF);
    fwrite(x, sizeof(float), nactive, coF);
    fwrite(y, sizeof(float), nactive, coF);
    fwrite(z, sizeof(float), nactive, coF);

    FREE(x);
    FREE(y);
    FREE(z);

    /* add vertex order and REALTIME attribute */

    strcpy(coRTattr, "REALTIME");
    sprintf(coRTval, "%30.15f", steptime);

    coAttrSize = sizeof(int);
    coAttrSize += 2 + strlen(coRTattr) + strlen(coRTval);

    coNumAttr = 1;

    fwrite(&coAttrSize, sizeof(int), 1, coF);
    fwrite(&coNumAttr, sizeof(int), 1, coF);
    fwrite(coRTattr, sizeof(char), strlen(coRTattr) + 1, coF);
    fwrite(coRTval, sizeof(char), strlen(coRTval) + 1, coF);
}

/* ---------------------------------------------------------- */

/* Write out the results to the appropriate files. */
void
write_result_vars()
{
    int i, j, k, l;
    void *lastu = 0, *lastv = 0;
    char filename[80];
    FILE *f;

    covise_message(ENSIGHT_INFO, "Writing result variables: ");

    /* Iterate through the possible result variables, checking to see if
     we need to write each one. */
    for (i = 0; i < NUMENSIGHTVARS; i++)
    {
        if (have_var[ensightvars[i].ident])
        {

            /* If this variable hasn't yet been encountered, allocate a new
	 data struct for it. */
            if (!ensightvars[i].data)
            {
                ensightvars[i].data = (ensightvardata *)RU_allocMem(sizeof(ensightvardata), "ensightvarsdata");

                /* Assign the variable a new number. */
                ensightvars[i].data->number = numvariables++;
                fprintf(stderr, "+");
            }
            else
            {
                fprintf(stderr, ".");
            }

            /* Open the variable file for this variable. */
            sprintf(filename, "var%d.%04d",
                    ensightvars[i].data->number, numresultsets);
            f = open_ensight_file(filename, "w");

            /* Write out the variable's name. */
            fprintf(f, "%s\n", ensightvars[i].name);

            /* Write out the value of the variable, depending upon its type. */
            switch (ensightvars[i].type)
            {

            /* One fortran_real4 value per cell. */
            case evt_real_ncells:
                fprintf(f, "part\n%10d\n", part_scalar_cells);
                fprintf(f, "tetra4\n");
                for (j = 0; j < ncells; j++)
                {
                    for (k = 0; k < numtetra[j]; k++)
                    {
                        fprintf(f, "%12.5e\n",
                                (*(fortran_real4 **)ensightvars[i].var)[j]);
                    }
                }
                fprintf(f, "pyramid5\n");
                for (j = 0; j < ncells; j++)
                {
                    for (k = 0; k < numpyra[j]; k++)
                    {
                        fprintf(f, "%12.5e\n",
                                (*(fortran_real4 **)ensightvars[i].var)[j]);
                    }
                }
                fprintf(f, "hexa8\n");
                for (j = 0; j < ncells; j++)
                {
                    for (k = 0; k < numhexa[j]; k++)
                    {
                        fprintf(f, "%12.5e\n",
                                (*(fortran_real4 **)ensightvars[i].var)[j]);
                    }
                }
                break;

            /* One fortran_real4 value per patch. */
            case evt_real_nbpatch:
                fprintf(f, "part\n%10d\n", part_patches);
#ifdef GENERATE_NSIDED
                fprintf(f, "nsided\n");
                for (j = 0; j < nbpatch; j++)
                {
                    fprintf(f, "%12.5e\n",
                            (*(fortran_real4 **)ensightvars[i].var)[j]);
                }
#else
                /* Generate repeated values for quads and trias. */
                fprintf(f, "quad4\n");
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMQUADS(j); k++)
                    {
                        fprintf(f, "%12.5e\n",
                                (*(fortran_real4 **)ensightvars[i].var)[j]);
                    }
                }
                fprintf(f, "tria3\n");
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMTRIAS(j); k++)
                    {
                        fprintf(f, "%12.5e\n",
                                (*(fortran_real4 **)ensightvars[i].var)[j]);
                    }
                }
#endif
                break;

            /* This is for the particular case where the var is an array
	   with values {X1, Y1, Z1, X2, Y2, Z2... Xn, Yn, Zn} where n
	   is the number of cells.  Unfortunately, Ensight expects {
	   X1 ... Xn, Y1 ... Yn, Z1 ... Zn }. */
            case evt_vector_ncells2d:
                fprintf(f, "part\n%10d\n", part_scalar_cells);
                fprintf(f, "tetra4\n");
                for (j = 0; j < 3; j++)
                { /* iterate over vector components */
                    for (k = 0; k < ncells; k++)
                    { /* iterate over cells */
                        for (l = 0; l < numtetra[k]; l++)
                        {
                            fprintf(f, "%12.5e\n",
                                    (*(fortran_real4 **)ensightvars[i].var)[k * 3 + j]);
                        }
                    }
                }
                fprintf(f, "pyramid5\n");
                for (j = 0; j < 3; j++)
                { /* iterate over vector components */
                    for (k = 0; k < ncells; k++)
                    { /* iterate over cells */
                        for (l = 0; l < numpyra[k]; l++)
                        {
                            fprintf(f, "%12.5e\n",
                                    (*(fortran_real4 **)ensightvars[i].var)[k * 3 + j]);
                        }
                    }
                }
                fprintf(f, "hexa8\n");
                for (j = 0; j < 3; j++)
                { /* iterate over vector components */
                    for (k = 0; k < ncells; k++)
                    { /* iterate over cells */
                        for (l = 0; l < numhexa[k]; l++)
                        {
                            fprintf(f, "%12.5e\n",
                                    (*(fortran_real4 **)ensightvars[i].var)[k * 3 + j]);
                        }
                    }
                }
                break;

            /* For U variables, just save the pointers for when we hit the
           matching W */
            case evt_u:
                lastu = ensightvars[i].var;
                break;

            /* For V variables, just save the pointers for when we hit the
           matching W */
            case evt_v:
                lastv = ensightvars[i].var;
                break;

            /* Vector variables where the three components are different
	   arrays. We can always assume that we have already
	   encountered the matching U and V when we've got the W. */
            case evt_vectorw_nbpatch:
                fprintf(f, "part\n%10d\n", part_patches);
#ifdef GENERATE_NSIDED
                fprintf(f, "nsided\n");
                for (j = 0; j < nbpatch; j++)
                {
                    fprintf(f, "%12.5e\n", (*(fortran_real4 **)lastu)[j]);
                }
                for (j = 0; j < nbpatch; j++)
                {
                    fprintf(f, "%12.5e\n", (*(fortran_real4 **)lastv)[j]);
                }
                for (j = 0; j < nbpatch; j++)
                {
                    fprintf(f, "%12.5e\n", (*(fortran_real4 **)ensightvars[i].var)[j]);
                }
#else
                /* Generate repeated values for quads and trias. */
                fprintf(f, "quad4\n");
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMQUADS(j); k++)
                    {
                        fprintf(f, "%12.5e\n", (*(fortran_real4 **)lastu)[j]);
                    }
                }
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMQUADS(j); k++)
                    {
                        fprintf(f, "%12.5e\n", (*(fortran_real4 **)lastv)[j]);
                    }
                }
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMQUADS(j); k++)
                    {
                        fprintf(f, "%12.5e\n", (*(fortran_real4 **)ensightvars[i].var)[j]);
                    }
                }
                fprintf(f, "tria3\n");
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMTRIAS(j); k++)
                    {
                        fprintf(f, "%12.5e\n", (*(fortran_real4 **)lastu)[j]);
                    }
                }
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMTRIAS(j); k++)
                    {
                        fprintf(f, "%12.5e\n", (*(fortran_real4 **)lastv)[j]);
                    }
                }
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMTRIAS(j); k++)
                    {
                        fprintf(f, "%12.5e\n", (*(fortran_real4 **)ensightvars[i].var)[j]);
                    }
                }
#endif
                break;

            case evt_vectorw_ndrops:
                fprintf(f, "part\n%10d\n", part_droplets);
                fprintf(f, "point\n");
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                    {
                        fprintf(f, "%12.5e\n", (*(fortran_real4 **)lastu)[j]);
                    }
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                    {
                        fprintf(f, "%12.5e\n", (*(fortran_real4 **)lastv)[j]);
                    }
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                    {
                        fprintf(f, "%12.5e\n", (*(fortran_real4 **)ensightvars[i].var)[j]);
                    }
                break;

            case evt_real_ndrops:
                fprintf(f, "part\n%10d\n", part_droplets);
                fprintf(f, "point\n");
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                    {
                        fprintf(f, "%12.5e\n", (*(fortran_real4 **)ensightvars[i].var)[j]);
                    }
                break;

            case evt_int_ndrops:
                fprintf(f, "part\n%10d\n", part_droplets);
                fprintf(f, "point\n");
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                    {
                        fortran_real4 val;

                        val = (*(fortran_integer **)ensightvars[i].var)[j];
                        fprintf(f, "%12.5e\n", val);
                    }
                break;
            } /* matches switch(ensightvars[i].type... */

            /* Close the variable output file. */
            close_file(f);

        } /* matches if(have_var[... */
    } /* matches for(i=0; i<NUMENSIGHTVARS... */
    fprintf(stderr, " done \n");
}

/* ---------------------------------------------------------- */

/* Write out the results to the appropriate files. */
void
write_binary_result_vars()
{
    int count, patch_quads, patch_trias;
    int i, j, k, l;

    int nactive;
    void *lastu = 0, *lastv = 0;
    char filename[80], buffer[80];

    FILE *fCov;
    float *data;

    int totNum;
    int vCnt;
    float *coData;
    float *coVData[3];

    int covCnt;
    char coRTval[256];
    char coRTattr[80];
    char name[80], *s;
    int coAttrSize;
    int coNumAttr;

    covise_message(ENSIGHT_INFO, "Writing binary result variables: ");

    /* Iterate through the possible result variables, checking to see if
       we need to write each one. */
    for (i = 0; i < NUMENSIGHTVARS; i++)
    {

        covCnt = 0;

        /* Replace spaces in the name with underscores, as Ensight
	   doesn't like variable names including spaces. */
        strcpy(name, ensightvars[i].name);
        while ((s = strchr(name, ' ')))
            *s = '_';

        if (have_var[ensightvars[i].ident])
        {

            /* If this variable hasn't yet been encountered, allocate a new
	       data struct for it. */
            if (!ensightvars[i].data)
            {
                ensightvars[i].data = (ensightvardata *)RU_allocMem(sizeof(ensightvardata), "ensightvarsdata");

                /* Assign the variable a new number. */
                ensightvars[i].data->number = numvariables++;
                fprintf(stderr, "+");
            }
            else
            {
                fprintf(stderr, ".");
            }

            /* Open COVISE file for scal data */
            sprintf(filename, "_var%d",
                    ensightvars[i].data->number);
            fCov = fopen(filename, "ab+");

            /* Write out the value of the variable, depending upon its type. */
            switch (ensightvars[i].type)
            {
            /* One fortran_real4 value per cell. */
            case evt_real_ncells:

                /* COVISE scalar data */
                strcpy(buffer, "USTSDT");
                fwrite(buffer, sizeof(char), 6, fCov);

                totNum = totaltetra + totalpyra + totalhexa;

                coData = (float *)RU_allocMem(totNum * sizeof(float), "coData");

                data = (float *)RU_allocMem(totaltetra * sizeof(float), "data");

                /* set scalar data for tetras */
                count = 0;
                for (j = 0; j < ncells; j++)
                {
                    for (k = 0; k < numtetra[j]; k++)
                    {
                        data[count] = (*(fortran_real4 **)ensightvars[i].var)[j];
                        if (polyIdx.tetraIdx[count] >= totNum)
                        {
                            fprintf(stderr, "         TETRA:  wrong value in index translation %d  %d\n", polyIdx.tetraIdx[count], totNum);
                        }
                        if (polyIdx.tetraIdx[count] > -1)
                        {
                            coData[polyIdx.tetraIdx[count]] = data[count];
                            covCnt++;
                        }
                        count++;
                    }
                }

                if (count != totaltetra)
                    covise_message(ENSIGHT_WARNING, "count mismatch tetra4 \n");

                FREE(data);

                data = (float *)RU_allocMem(totalpyra * sizeof(float), "data");
                count = 0;

                for (j = 0; j < ncells; j++)
                {
                    for (k = 0; k < numpyra[j]; k++)
                    {
                        data[count] = (*(fortran_real4 **)ensightvars[i].var)[j];
                        if (polyIdx.pyraIdx[count] >= totNum)
                        {
                            fprintf(stderr, "         PYRA:  wrong value in index translation %d  %d\n", polyIdx.tetraIdx[count], totNum);
                        }
                        if (polyIdx.pyraIdx[count] > -1)
                        {
                            coData[polyIdx.pyraIdx[count]] = data[count];
                            covCnt++;
                        }
                        count++;
                    }
                }

                if (count != totalpyra)
                    covise_message(ENSIGHT_WARNING, "count mismatch pyramid5 \n");

                FREE(data);

                data = (float *)RU_allocMem(totalhexa * sizeof(float), "data");
                count = 0;

                for (j = 0; j < ncells; j++)
                {
                    for (k = 0; k < numhexa[j]; k++)
                    {
                        data[count] = (*(fortran_real4 **)ensightvars[i].var)[j];
                        if (polyIdx.hexaIdx[count] >= totNum)
                        {
                            fprintf(stderr, "         HEXA:  wrong value in index translation %d  %d\n", polyIdx.tetraIdx[count], totNum);
                        }
                        if (polyIdx.hexaIdx[count] > -1)
                        {
                            coData[polyIdx.hexaIdx[count]] = data[count];
                            covCnt++;
                        }
                        count++;
                    }
                }

                if (count != totalhexa)
                    covise_message(ENSIGHT_WARNING, "count mismatch hexa8 \n");

                fwrite(&covCnt, sizeof(int), 1, fCov);
                fwrite(coData, sizeof(float), covCnt, fCov);

                FREE(data);
                FREE(coData);

                break;

            /* One fortran_real4 value per patch. */
            case evt_real_nbpatch:

                strcpy(buffer, "USTSDT");
                fwrite(buffer, sizeof(char), 6, fCov);

                coData = (float *)RU_allocMem(2 * nbpatch * sizeof(float), "coData");

#ifdef GENERATE_NSIDED

                data = (float *)RU_allocMem(nbpatch * sizeof(float), "data");

                for (j = 0; j < nbpatch; j++)
                {
                    data[j] = (*(fortran_real4 **)ensightvars[i].var)[j];
                }

                FREE(data);
#else
                /* Generate repeated values for quads and trias. */

                patch_quads = 0;
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMQUADS(j); k++)
                    {
                        patch_quads++;
                    }
                }

                data = (float *)RU_allocMem(patch_quads * sizeof(float), "data");
                count = 0;

                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMQUADS(j); k++)
                    {
                        data[count] = (*(fortran_real4 **)ensightvars[i].var)[j];
                        coData[patchIdx.quadIdx[count]] = data[count];
                        count++;
                    }
                }

                if (count != patch_quads)
                    covise_message(ENSIGHT_WARNING, "count mismatch quad4 \n");

                FREE(data);

                patch_trias = 0;
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMTRIAS(j); k++)
                    {
                        patch_trias++;
                    }
                }

                data = (float *)RU_allocMem(patch_trias * sizeof(float), "data");
                count = 0;

                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMTRIAS(j); k++)
                    {
                        data[count] = (*(fortran_real4 **)ensightvars[i].var)[j];
                        coData[patchIdx.triIdx[count]] = data[count];
                        count++;
                    }
                }

                if (count != patch_trias)
                    covise_message(ENSIGHT_WARNING, "count mismatch tria3 \n");

                /* everything collected - finish writing COVISE file */
                totNum = patch_trias + patch_quads;
                fwrite(&totNum, sizeof(int), 1, fCov);
                fwrite(coData, sizeof(float), totNum, fCov);

                FREE(coData);
                FREE(data);

#endif
                break;

            /* This is for the particular case where the var is an array
		   with values {X1, Y1, Z1, X2, Y2, Z2... Xn, Yn, Zn} where n
		   is the number of cells.  Unfortunately, Ensight expects {
		   X1 ... Xn, Y1 ... Yn, Z1 ... Zn }. */
            case evt_vector_ncells2d:

                covCnt = 0;
                /* COVISE vector data */
                strcpy(buffer, "USTVDT");
                fwrite(buffer, sizeof(char), 6, fCov);

                totNum = totaltetra + totalpyra + totalhexa;

                for (j = 0; j < 3; ++j)
                {
                    coVData[j] = (float *)RU_allocMem(totNum * sizeof(float), "coVData");
                }
                vCnt = 0;

                data = (float *)RU_allocMem(3 * totaltetra * sizeof(float), "data");

                /* set vector data for TETRAS */
                count = 0;
                for (j = 0; j < 3; j++)
                { /* iterate over vector components */
                    for (k = 0; k < ncells; k++)
                    { /* iterate over cells */
                        for (l = 0; l < numtetra[k]; l++)
                        {
                            data[count] = (*(fortran_real4 **)ensightvars[i].var)[k * 3 + j];
                            if (polyIdx.tetraIdx[vCnt] > -1)
                            {
                                coVData[j][polyIdx.tetraIdx[vCnt]] = data[count];
                                if (j == 0)
                                    covCnt++;
                            }
                            count++;
                            vCnt++;
                        }
                    }
                    vCnt = 0;
                }

                if (count != 3 * totaltetra)
                    covise_message(ENSIGHT_WARNING, "count mismatch tetra4\n");

                FREE(data);

                data = (float *)RU_allocMem(3 * totalpyra * sizeof(float), "data");

                /* set vector data for pyras */
                count = 0;
                vCnt = 0;
                for (j = 0; j < 3; j++)
                { /* iterate over vector components */
                    for (k = 0; k < ncells; k++)
                    { /* iterate over cells */
                        for (l = 0; l < numpyra[k]; l++)
                        {
                            data[count] = (*(fortran_real4 **)ensightvars[i].var)[k * 3 + j];
                            if (polyIdx.pyraIdx[vCnt] > -1)
                            {
                                coVData[j][polyIdx.pyraIdx[vCnt]] = data[count];
                                if (j == 0)
                                    covCnt++;
                            }
                            count++;
                            vCnt++;
                        }
                    }
                    vCnt = 0;
                }

                if (count != 3 * totalpyra)
                    covise_message(ENSIGHT_WARNING, "count mismatch pyramid5\n");

                FREE(data);

                data = (float *)RU_allocMem(3 * totalhexa * sizeof(float), "data");

                /* set vector data for HEXAS */
                count = 0;
                for (j = 0; j < 3; j++)
                { /* iterate over vector components */
                    for (k = 0; k < ncells; k++)
                    { /* iterate over cells */
                        for (l = 0; l < numhexa[k]; l++)
                        {
                            data[count] = (*(fortran_real4 **)ensightvars[i].var)[k * 3 + j];
                            if (polyIdx.hexaIdx[vCnt] > -1)
                            {
                                coVData[j][polyIdx.hexaIdx[vCnt]] = data[count];
                                if (j == 0)
                                    covCnt++;
                            }
                            count++;
                            vCnt++;
                        }
                    }
                    vCnt = 0;
                }

                if (count != 3 * totalhexa)
                    covise_message(ENSIGHT_WARNING, "count mismatch hexa8\n");

                fwrite(&covCnt, sizeof(int), 1, fCov);

                for (j = 0; j < 3; j++)
                {
                    fwrite(coVData[j], sizeof(float), covCnt, fCov);
                }

                for (j = 0; j < 3; j++)
                {
                    FREE(coVData[j]);
                }

                FREE(data);

                break;

            /* For U variables, just save the pointers for when we hit the
		   matching W */
            case evt_u:
                lastu = ensightvars[i].var;
                break;

            /* For V variables, just save the pointers for when we hit the
		   matching W */
            case evt_v:
                lastv = ensightvars[i].var;
                break;

            /* Vector variables where the three components are different
		   arrays. We can always assume that we have already
		   encountered the matching U and V when we've got the W. */
            case evt_vectorw_nbpatch:

                /* COVISE vector data */
                strcpy(buffer, "USTVDT");
                fwrite(buffer, sizeof(char), 6, fCov);

                for (j = 0; j < 3; ++j)
                {
                    coVData[j] = (float *)RU_allocMem(2 * nbpatch * sizeof(float), "coVData");
                }
                vCnt = 0;

#ifdef GENERATE_NSIDED
                fprintf(f, "nsided\n");
                for (j = 0; j < nbpatch; j++)
                {
                    fprintf(f, "%12.5e\n", (*(fortran_real4 **)lastu)[j]);
                }
                for (j = 0; j < nbpatch; j++)
                {
                    fprintf(f, "%12.5e\n", (*(fortran_real4 **)lastv)[j]);
                }
                for (j = 0; j < nbpatch; j++)
                {
                    fprintf(f, "%12.5e\n", (*(fortran_real4 **)ensightvars[i].var)[j]);
                }
#else
                /* Generate repeated values for quads and trias. */

                patch_quads = 0;
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMQUADS(j); k++)
                    {
                        patch_quads++;
                    }
                }

                data = (float *)RU_allocMem(patch_quads * sizeof(float), "data");

                count = 0;
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMQUADS(j); k++)
                    {
                        data[count] = (*(fortran_real4 **)lastu)[j];
                        coVData[0][patchIdx.quadIdx[count]] = data[count];
                        count++;
                    }
                }

                if (count != patch_quads)
                    covise_message(ENSIGHT_WARNING, "count mismatch quad4:lastu\n");

                count = 0;
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMQUADS(j); k++)
                    {
                        data[count] = (*(fortran_real4 **)lastv)[j];
                        coVData[1][patchIdx.quadIdx[count]] = data[count];
                        count++;
                    }
                }

                if (count != patch_quads)
                    covise_message(ENSIGHT_WARNING, "count mismatch quad4:lastv\n");

                count = 0;
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMQUADS(j); k++)
                    {
                        data[count] = (*(fortran_real4 **)ensightvars[i].var)[j];
                        coVData[2][patchIdx.quadIdx[count]] = data[count];
                        count++;
                    }
                }

                if (count != patch_quads)
                    covise_message(ENSIGHT_WARNING, "count mismatch quad4:var\n");

                FREE(data);

                patch_trias = 0;
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMTRIAS(j); k++)
                    {
                        patch_trias++;
                    }
                }

                data = (float *)RU_allocMem(patch_trias * sizeof(float), "data");

                count = 0;
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMTRIAS(j); k++)
                    {
                        data[count] = (*(fortran_real4 **)lastu)[j];
                        coVData[0][patchIdx.triIdx[count]] = data[count];
                        count++;
                    }
                }

                if (count != patch_trias)
                    covise_message(ENSIGHT_WARNING, "count mismatch tria3:lastu\n");

                count = 0;
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMTRIAS(j); k++)
                    {
                        data[count] = (*(fortran_real4 **)lastv)[j];
                        coVData[1][patchIdx.triIdx[count]] = data[count];
                        count++;
                    }
                }

                if (count != patch_trias)
                    covise_message(ENSIGHT_WARNING, "count mismatch tria3:lastv\n");

                count = 0;
                for (j = 0; j < nbpatch; j++)
                {
                    for (k = 0; k < NUMTRIAS(j); k++)
                    {
                        data[count] = (*(fortran_real4 **)ensightvars[i].var)[j];
                        coVData[2][patchIdx.triIdx[count]] = data[count];
                        count++;
                    }
                }

                if (count != patch_trias)
                    covise_message(ENSIGHT_WARNING, "count mismatch tria3:var\n");

                /* everything collected - finish writing COVISE file */
                totNum = patch_trias + patch_quads;
                fwrite(&totNum, sizeof(int), 1, fCov);

                for (j = 0; j < 3; j++)
                {
                    fwrite(coVData[j], sizeof(float), totNum, fCov);
                }

                for (j = 0; j < 3; j++)
                {
                    FREE(coVData[j]);
                }

                FREE(data);

#endif
                break;

            case evt_vectorw_ndrops:

                nactive = 0;
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                        nactive++;

                /* COVISE vector data */
                strcpy(buffer, "USTVDT");
                fwrite(buffer, sizeof(char), 6, fCov);
                fwrite(&nactive, sizeof(int), 1, fCov);

                data = (float *)RU_allocMem(nactive * sizeof(float), "data");

                count = 0;
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                    {
                        data[count] = (*(fortran_real4 **)lastu)[j];
                        count++;
                    }

                if (count != nactive)
                    covise_message(ENSIGHT_WARNING, "count mismatch drops:lastu\n");

                fwrite(data, sizeof(float), nactive, fCov);

                count = 0;
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                    {
                        data[count] = (*(fortran_real4 **)lastv)[j];
                        count++;
                    }

                if (count != nactive)
                    covise_message(ENSIGHT_WARNING, "count mismatch drops:lastv\n");

                fwrite(data, sizeof(float), nactive, fCov);

                count = 0;
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                    {
                        data[count] = (*(fortran_real4 **)ensightvars[i].var)[j];
                        count++;
                    }

                if (count != nactive)
                    covise_message(ENSIGHT_WARNING, "count mismatch drops:var\n");

                fwrite(data, sizeof(float), nactive, fCov);

                FREE(data);

                break;

            case evt_real_ndrops:

                nactive = 0;
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                        nactive++;

                /* COVISE scalar data */
                strcpy(buffer, "USTSDT");
                fwrite(buffer, sizeof(char), 6, fCov);
                fwrite(&nactive, sizeof(int), 1, fCov);

                data = (float *)RU_allocMem(nactive * sizeof(float), "data");

                count = 0;
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                    {
                        data[count] = (*(fortran_real4 **)ensightvars[i].var)[j];
                        count++;
                    }

                if (count != nactive)
                    covise_message(ENSIGHT_WARNING, "count mismatch drops,real:var\n");

                fwrite(data, sizeof(float), nactive, fCov);

                FREE(data);

                break;

            case evt_int_ndrops:

                nactive = 0;
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                        nactive++;

                /* COVISE scalar data */
                strcpy(buffer, "USTSDT");
                fwrite(buffer, sizeof(char), 6, fCov);
                fwrite(&nactive, sizeof(int), 1, fCov);

                data = (float *)RU_allocMem(nactive * sizeof(float), "data");

                count = 0;
                for (j = 0; j < ndrops; j++)
                    if (!dstat[j])
                    {
                        data[count] = (*(fortran_integer **)ensightvars[i].var)[j];
                        count++;
                    }

                if (count != nactive)
                    covise_message(ENSIGHT_WARNING, "count mismatch drops,int:var\n");

                fwrite(data, sizeof(float), nactive, fCov);

                FREE(data);

                break;

            } /* matches switch(ensightvars[i].type... */

            /* add SPECIES and REALTIME attribute */
            strcpy(buffer, "SPECIES");
            strcpy(coRTattr, "REALTIME");
            sprintf(coRTval, "%30.15f", steptime);

            coAttrSize = sizeof(int) + 2 + strlen(buffer) + strlen(name);
            coAttrSize += 2 + strlen(coRTattr) + strlen(coRTval);

            coNumAttr = 2;

            fwrite(&coAttrSize, sizeof(int), 1, fCov);
            fwrite(&coNumAttr, sizeof(int), 1, fCov);
            fwrite(buffer, sizeof(char), strlen(buffer) + 1, fCov);
            fwrite(name, sizeof(char), strlen(name) + 1, fCov);
            fwrite(coRTattr, sizeof(char), strlen(coRTattr) + 1, fCov);
            fwrite(coRTval, sizeof(char), strlen(coRTval) + 1, fCov);

            /* Close the variable output file. */
            fclose(fCov);
        } /* matches if(have_var[... */
    } /* matches for(i=0; i<NUMENSIGHTVARS... */

    fprintf(stderr, " done \n");
}

/* ---------------------------------------------------------- */

/* Write out a geometry file. */
void
write_geometry_file(FILE *f)
{
    FILE *fe;

    /* Two lines of description, up to 80 characters each. If it is
     possible to obtain a name for the model from Vectis, it would
     make more sense to print that here. */
    fprintf(f, "v2e output\n");
    fprintf(f, "Data converted from Vectis Phase 5 POST file\n");

    fprintf(f, "node id assign\n");
    fprintf(f, "element id given\n");

    fprintf(f, "extents\n");
    fprintf(f, "%12.5e%12.5e\n%12.5e%12.5e\n%12.5e%12.5e\n",
            xmin, xmax, ymin, ymax, zmin, zmax);

    /* Write out the patches */
    write_patches(f);

    /* Write out the scalar cells based on the scalar cell
     connectivity data. */
    fe = open_file("_cellconn", "r");
    write_scalar_cells(f, fe);
    close_file(fe);

    /* Write out the droplets. */
    if (have_var[v_xdrop])
        write_droplets(f);
}

/* ---------------------------------------------------------- */

/* Write out a binary geometry file. */
void
write_binary_geometry_file()
{

    /* Write out the patches */
    write_binary_patches();

    /* Write out the scalar cells based on the scalar cell
     connectivity data. */

    write_binary_scalar_cells();

    /* Write out the droplets. */
    if (have_var[v_xdrop])
        write_binary_droplets();

    /* Write out boundaries */
    if (write_boundaries)
        write_binary_boundaries();
}

/* ---------------------------------------------------------- */

/* Write out "general" data. This should be called before the pts
   array is modified for changing geometry. */
void
write_general_data()
{

    calculate_scalar_cell_connectivity();
}

/* ---------------------------------------------------------- */

/* Write a result set out. */
void
write_result_set()
{
    resultset *r;

    char buffer[80];
    char coRTval[256];
    int coAttrSize, coNumAttr;

    /* Store the details of the result set in the list. */
    r = (resultset *)RU_allocMem(sizeof(resultset), "resultset");
    if (!first_result)
    {
        /* No result sets have been allocated yet. */
        first_result = last_result = r;
    }
    else
    {
        /* Add to the end of the list. */
        last_result->_next = r;
        last_result = r;
    }
    r->_next = NULL;

    /* Save the time. */
    r->time = steptime;

    /* Check to see if changing geometry is in use or if this is the
       first result set. */
    if (have_var[v_pts] || (first_result == r) || any_new_geometry)
    {

        /* binary files */
        if (write_binary)
        {

            write_binary_geometry_file();

            writeUNSGRD(covGeoF, &CoUNSGRD);

            /* add REALTIME attribute */
            strcpy(buffer, "REALTIME");
            sprintf(coRTval, "%30.15f", steptime);

            coAttrSize = sizeof(int) + 2 + strlen(buffer) + strlen(coRTval);
            coNumAttr = 1;

            fwrite(&coAttrSize, sizeof(int), 1, covGeoF);
            fwrite(&coNumAttr, sizeof(int), 1, covGeoF);
            fwrite(buffer, sizeof(char), strlen(buffer) + 1, covGeoF);
            fwrite(coRTval, sizeof(char), strlen(coRTval) + 1, covGeoF);
        }

        new_geometry = FALSE;
    }

    /* Write out the result vars */
    if (write_ascii)
        write_result_vars();

    if (write_binary)
        write_binary_result_vars();

    /* Increment the counter */
    ++numresultsets;
}

/* ---------------------------------------------------------- */

/* Write out a case file and other appropriate information. */
void
write_case()
{
    FILE *f;
    int i;
    resultset *r;

    /* Remove the cell connectivity file. */
    unlink_ensight_file("_cellconn");

    /* Write out the case file. */
    f = open_ensight_file("out.case", "w");

    fprintf(f, "FORMAT\n");
    fprintf(f, "type: ensight gold\n");
    fprintf(f, "\n");

    /* Write out the geometry file definition; this depends on whether
     changing geometry is in use or not. */
    fprintf(f, "GEOMETRY\n");
    if (have_var[v_pts] || any_new_geometry)
    {
        fprintf(f, "model: out.geo.****\n");
    }
    else
    {
        fprintf(f, "model: out.geo.0000\n");
    }
    fprintf(f, "\n");

    fprintf(f, "VARIABLE\n");

    /* Iterate through the possible variables, writing out description
     lines as appropriate for the ones that were actually found. */
    for (i = 0; i < NUMENSIGHTVARS; i++)
    {
        if (ensightvars[i].data)
        {
            char name[80], *s;

            /* Replace spaces in the name with underscores, as Ensight
	 doesn't like variable names including spaces. */
            strcpy(name, ensightvars[i].name);
            while ((s = strchr(name, ' ')))
                *s = '_';

            switch (ensightvars[i].type)
            {
            case evt_real_ncells:
            case evt_real_nbpatch:
            case evt_real_ndrops:
            case evt_int_ndrops:
                fprintf(f, "scalar per element: %10d %s var%d.****\n",
                        1, name, ensightvars[i].data->number);
                break;
            case evt_vector_ncells2d:
            case evt_vectorw_nbpatch:
            case evt_vectorw_ndrops:
                fprintf(f, "vector per element: %10d %s var%d.****\n",
                        1, name, ensightvars[i].data->number);
                break;
            }
        }
    } /* matches for(i=0; i<NUMENSIGHTVARS... */

    fprintf(f, "\n");

    fprintf(f, "TIME\n");
    fprintf(f, "time set: %10d\n", 1);

    fprintf(f, "number of steps: %10d\n", numresultsets);
    fprintf(f, "filename start number: %10d\n", 0);
    fprintf(f, "filename increment: %10d\n", 1);

    /* Write out the result set times. */
    fprintf(f, "time values:");
    i = 0;
    for (r = first_result; r; r = r->_next)
    {
        fprintf(f, " %12.5e", r->time);
        /* Ensight expects to have at least two items on each line that
       isn't the last, but won't accept lines longer than 79
       characters. This performs simple word wrap to keep it happy. */
        if (i++ == 3)
        {
            fprintf(f, "\n");
            i = 0;
        }
    }
    fprintf(f, "\n\n");

    close_file(f);
}

/* ------------------- Vectis reading functions ----------------- */

#ifdef STANDALONE_CONVERTOR

/* The input file. */
FILE *input_file;

#ifdef INPUT_PROGRESS_METER
/* The length of the input file. */
long input_file_len;
#endif

/* Conditionally enable byteswapping. */
#ifdef NEED_TO_SWAP
#define SWAPFUNC RU_changeSex
#else
/* If we're not using byteswapping, don't bother generating any code */
#define SWAPFUNC(a, b, c, d)
#endif

/* Byteswap various types of data. */
#define SWAPHDR(i) SWAPFUNC(&i, &i, sizeof(fortran_header), 1)
#define SWAPINTS(p, n) SWAPFUNC(p, p, sizeof(fortran_integer), n)
#define SWAPINT(i) SWAPINTS(&i, 1)
#define SWAPREALS(p, n) SWAPFUNC(p, p, sizeof(fortran_real4), n)
#define SWAPREAL(i) SWAPREALS(&i, 1)

/* Read Fortran unformatted record header, optionally checking both
   copies. */
fortran_header
read_record_header()
{
    fortran_header h1, h2;
#if defined(INPUT_PROGRESS_METER) || defined(CHECK_RECORD_ENDS)
    long pos;
#endif

    /* Read the start-of-record header */
    fread(&h1, sizeof(fortran_header), 1, input_file);
    SWAPHDR(h1);
    if (feof(input_file))
        return 0; /* so that if the end of the file is hit,
				     we don't lose the EOF flag by seeking */

#if defined(INPUT_PROGRESS_METER) || defined(CHECK_RECORD_ENDS)
    pos = ftell(input_file);
#endif

#ifdef INPUT_PROGRESS_METER
    /* Show the input progress meter */
    fprintf(stderr, "%3.0f%%%c%c%c%c", (100.0 * pos) / input_file_len, 8, 8, 8, 8);
    fflush(stderr);
#endif

#ifdef CHECK_RECORD_ENDS
    /* Attempt to read the end-of-record header */
    fseek(input_file, h1, SEEK_CUR);
    fread(&h2, sizeof(fortran_header), 1, input_file);
    SWAPHDR(h2);
    fseek(input_file, pos, SEEK_SET);

    /* Check the two values for a match, and complain if they don't */
    if (h1 != h2)
    {
        sprintf(message, "Fortran binary headers do not match at "
                         "position %ld\n",
                pos - sizeof(fortran_header));
        covise_message(ENSIGHT_WARNING, message);
    }
#endif

    return h1;
}

/* Begin a Fortran record. */
long newpos, temppos;
#define BEGINRECORD                         \
    temppos = ftell(input_file);            \
    newpos = temppos + read_record_header() \
             + 2 * sizeof(fortran_header)

/* End a Fortran record, optionally checking that all the data was read. */
#ifdef CHECK_RECORD_ENDS
#define ENDRECORD                                                                                                                                                         \
    if (sizeof(fortran_header) + ftell(input_file) != newpos)                                                                                                             \
    {                                                                                                                                                                     \
        sprintf(message, "Position mismatch while reading " "from Fortran binary file (is %ld, " "should be %ld)\n", ftell(input_file), newpos - sizeof(fortran_header)); \
        covise_message(ENSIGHT_FATAL_ERROR, message);                                                                                                                     \
    }                                                                                                                                                                     \
    fseek(input_file, newpos, SEEK_SET)
#else
#define ENDRECORD fseek(input_file, newpos, SEEK_SET)
#endif

/* Skip the remaining contents in a Fortran record. */
#define SKIPCONTENT fseek(input_file, newpos - sizeof(fortran_header), SEEK_SET)

/* Read various types of data from a Fortran record. */
#define READINT(i)                                     \
    fread(&i, sizeof(fortran_integer), 1, input_file); \
    SWAPINT(i)
#define READINTS(i, j)                                \
    fread(i, sizeof(fortran_integer), j, input_file); \
    SWAPINTS(i, j)
#define READREAL(i)                                  \
    fread(&i, sizeof(fortran_real4), 1, input_file); \
    SWAPREAL(i)
#define READREALS(i, j)                             \
    fread(i, sizeof(fortran_real4), j, input_file); \
    SWAPREALS(i, j)
#define READ2DREALS(i, j, k)                                \
    fread(i, sizeof(fortran_real4), (j) * (k), input_file); \
    SWAPREALS(i, (j) * (k))

/* Read a character string, stripping trailing spaces. */
fortran_character *t_ch;
#define READCHARS(i, j)                                      \
    fread(i, sizeof(fortran_character), j, input_file);      \
    for (t_ch = i + j - 1; t_ch > i && *t_ch == ' '; --t_ch) \
        *t_ch = '\0';

/* Allocate arrays of various types. */
#define ALLOCINTS(i, j) i = (fortran_integer_array)RU_allocMem((j)                        \
                                                               * sizeof(fortran_integer), \
                                                               "allocints")
#define ALLOCREALS(i, j) i = (fortran_real4_array)RU_allocMem((j)                      \
                                                              * sizeof(fortran_real4), \
                                                              "allocreals")
#define ALLOC2DREALS(i, j, k) i = (fortran_real4_2darray)RU_allocMem((j)                            \
                                                                     * (k) * sizeof(fortran_real4), \
                                                                     "alloc2dreals")

/* Allocate and read an array of data contained within a record. */
#define RECORDSKIP \
    BEGINRECORD;   \
    SKIPCONTENT;   \
    ENDRECORD
#define RECORDINT(i) \
    BEGINRECORD;     \
    READINT(i);      \
    ENDRECORD
#define RECORDINTS(i, j) \
    BEGINRECORD;         \
    ALLOCINTS(i, j);     \
    READINTS(i, j);      \
    ENDRECORD
#define RECORDREAL(i) \
    BEGINRECORD;      \
    READREAL(i);      \
    ENDRECORD
#define RECORDREALS(i, j) \
    BEGINRECORD;          \
    ALLOCREALS(i, j);     \
    READREALS(i, j);      \
    ENDRECORD
#define RECORD2DREALS(i, j, k) \
    BEGINRECORD;               \
    ALLOC2DREALS(i, j, k);     \
    READ2DREALS(i, j, k);      \
    ENDRECORD

/* Slurp the general data from the POST file. */
void
slurp_general()
{
    BOOL done = FALSE,
         new_geometry_block1 = FALSE;
    fortran_integer ident;
    int i;

    covise_message(ENSIGHT_INFO, "Reading general data\n");

    if (new_geometry)
        new_geometry_block1 = TRUE;

    while (!done)
    {
        /* Read the block ident. */
        if (new_geometry_block1)
        {
            ident = 1;
            new_geometry_block1 = FALSE;
        }
        else
        {
            BEGINRECORD;
            READINT(ident);
            ENDRECORD;
        }

        switch (ident)
        {
        case 600:
            /* If it's a 600 block, then we've evidently missed something;
         the loop should have stopped by now. */
            covise_message(ENSIGHT_FATAL_ERROR, "encountered premature 600 block\n");

        case 1:
            covise_message(ENSIGHT_INFO, "... grid specifications\n");

            BEGINRECORD;
            READREAL(xmin);
            READREAL(xmax);
            READREAL(ymin);
            READREAL(ymax);
            READREAL(zmin);
            READREAL(zmax);
            ENDRECORD;

            BEGINRECORD;
            READINT(ncells);
            READINT(nts);
            READINT(nbwss);
            READINT(nbess);
            READINT(nbsss);
            READINT(nbnss);
            READINT(nblss);
            READINT(nbhss);
            ENDRECORD;

            BEGINRECORD;
            READINT(icube);
            READINT(jcube);
            READINT(kcube);
            READINT(ni);
            READINT(nj);
            READINT(nk);
            ENDRECORD;

            ALLOCREALS(xndim, ni + 1);
            for (i = 0; i < ni + 1; i++)
            {
                RECORDREAL(xndim[i]);
            }
            ALLOCREALS(yndim, nj + 1);
            for (i = 0; i < nj + 1; i++)
            {
                RECORDREAL(yndim[i]);
            }
            ALLOCREALS(zndim, nk + 1);
            for (i = 0; i < nk + 1; i++)
            {
                RECORDREAL(zndim[i]);
            }

            break;

        case 24:
            covise_message(ENSIGHT_INFO, "... scalar cell positions\n");

            RECORDINTS(iglobe, nts);
            RECORDINTS(jglobe, nts);
            RECORDINTS(kglobe, nts);

            RECORDINTS(ilpack, nts);
            RECORDINTS(itpack, nts);

            ALLOCINTS(ils, nts);
            ALLOCINTS(ile, nts);
            ALLOCINTS(jls, nts);
            ALLOCINTS(jle, nts);
            ALLOCINTS(kls, nts);
            ALLOCINTS(kle, nts);

            ALLOCINTS(itypew, nts);
            ALLOCINTS(itypee, nts);
            ALLOCINTS(itypes, nts);
            ALLOCINTS(itypen, nts);
            ALLOCINTS(itypel, nts);
            ALLOCINTS(itypeh, nts);

            for (i = 0; i < nts; i++)
            {
                ils[i] = (ilpack[i]) & 31;
                ile[i] = (ilpack[i] >> 5) & 31;
                jls[i] = (ilpack[i] >> 10) & 31;
                jle[i] = (ilpack[i] >> 15) & 31;
                kls[i] = (ilpack[i] >> 20) & 31;
                kle[i] = (ilpack[i] >> 25) & 31;

                itypew[i] = (itpack[i]) & 31;
                itypee[i] = (itpack[i] >> 5) & 31;
                itypes[i] = (itpack[i] >> 10) & 31;
                itypen[i] = (itpack[i] >> 15) & 31;
                itypel[i] = (itpack[i] >> 20) & 31;
                itypeh[i] = (itpack[i] >> 25) & 31;
            }

            RECORDREALS(voln, nts);
            break;

        case 25:
            covise_message(ENSIGHT_INFO, "... scalar cell positions\n");

            RECORDINTS(iglobe, nts);
            RECORDINTS(jglobe, nts);
            RECORDINTS(kglobe, nts);

            RECORDINTS(ils, nts);
            RECORDINTS(ile, nts);
            RECORDINTS(jls, nts);
            RECORDINTS(jle, nts);
            RECORDINTS(kls, nts);
            RECORDINTS(kle, nts);

            RECORDINTS(itpack, nts);

            ALLOCINTS(itypew, nts);
            ALLOCINTS(itypee, nts);
            ALLOCINTS(itypes, nts);
            ALLOCINTS(itypen, nts);
            ALLOCINTS(itypel, nts);
            ALLOCINTS(itypeh, nts);

            for (i = 0; i < nts; i++)
            {
                itypew[i] = (itpack[i]) & 31;
                itypee[i] = (itpack[i] >> 5) & 31;
                itypes[i] = (itpack[i] >> 10) & 31;
                itypen[i] = (itpack[i] >> 15) & 31;
                itypel[i] = (itpack[i] >> 20) & 31;
                itypeh[i] = (itpack[i] >> 25) & 31;
            }

            RECORDREALS(voln, nts);
            break;

        case 8:
            covise_message(ENSIGHT_INFO, "... scalar cell faces\n");

            BEGINRECORD;
            READINT(ncellu);
            READINT(ntu);
            READINT(ncellv);
            READINT(ntv);
            READINT(ncellw);
            READINT(ntw);
            ENDRECORD;

            BEGINRECORD;
            READREAL(iafactor);
            READREAL(jafactor);
            READREAL(kafactor);
            ENDRECORD;

            RECORDREALS(areau, ntu);
            RECORDINTS(lwus, ntu);
            RECORDINTS(leus, ntu);
            RECORDREALS(areav, ntv);
            RECORDINTS(lsvs, ntv);
            RECORDINTS(lnvs, ntv);
            RECORDREALS(areaw, ntw);
            RECORDINTS(llws, ntw);
            RECORDINTS(lhws, ntw);
            RECORDINT(nfpadr);
            RECORDINTS(nfpol, ntu + ntv + ntw);
            RECORDINTS(lbfpol, ntu + ntv + ntw);
            RECORDINTS(lfpol, nfpadr);

            break;

        case 45:
            covise_message(ENSIGHT_INFO, "... patches\n");

            BEGINRECORD;
            READINT(nbpatch);
            READINT(nbound);
            READINT(nnode);
            READINT(nnodref);
            ENDRECORD;

            RECORDINTS(ncpactual, nbpatch);
            RECORDINTS(mpatch, nbpatch);
            RECORDINTS(ltype, nbound);
            RECORDINTS(nodspp, nbpatch);
            RECORDINTS(lbnod, nbpatch);
            RECORDINTS(nodlist, nnodref);
            RECORD2DREALS(pts, 3, nnode);

            /* The next block will be a 600, so we're done. If we ever get
	 46 block handling, done would need to be set there too. */
            done = TRUE;

            break;

        default:
            sprintf(message, "unknown block %d \n", ident);
            covise_message(ENSIGHT_FATAL_ERROR, message);
            break;
        } /* matches switch (ident)... */
    } /* matches while (!done)... */
}

void
clear_result_vars()
{
    int i;

    /* Allocate memory for result variables. Note that we can't
     allocate memory for droplet arrays at this point, as we
     don't get a value for ndrops until we've read the 600 block. */

    ALLOC2DREALS(velcent, 3, ncells);

    /* clear the array indicating which variables we have. */
    for (i = 0; i < num_varidents; i++)
        have_var[i] = 0;
}

/* ---------------------------------------------------------- */

/* Allocate result variables for droplets. This should not be called
   until ndrops has been read. */
void
allocate_droplet_vars()
{

    if (xdrop)
        free_droplet_vars();

    /* Allocate the droplet variables. */
    ALLOCREALS(xdrop, ndrops);
    ALLOCREALS(ydrop, ndrops);
    ALLOCREALS(zdrop, ndrops);
    ALLOCREALS(udrop, ndrops);
    ALLOCREALS(vdrop, ndrops);
    ALLOCREALS(wdrop, ndrops);
    ALLOCREALS(dendr, ndrops);
    ALLOCREALS(tdrop, ndrops);
    ALLOCREALS(ddrop, ndrops);
    ALLOCINTS(count, ndrops);
    ALLOCINTS(dstat, ndrops);
    ALLOCINTS(dhol, ndrops);
    ALLOCINTS(ncdrop, ndrops);
}

/* ---------------------------------------------------------- */

/* Allocate result variables for droplets. This should not be called
   until ndrops has been read. */
void
free_droplet_vars()
{

    FREE(xdrop);
    FREE(ydrop);
    FREE(zdrop);
    FREE(udrop);
    FREE(vdrop);
    FREE(wdrop);
    FREE(dendr);
    FREE(tdrop);
    FREE(ddrop);
    FREE(count);
    FREE(dstat);
    FREE(dhol);
    FREE(ncdrop);
}

/* Slurp the next result set from the POST file. 
   Returns FALSE for EOF. */
BOOL
slurp_result_set()
{
    fortran_integer ident;
    fortran_character name[81];
    int i;

    /* Read ident and check that it's 600. */
    BEGINRECORD;
    READINT(ident);
    /* Give up if we've hit the end of the file; if we don't check for EOF
     here, then the fseek in the ENDRECORD macro will clear the EOF flag
     and we'll be none the wiser. */
    if (feof(input_file))
        return FALSE;
    ENDRECORD;

    new_geometry = FALSE;

    if (ident == 600)
    {

        covise_message(ENSIGHT_INFO, "Reading result set\n");
    }
    else if (ident == 1)
    { /* new geometry */

        new_geometry = TRUE;
        any_new_geometry = TRUE;

        covise_message(ENSIGHT_INFO, "New geometry\n");
        free_geometry();
        slurp_general();
        free_result_vars();
        clear_result_vars();
        write_general_data();

        /* Read ident and check that it's 600. */
        BEGINRECORD;
        READINT(ident);
        /* Give up if we've hit the end of the file; if we don't check for EOF
       here, then the fseek in the ENDRECORD macro will clear the EOF flag
       and we'll be none the wiser. */
        if (feof(input_file))
            return FALSE;
        ENDRECORD;
    }
    else
    {

        sprintf(message, "unexpected block %d \n", ident);
        covise_message(ENSIGHT_FATAL_ERROR, message);
    }

    /* Iterate through all the variables in the block. */
    for (;;)
    {

        /* Read the variable name. */
        BEGINRECORD;
        READCHARS(name, 80);
        ENDRECORD;

        /* If we've reached the end of the data, then we're done with this
       result set. */
        if (!strcmp(name, "END_DATA"))
            return TRUE;

        BEGINRECORD;

/* Macro to set a value in that array. */
#define HAVEVAR(X) have_var[X] = TRUE

/* Macro to generate an if clause matching a variable name. */
#define MATCH(X) else if (!strcmp(name, X))

        if (0)
        {
        } /* necessary to get the "else" in MATCH() to work */
        MATCH("REFERENCE_PRESSURE")
        {
            READREAL(pref);
            HAVEVAR(v_pref);
        }
        MATCH("TIME")
        {
            READREAL(steptime);
            HAVEVAR(v_steptime);
        }
        MATCH("CRANKANGLE")
        {
            READREAL(cangle);
            HAVEVAR(v_cangle);
        }
        MATCH("NUMBER_OF_DROPS")
        {
            READINT(ndrops);
            HAVEVAR(v_ndrops);

            /* Now that we know the number of drops, we can allocate the
	 droplet arrays. */
            allocate_droplet_vars();
        }

        MATCH("PRESSURE")
        {
            if (!have_var[v_p])
                ALLOCREALS(p, ncells);
            READREALS(p, ncells);
            HAVEVAR(v_p);
        }
        MATCH("DENSITY")
        {
            if (!have_var[v_den])
                ALLOCREALS(den, ncells);
            READREALS(den, ncells);
            HAVEVAR(v_den);
        }
        MATCH("TEMPERATURE")
        {
            if (!have_var[v_t])
                ALLOCREALS(t, ncells);
            READREALS(t, ncells);
            HAVEVAR(v_t);
        }
        MATCH("CELL_SCALAR_FIELD Mach Number")
        {
            if (!have_var[v_mach])
                ALLOCREALS(mach, ncells);
            READREALS(mach, ncells);
            HAVEVAR(v_mach);
        }
        MATCH("PASSIVE_SCALAR")
        {
            if (!have_var[v_ps1])
                ALLOCREALS(ps1, ncells);
            READREALS(ps1, ncells);
            HAVEVAR(v_ps1);
        }
        MATCH("TURBULENCE_ENERGY")
        {
            if (!have_var[v_te])
                ALLOCREALS(te, ncells);
            READREALS(te, ncells);
            HAVEVAR(v_te);
        }
        MATCH("TURBULENCE_DISSIPATION")
        {
            if (!have_var[v_ed])
                ALLOCREALS(ed, ncells);
            READREALS(ed, ncells);
            HAVEVAR(v_ed);
        }

        MATCH("U_VELOCITY")
        {
            for (i = 0; i < ncells; i++)
            {
                READINT(velcent[i * 3]);
            }
        }
        MATCH("V_VELOCITY")
        {
            for (i = 0; i < ncells; i++)
            {
                READINT(velcent[1 + (i * 3)]);
            }
        }
        MATCH("W_VELOCITY")
        {
            for (i = 0; i < ncells; i++)
            {
                READINT(velcent[2 + (i * 3)]);
            }
            HAVEVAR(v_velcent);
        }

        MATCH("CELL_SCALAR_FIELD Combustion progress variable")
        {
            if (!have_var[v_combpro])
                ALLOCREALS(combpro, ncells);
            READREALS(combpro, ncells);
            HAVEVAR(v_combpro);
        }
        MATCH("CELL_SCALAR_FIELD Reaction rate (kg/m3s)")
        {
            if (!have_var[v_react_rate])
                ALLOCREALS(react_rate, ncells);
            READREALS(react_rate, ncells);
            HAVEVAR(v_react_rate);
        }
        MATCH("CELL_SCALAR_FIELD Ignition Probability")
        {
            if (!have_var[v_ignprob])
                ALLOCREALS(ignprob, ncells);
            READREALS(ignprob, ncells);
            HAVEVAR(v_ignprob);
        }
        MATCH("CELL_SCALAR_FIELD Unburned Temperature (K)")
        {
            if (!have_var[v_unt])
                ALLOCREALS(unt, ncells);
            READREALS(unt, ncells);
            HAVEVAR(v_unt);
        }
        MATCH("CELL_SCALAR_FIELD Burned Temperature (K)")
        {
            if (!have_var[v_bt])
                ALLOCREALS(bt, ncells);
            READREALS(bt, ncells);
            HAVEVAR(v_bt);
        }
        MATCH("CELL_SCALAR_FIELD NOx mass fraction")
        {
            if (!have_var[v_nox])
                ALLOCREALS(nox, ncells);
            READREALS(nox, ncells);
            HAVEVAR(v_nox);
        }
        MATCH("CELL_SCALAR_FIELD Soot mass fraction")
        {
            if (!have_var[v_sootmf])
                ALLOCREALS(sootmf, ncells);
            READREALS(sootmf, ncells);
            HAVEVAR(v_sootmf);
        }
        MATCH("CELL_SCALAR_FIELD Soot concentration (g/m3)")
        {
            if (!have_var[v_sootcon])
                ALLOCREALS(sootcon, ncells);
            READREALS(sootcon, ncells);
            HAVEVAR(v_sootcon);
        }
        MATCH("CELL_SCALAR_FIELD Radiation Temperature (K)")
        {
            if (!have_var[v_radt])
                ALLOCREALS(radt, ncells);
            READREALS(radt, ncells);
            HAVEVAR(v_radt);
        }
        MATCH("CELL_SCALAR_FIELD Gas Radiation Source (W/m3)")
        {
            if (!have_var[v_gasrads])
                ALLOCREALS(gasrads, ncells);
            READREALS(gasrads, ncells);
            HAVEVAR(v_gasrads);
        }
        MATCH("CELL_SCALAR_FIELD Droplet Radiation Source (W/m3)")
        {
            if (!have_var[v_droprads])
                ALLOCREALS(droprads, ncells);
            READREALS(droprads, ncells);
            HAVEVAR(v_droprads);
        }
        MATCH("CELL_SCALAR_FIELD Absorption Coefficient (1/m)")
        {
            if (!have_var[v_absc])
                ALLOCREALS(absc, ncells);
            READREALS(absc, ncells);
            HAVEVAR(v_absc);
        }
        MATCH("CELL_SCALAR_FIELD Scattering Coefficient (1/m)")
        {
            if (!have_var[v_scats])
                ALLOCREALS(scats, ncells);
            READREALS(scats, ncells);
            HAVEVAR(v_scats);
        }
        MATCH("CELL_SCALAR_FIELD Flowmaster Heat Source (W/m3)")
        {
            if (!have_var[v_fmsource])
                ALLOCREALS(fmsource, ncells);
            READREALS(fmsource, ncells);
            HAVEVAR(v_fmsource);
        }

        MATCH("FACE_U_VELOCITY")
        {
            if (!have_var[v_ua])
                ALLOCREALS(ua, ncellu);
            READREALS(ua, ncellu);
            HAVEVAR(v_ua);
        }
        MATCH("FACE_V_VELOCITY")
        {
            if (!have_var[v_va])
                ALLOCREALS(va, ncellv);
            READREALS(va, ncellv);
            HAVEVAR(v_va);
        }
        MATCH("FACE_W_VELOCITY")
        {
            if (!have_var[v_wa])
                ALLOCREALS(wa, ncellw);
            READREALS(wa, ncellw);
            HAVEVAR(v_wa);
        }
        MATCH("MASS_FRACTION_1")
        {
            if (!have_var[v_amfu])
                ALLOCREALS(amfu, ncells);
            READREALS(amfu, ncells);
            HAVEVAR(v_amfu);
        }
        MATCH("MASS_FRACTION_2")
        {
            if (!have_var[v_amox])
                ALLOCREALS(amox, ncells);
            READREALS(amox, ncells);
            HAVEVAR(v_amox);
        }
        MATCH("MASS_FRACTION_3")
        {
            if (!have_var[v_ampr])
                ALLOCREALS(ampr, ncells);
            READREALS(ampr, ncells);
            HAVEVAR(v_ampr);
        }
        MATCH("MASS_FRACTION_4")
        {
            if (!have_var[v_amin])
                ALLOCREALS(amin, ncells);
            READREALS(amin, ncells);
            HAVEVAR(v_amin);
        }
        MATCH("WAVE_FRACTION_1")
        {
            if (!have_var[v_amfu])
                ALLOCREALS(amfu, ncells);
            READREALS(amfu, ncells);
            HAVEVAR(v_amfu);
        }
        MATCH("WAVE_FRACTION_2")
        {
            if (!have_var[v_amox])
                ALLOCREALS(amox, ncells);
            READREALS(amox, ncells);
            HAVEVAR(v_amox);
        }
        MATCH("WAVE_FRACTION_3")
        {
            if (!have_var[v_ampr])
                ALLOCREALS(ampr, ncells);
            READREALS(ampr, ncells);
            HAVEVAR(v_ampr);
        }
        MATCH("WAVE_FRACTION_4")
        {
            if (!have_var[v_amin])
                ALLOCREALS(amin, ncells);
            READREALS(amin, ncells);
            HAVEVAR(v_amin);
        }
        MATCH("WAVE_FRACTION_5")
        {
            if (!have_var[v_amw5])
                ALLOCREALS(amw5, ncells);
            READREALS(amw5, ncells);
            HAVEVAR(v_amw5);
        }
        MATCH("WAVE_FRACTION_6")
        {
            if (!have_var[v_amw6])
                ALLOCREALS(amw6, ncells);
            READREALS(amw6, ncells);
            HAVEVAR(v_amw6);
        }
        MATCH("WAVE_FRACTION_7")
        {
            if (!have_var[v_amw7])
                ALLOCREALS(amw7, ncells);
            READREALS(amw7, ncells);
            HAVEVAR(v_amw7);
        }

        MATCH("PATCH_U_VELOCITY")
        {
            if (!have_var[v_uapatch])
                ALLOCREALS(uapatch, nbpatch);
            READREALS(uapatch, nbpatch);
            HAVEVAR(v_uapatch);
        }
        MATCH("PATCH_V_VELOCITY")
        {
            if (!have_var[v_vapatch])
                ALLOCREALS(vapatch, nbpatch);
            READREALS(vapatch, nbpatch);
            HAVEVAR(v_vapatch);
        }
        MATCH("PATCH_W_VELOCITY")
        {
            if (!have_var[v_wapatch])
                ALLOCREALS(wapatch, nbpatch);
            READREALS(wapatch, nbpatch);
            HAVEVAR(v_wapatch);
        }
        MATCH("PATCH_TEMPERATURE")
        {
            if (!have_var[v_tpatch])
                ALLOCREALS(tpatch, nbpatch);
            READREALS(tpatch, nbpatch);
            HAVEVAR(v_tpatch);
        }
        MATCH("PATCH_FLUID_TEMP")
        {
            if (!have_var[v_tflpatch])
                ALLOCREALS(tflpatch, nbpatch);
            READREALS(tflpatch, nbpatch);
            HAVEVAR(v_tflpatch);
        }
        MATCH("PATCH_HTC")
        {
            if (!have_var[v_gpatch])
                ALLOCREALS(gpatch, nbpatch);
            READREALS(gpatch, nbpatch);
            HAVEVAR(v_gpatch);
        }
        MATCH("PATCH_SHEAR")
        {
            if (!have_var[v_taupatch])
                ALLOCREALS(taupatch, nbpatch);
            READREALS(taupatch, nbpatch);
            HAVEVAR(v_taupatch);
        }
        MATCH("PATCH_DISTANCE")
        {
            if (!have_var[v_yppatch])
                ALLOCREALS(yppatch, nbpatch);
            READREALS(yppatch, nbpatch);
            HAVEVAR(v_yppatch);
        }

        MATCH("PATCH_SCALAR_FIELD Film Temperature (K)")
        {
            if (!have_var[v_filmt])
                ALLOCREALS(filmt, nbpatch);
            READREALS(filmt, nbpatch);
            HAVEVAR(v_filmt);
        }
        MATCH("PATCH_SCALAR_FIELD Film Thickness (m)")
        {
            if (!have_var[v_filmthick])
                ALLOCREALS(filmthick, nbpatch);
            READREALS(filmthick, nbpatch);
            HAVEVAR(v_filmthick);
        }
        MATCH("PATCH_SCALAR_FIELD Film Velocity U (m/s)")
        {
            if (!have_var[v_filmu])
                ALLOCREALS(filmu, nbpatch);
            READREALS(filmu, nbpatch);
            HAVEVAR(v_filmu);
        }
        MATCH("PATCH_SCALAR_FIELD Film Velocity V (m/s)")
        {
            if (!have_var[v_filmv])
                ALLOCREALS(filmv, nbpatch);
            READREALS(filmv, nbpatch);
            HAVEVAR(v_filmv);
        }
        MATCH("PATCH_SCALAR_FIELD Film Velocity W (m/s)")
        {
            if (!have_var[v_filmw])
                ALLOCREALS(filmw, nbpatch);
            READREALS(filmw, nbpatch);
            HAVEVAR(v_filmw);
        }
        MATCH("PATCH_SCALAR_FIELD Radiation Heat Flux (W/m2)")
        {
            if (!have_var[v_radheatflux])
                ALLOCREALS(radheatflux, nbpatch);
            READREALS(radheatflux, nbpatch);
            HAVEVAR(v_radheatflux);
        }
        MATCH("PATCH_SCALAR_FIELD Near-wall velocity (m/s)")
        {
            if (!have_var[v_nearwallv])
                ALLOCREALS(nearwallv, nbpatch);
            READREALS(nearwallv, nbpatch);
            HAVEVAR(v_nearwallv);
        }

        MATCH("PATCH_SCALAR_FIELD Linked Percentage (%)")
        {
            if (!have_var[v_lppatch])
                ALLOCREALS(linkper, nbpatch);
            READREALS(linkper, nbpatch);
            HAVEVAR(v_lppatch);
        }
        MATCH("PATCH_SCALAR_FIELD Link cell temperature (K)")
        {
            if (!have_var[v_lctpatch])
                ALLOCREALS(linkcellT, nbpatch);
            READREALS(linkcellT, nbpatch);
            HAVEVAR(v_lctpatch);
        }
        MATCH("PATCH_SCALAR_FIELD Link patch temperature (K)")
        {
            if (!have_var[v_lptpatch])
                ALLOCREALS(linkpatchT, nbpatch);
            READREALS(linkpatchT, nbpatch);
            HAVEVAR(v_lptpatch);
        }
        MATCH("PATCH_SCALAR_FIELD Link heat flux (W/m2)")
        {
            if (!have_var[v_lhpatch])
                ALLOCREALS(linkheat, nbpatch);
            READREALS(linkheat, nbpatch);
            HAVEVAR(v_lhpatch);
        }
        MATCH("PATCH_SCALAR_FIELD Link heat transfer coeff. (W/m2K)")
        {
            if (!have_var[v_lhtcpatch])
                ALLOCREALS(linkhtc, nbpatch);
            READREALS(linkhtc, nbpatch);
            HAVEVAR(v_lhtcpatch);
        }

        /* Ignore [XYZ]_GRIDLINES ([xyz]grid), as they are obsolete */

        MATCH("NODE_COORDINATES")
        {
            READ2DREALS(pts, 3, nnode);
            HAVEVAR(v_pts);
        }

        MATCH("VOLUME")
        {
            READREALS(voln, nts);
            HAVEVAR(v_voln);
        }
        MATCH("FACE_U_AREA")
        {
            READREALS(areau, ntu);
            HAVEVAR(v_areau);
        }
        MATCH("FACE_V_AREA")
        {
            READREALS(areav, ntv);
            HAVEVAR(v_areav);
        }
        MATCH("FACE_W_AREA")
        {
            READREALS(areaw, ntw);
            HAVEVAR(v_areaw);
        }
        MATCH("DROP_X_COORDINATE")
        {
            READREALS(xdrop, ndrops);
            HAVEVAR(v_xdrop);
        }
        MATCH("DROP_Y_COORDINATE")
        {
            READREALS(ydrop, ndrops);
            HAVEVAR(v_ydrop);
        }
        MATCH("DROP_Z_COORDINATE")
        {
            READREALS(zdrop, ndrops);
            HAVEVAR(v_zdrop);
        }
        MATCH("DROP_U_VELOCITY")
        {
            READREALS(udrop, ndrops);
            HAVEVAR(v_udrop);
        }
        MATCH("DROP_V_VELOCITY")
        {
            READREALS(vdrop, ndrops);
            HAVEVAR(v_vdrop);
        }
        MATCH("DROP_W_VELOCITY")
        {
            READREALS(wdrop, ndrops);
            HAVEVAR(v_wdrop);
        }

        MATCH("DROP_DENSITY")
        {
            READREALS(dendr, ndrops);
            HAVEVAR(v_dendr);
        }
        MATCH("DROP_TEMPERATURE")
        {
            READREALS(tdrop, ndrops);
            HAVEVAR(v_tdrop);
        }
        MATCH("DROP_DIAMETER")
        {
            READREALS(ddrop, ndrops);
            HAVEVAR(v_ddrop);
        }
        MATCH("DROP_COUNT")
        {
            READINTS(count, ndrops);
            HAVEVAR(v_count);
        }
        MATCH("DROP_STATUS")
        {
            READINTS(dstat, ndrops);
        }
        MATCH("DROP_HOLE")
        {
            READINTS(dhol, ndrops);
            HAVEVAR(v_dhol);
        }
        MATCH("DROP_CELL_NUMBER")
        {
            READINTS(ncdrop, ndrops);
            HAVEVAR(v_ncdrop);
        }
        else
        {
            sprintf(message, "Unknown variable '%s'; ignoring.\n", name);
            covise_message(ENSIGHT_WARNING, message);
            SKIPCONTENT;
        }

        ENDRECORD;
    }
}

/* ---------------------------------------------------------- */

void
free_geometry()
{
    /* Block 1 */

    FREE(xndim);
    FREE(yndim);
    FREE(zndim);

    /* Block 24/25 */

    FREE(iglobe);
    FREE(jglobe);
    FREE(kglobe);

    FREE(ilpack);
    FREE(itpack);

    FREE(ils);
    FREE(ile);
    FREE(jls);
    FREE(jle);
    FREE(kls);
    FREE(kle);

    FREE(itypew);
    FREE(itypee);
    FREE(itypes);
    FREE(itypen);
    FREE(itypel);
    FREE(itypeh);

    FREE(voln);

    /* Block 8 */

    FREE(areau);
    FREE(lwus);
    FREE(leus);
    FREE(areav);
    FREE(lsvs);
    FREE(lnvs);
    FREE(areaw);
    FREE(llws);
    FREE(lhws);
    FREE(nfpol);
    FREE(lbfpol);
    FREE(lfpol);

    /* Block 45 */

    FREE(ncpactual);
    FREE(mpatch);
    FREE(ltype);
    FREE(nodspp);
    FREE(lbnod);
    FREE(nodlist);
    FREE(pts);
}

/* ---------------------------------------------------------- */

void
free_result_vars()
{

    if (have_var[v_p])
        FREE(p);
    if (have_var[v_den])
        FREE(den);
    if (have_var[v_t])
        FREE(t);
    if (have_var[v_mach])
        FREE(mach);
    if (have_var[v_ps1])
        FREE(ps1);
    if (have_var[v_te])
        FREE(te);
    if (have_var[v_ed])
        FREE(ed);

    FREE(velcent);

    if (have_var[v_combpro])
        FREE(combpro);
    if (have_var[v_react_rate])
        FREE(react_rate);
    if (have_var[v_ignprob])
        FREE(ignprob);
    if (have_var[v_unt])
        FREE(unt);
    if (have_var[v_bt])
        FREE(bt);
    if (have_var[v_nox])
        FREE(nox);
    if (have_var[v_sootmf])
        FREE(sootmf);
    if (have_var[v_sootcon])
        FREE(sootcon);
    if (have_var[v_radt])
        FREE(radt);
    if (have_var[v_gasrads])
        FREE(gasrads);
    if (have_var[v_droprads])
        FREE(droprads);
    if (have_var[v_absc])
        FREE(absc);
    if (have_var[v_scats])
        FREE(scats);
    if (have_var[v_fmsource])
        FREE(fmsource);

    if (have_var[v_ua])
        FREE(ua);
    if (have_var[v_va])
        FREE(va);
    if (have_var[v_wa])
        FREE(wa);
    if (have_var[v_amfu])
        FREE(amfu);
    if (have_var[v_amox])
        FREE(amox);
    if (have_var[v_ampr])
        FREE(ampr);
    if (have_var[v_amin])
        FREE(amin);
    if (have_var[v_amw5])
        FREE(amw5);
    if (have_var[v_amw6])
        FREE(amw6);
    if (have_var[v_amw7])
        FREE(amw7);

    if (have_var[v_uapatch])
        FREE(uapatch);
    if (have_var[v_vapatch])
        FREE(vapatch);
    if (have_var[v_wapatch])
        FREE(wapatch);
    if (have_var[v_tpatch])
        FREE(tpatch);
    if (have_var[v_tflpatch])
        FREE(tflpatch);
    if (have_var[v_gpatch])
        FREE(gpatch);
    if (have_var[v_taupatch])
        FREE(taupatch);
    if (have_var[v_yppatch])
        FREE(yppatch);

    if (have_var[v_filmt])
        FREE(filmt);
    if (have_var[v_filmthick])
        FREE(filmthick);
    if (have_var[v_filmu])
        FREE(filmu);
    if (have_var[v_filmv])
        FREE(filmv);
    if (have_var[v_filmw])
        FREE(filmw);
    if (have_var[v_radheatflux])
        FREE(radheatflux);
    if (have_var[v_nearwallv])
        FREE(nearwallv);

    if (have_var[v_lppatch])
        FREE(linkper);
    if (have_var[v_lctpatch])
        FREE(linkcellT);
    if (have_var[v_lptpatch])
        FREE(linkpatchT);
    if (have_var[v_lhpatch])
        FREE(linkheat);
    if (have_var[v_lhtcpatch])
        FREE(linkhtc);
}

/* ---------------------------------------------------------- */

void
free_result_sets(void)
{
    resultset *current;

    current = first_result;

    while (current)
    {
        resultset *next = current->_next;
        FREE(current);
        current = next;
    }

    first_result = NULL;
}

/* ---------------------------------------------------------- */

void
remove_temp_files(void)
{
    unlink("_quad");
    unlink("_tria");
    unlink("_tetra");
    unlink("_pyra");
    unlink("_hexa");
    unlink("_elements");
    unlink("_cellconn");
}

/* ---------------------------------------------------------- */

void
free_memory(void)
{
    int i;

    /* free memory */
    free_droplet_vars();
    for (i = 0; i < NUMENSIGHTVARS; i++)
    {
        if (ensightvars[i].data)
            FREE(ensightvars[i].data);
    }
    free_node_table();
    FREE(numtetra);
    FREE(numpyra);
    FREE(numhexa);
    FREE(backlink);
    FREE(backlink_index);
    FREE(backlink_count);
    free_geometry();
    free_result_vars();
    free_result_sets();
}

/* ---------------------------------------------------------- */

/* Standalone entry point. */
int
main(int argc, char **argv)
{
    char message[MAXLINE];

    /* process options */

    while ((argc > 1) && (argv[1][0] == '-'))
    {

        switch (argv[1][1])
        {

        case 'b':

            write_boundaries = TRUE;
            break;

        case 'v':

            fprintf(stdout, VERSION);
            fprintf(stdout, "\n");
            return EXIT_SUCCESS;
            break;

        case 'h':

            usage();
            break;

        default:

            sprintf(message, "Bad option %s\n", argv[1]);
            covise_message(ENSIGHT_WARNING, message);
            usage();
        }

        ++argv;
        --argc;
    }

    /* process POST files */

    if (argc == 1)
    {

        usage();
    }
    else
    {

        /* header */

        sprintf(message, "COVISE V5.0 Translator for VECTIS Phase 5 : Version %s\n", VERSION);
        covise_message(ENSIGHT_INFO, message);

        while (argc > 1)
        {

            if (write_ascii)
                covise_message(ENSIGHT_INFO, "writing ASCII output\n");
            if (write_binary)
                covise_message(ENSIGHT_INFO, "writing binary output\n");

            if (translate_file(argv[1]))
                covise_message(ENSIGHT_FATAL_ERROR, "failed to translate_file");
            ++argv;
            --argc;
        }
    }

    return EXIT_SUCCESS;
}

/* ---------------------------------------------------------- */

void
usage(void)
{
    char message[MAXLINE];

    fprintf(stdout, "\n");
    sprintf(message, " COVISE V5.0 Translator for VECTIS Phase 5 : Version %s\n", VERSION);
    fprintf(stdout, message);
    fprintf(stdout, "\n");
    fprintf(stdout, "    Usage: v2e [-a|-b|-v|-h] <POST filename(s)>\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "    Options: \n");
    fprintf(stdout, "             -b  Write separate parts for each boundary (binary mode only)\n");
    fprintf(stdout, "             -v  Print version number to stdout\n");
    fprintf(stdout, "             -h  Print this help information to stdout\n");
    fprintf(stdout, "\n");
    fprintf(stdout, "    Example: v2e *.POST* \n");
    fprintf(stdout, "\n");
}

/* ---------------------------------------------------------- */

int
translate_file(char *post_filename)
{
    FILE *coF, *coFO;
    int i;
    int varCnt;
    int *numpatchs;

    char varName[64];
    char varOutName[64];

    initUNSGRD(&CoUNSGRD);
    initIdxPhd(&polyIdx);
    initIdxPgn(&patchIdx);

    /* Attempt to open the input file. */
    input_file = open_file(post_filename, "rb");

    make_covise_directory(post_filename);

    remove_temp_files();

#ifdef INPUT_PROGRESS_METER
    /* If we're using the input progress meter, then we need to find the
       length of the input file before reading any data from it. */
    fseek(input_file, 0, SEEK_END);
    input_file_len = ftell(input_file);
    fseek(input_file, 0, SEEK_SET);
#endif

    slurp_general();
    clear_result_vars();
    write_general_data();

    covGeoF = open_file("_cov_geom", "wb+");
    if (!covGeoF)
    {
        fprintf(stderr, "  cannot open covise temp obj. file\n");
    }

    while (slurp_result_set())
        write_result_set();

    coF = open_covise_file("cells.covise", "wb");
    if (!coF)
    {
        fprintf(stderr, "  cannot open covise output file\n");
    }

    writeCoFileHeader(coF);
    if (numresultsets == 1)
    {
        append_file(coF, covGeoF);
    }
    else
    {

        writeCoSetHeader(coF, numresultsets);
        append_file(coF, covGeoF);
        writeCoTimeStepsAttr(coF, numresultsets);
    }
    fclose(coF);
    fclose(covGeoF);
    unlink("_cov_geom");

    /* assemble patches */
    coF = fopen("_v2c_patches", "rb+");
    if (!coF)
    {
        fprintf(stderr, "  cannot open covise temp. output file\n");
    }

    coFO = open_covise_file("patches.covise", "wb");
    if (!coFO)
    {
        fprintf(stderr, "  cannot open covise output file \n");
    }

    writeCoFileHeader(coFO);
    if (numresultsets == 1)
    {
        append_file(coFO, coF);
    }
    else
    {
        writeCoSetHeader(coFO, numresultsets);
        append_file(coFO, coF);
        writeCoTimeStepsAttr(coFO, numresultsets);
    }
    fclose(coFO);
    fclose(coF);
    unlink("_v2c_patches");

    /* assemble droplets */
    coF = fopen("_v2c_drops", "rb+");
    if (coF)
    {
        coFO = open_covise_file("droplets.covise", "wb");
        if (!coFO)
        {
            fprintf(stderr, "  cannot open covise output file \n");
        }

        writeCoFileHeader(coFO);
        if (numresultsets == 1)
        {
            append_file(coFO, coF);
        }
        else
        {
            writeCoSetHeader(coFO, numresultsets);
            append_file(coFO, coF);
            writeCoTimeStepsAttr(coFO, numresultsets);
        }
        fclose(coFO);
        fclose(coF);
        unlink("_v2c_drops");
    }

    /* write out boundaries */

    numpatchs = (int *)RU_allocMem(nbound * sizeof(int), "numpatchs");
    for (i = 0; i < nbound; i++)
        numpatchs[i] = 0;

    sprintf(varName, "_v2c_boundaries");
    sprintf(varOutName, "boundaries.covise");

    coF = fopen(varName, "rb+");

    if (coF)
    {
        coFO = open_covise_file(varOutName, "wb");
        if (!coFO)
        {
            fprintf(stderr, "  cannot open covise boundary file %s\n", varOutName);
        }

        writeCoFileHeader(coFO);
        if (numresultsets == 1)
        {
            append_file(coFO, coF);
        }
        else
        {
            writeCoSetHeader(coFO, numresultsets);
            append_file(coFO, coF);
            writeCoTimeStepsAttr(coFO, numresultsets);
        }
        close_file(coF);
        close_file(coFO);
        unlink(varName);
    }
    else
    {
        fclose(coF);
    }

    /*     for (i=0; i<nbound;++i) { */
    /* 	if ( numpatchs[i] > 0 ) {  */
    /* 	    fprintf(stderr,"    .....writing boundary output boundary%d.covise\n",i ); */
    /* 	    sprintf(varName,"_v2c_boundaries%d",i); */
    /* 	    sprintf(varOutName,"boundary%d.covise",i); */

    /* 	    coF = fopen(varName,"rb+"); */

    /* 	    if (coF) { */
    /* 		coFO = open_covise_file(varOutName,"wb"); */
    /* 		if (!coFO) { */
    /* 		    fprintf(stderr,"  cannot open covise boundary file %d\n",varCnt); */
    /* 		} */

    /* 		writeCoFileHeader(coFO); */
    /* 		if (numresultsets == 1) { */
    /* 		    append_file(coFO, coF); */
    /* 		} */
    /* 		else {		 */
    /* 		    writeCoSetHeader( coFO, numresultsets); */
    /* 		    append_file(coFO,coF); */
    /* 		    writeCoTimeStepsAttr( coFO,numresultsets ); */
    /* 		} */
    /* 		close_file(coF); */
    /* 		close_file(coFO); */
    /* 		unlink(varName);  */
    /* 	    } */
    /* 	    else { */
    /* 		fclose(coF); */
    /* 	    } */

    /* 	} */
    /*     }     */

    /* write out final covise result files */
    varCnt = 0;
    for (i = 0; i < NUMENSIGHTVARS; i++)
    {

        char name[80], *s;
        /* Replace spaces in the name with underscores, as Ensight
	   doesn't like variable names including spaces. */
        strcpy(name, ensightvars[i].name);
        while ((s = strchr(name, ' ')))
            *s = '_';

        if (have_var[ensightvars[i].ident])
        {
            fprintf(stderr, "    .....writing variable output %s.covise\n", name);
            sprintf(varName, "_var%d", varCnt);
            sprintf(varOutName, "%s.covise", name);

            coF = fopen(varName, "rb+");
            if (!coF)
            {
                fprintf(stderr, "  cannot open covise tmp var file\n");
            }
            else
            {

                coFO = open_covise_file(varOutName, "wb");
                if (!coFO)
                {
                    fprintf(stderr, "  cannot open covise var file %d\n", varCnt);
                }

                writeCoFileHeader(coFO);
                if (numresultsets == 1)
                {
                    append_file(coFO, coF);
                }
                else
                {
                    writeCoSetHeader(coFO, numresultsets);
                    append_file(coFO, coF);
                    writeCoTimeStepsAttr(coFO, numresultsets);
                }

                close_file(coFO);
            }
            close_file(coF);
            unlink(varName);
            varCnt++;
        }
    }

    /*      write_case();  */

    close_file(input_file);

    free_memory();

    covise_message(ENSIGHT_INFO, "Finished normally\n");

    covise_message(ENSIGHT_INFO, "write COVISE geom. file\n");

    unlink("_cellconn");

    /* There is still memory which has not been freed. As both Unix and
       NT will reclaim malloced memory at program exit, we don't
       explicitly free it to avoid swapping back in any memory which has 
       been swapped out during the run. */
    return 0;
}

#endif /* STANDALONE_CONVERTOR */
