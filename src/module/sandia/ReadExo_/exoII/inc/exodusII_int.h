/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*****************************************************************************
*
* exodusII_int.h - ExodusII header file for internal Exodus call use only
*
* author - Sandia National Laboratories
*          Vic Yarberry    - Added headers and error logging
*
*          
* environment - UNIX
*
* revision history - 
*
*  $Id: exodusII_int.h,v 1.8 1994/09/20 23:33:03 mksmith Exp $
*  $Log: exodusII_int.h,v $
 * Revision 1.8  1994/09/20  23:33:03  mksmith
 * Changes to include files for EXODUSII 2.01 to 2.02
 *
 * Revision 1.8  1994/08/24  15:29:51  laschoo
 * new API version 2.02 which includes Alpha support
 *
 * Revision 1.7  1994/03/30  15:37:43  vryarbe
 * Updated for V2.01
 *
 * Revision 1.6  1993/09/24  21:08:46  vryarbe
 * added definitions for new inquire parameters
 *
 * Revision 1.5  1993/08/27  22:52:53  vryarbe
 * modfied for use with property functions
 *
 * Revision 1.4  1993/08/23  20:44:35  vryarbe
 * Added CONST and some new error msg definitions.
 *
 * Revision 1.3  1993/07/08  21:49:08  vryarbe
 * bug fixes to date
 *
 * Revision 1.2  1993/07/01  22:27:16  vryarbe
 * added header
 *
*
*****************************************************************************/

#ifndef EXODUS_II_INT_HDR
#define EXODUS_II_INT_HDR

#include "netcdf.h"

#ifdef __STDC__
#include <stdlib.h>
#endif

#ifdef __STDC__
#define PROTO_ARGS(proto) proto
#else
#define PROTO_ARGS(proto) ()
#endif

/* these should be defined in ANSI C, and probably C++, but just in case ... */

#ifndef EXIT_SUCCESS
#define EXIT_SUCCESS 0
#endif
#ifndef EXIT_FAILURE
#define EXIT_FAILURE 1
#endif
#ifndef NULL
#define NULL 0
#endif

/* EXODUS II version number */

/* ExodusII file version */
#define EX_VERS 2.02
/* ExodusII access library version */
#define EX_API_VERS 2.02

/*
 * This file contains defined constants that are used internally in the
 * EXODUS II API.
 *
 * The first group of constants refer to netCDF variables, attributes, or 
 * dimensions in which the EXODUS II data are stored.  Using the defined 
 * constants will allow the names of the netCDF entities to be changed easily 
 * in the future if needed.  The first three letters of the constant identify 
 * the netCDF entity as a variable (VAR), dimension (DIM), or attribute (ATT).
 *
 * NOTE: The entity name should not have any blanks in it.  Blanks are
 *       technically legal but some netcdf utilities (ncgen in particular)
 *       fail when they encounter a blank in a name.
 *
 *      DEFINED CONSTANT	ENTITY NAME	DATA STORED IN ENTITY
 */
#define ATT_FILE_TYPE "type" /* obsolete                  */
#define ATT_TITLE "title" /* the database title        */
#define ATT_API_VERSION "api_version" /* the EXODUS II api vers #   */
#define ATT_API_VERSION_BLANK "api version" /* the EXODUS II api vers #   */
/*  used for db version 2.01 */
/*  and earlier              */
#define ATT_VERSION "version" /* the EXODUS II file vers # */
#define ATT_FLT_WORDSIZE "floating_point_word_size"
/* word size of floating     */
/* point numbers in file     */
#define ATT_FLT_WORDSIZE_BLANK "floating point word size"
/* word size of floating     */
/* point numbers in file     */
/* used for db version 2.01  */
/* and earlier               */
#define DIM_NUM_NODES "num_nodes" /* # of nodes                */
#define DIM_NUM_DIM "num_dim" /* # of dimensions; 2- or 3-d*/
#define DIM_NUM_ELEM "num_elem" /* # of elements             */
#define DIM_NUM_EL_BLK "num_el_blk" /* # of element blocks       */
#define VAR_COORD "coord" /* nodal coordinates         */
#define VAR_NAME_COOR "coor_names" /* names of coordinates      */
#define VAR_STAT_EL_BLK "eb_status" /* element block status      */
#define VAR_ID_EL_BLK VAR_EB_PROP(1) /* element block ids props   */
#define ATT_NAME_ELB "elem_type" /* element type names for    */
/*   each element block      */
#define DIM_NUM_EL_IN_BLK(num) ex_catstr("num_el_in_blk", num)
/* # of elements in element  */
/*   block num               */
#define DIM_NUM_NOD_PER_EL(num) ex_catstr("num_nod_per_el", num)
/* # of nodes per element in */
/*   element block num       */
#define DIM_NUM_ATT_IN_BLK(num) ex_catstr("num_att_in_blk", num)
/* # of attributes in element*/
/*   block num               */
#define VAR_CONN(num) ex_catstr("connect", num)
/* element connectivity for  */
/*   element block num       */
#define VAR_ATTRIB(num) ex_catstr("attrib", num)
/* list of attributes for    */
/*   element block num       */
#define VAR_EB_PROP(num) ex_catstr("eb_prop", num)
/* list of the numth property*/
/*   for all element blocks  */
#define ATT_PROP_NAME "name" /* name attached to element  */
/*   block, node set, or side*/
/*   set properties          */
#define VAR_MAP "elem_map" /* element order map         */
#define DIM_NUM_SS "num_side_sets" /* # of side sets            */
#define VAR_SS_STAT "ss_status" /* side set status           */
#define VAR_SS_IDS VAR_SS_PROP(1) /* side set id properties    */
#define DIM_NUM_SIDE_SS(num) ex_catstr("num_side_ss", num)
/* # of sides in side set num*/
#define DIM_NUM_DF_SS(num) ex_catstr("num_df_ss", num)
/* # of distribution factors */
/* in side set num           */
/*#define DIM_NUM_NOD_SS(num)	ex_catstr("num_nod_ss",num) *** obsolete *** */
/* # of nodes in side set num*/
#define VAR_FACT_SS(num) ex_catstr("dist_fact_ss", num)
/* the distribution factors  */
/*   for each node in side   */
/*   set num                 */
#define VAR_ELEM_SS(num) ex_catstr("elem_ss", num)
/* list of elements in side  */
/*   set num                 */
#define VAR_SIDE_SS(num) ex_catstr("side_ss", num)
/* list of sides in side set */
#define VAR_SS_PROP(num) ex_catstr("ss_prop", num)
/* list of the numth property*/
/*   for all side sets       */
#define DIM_NUM_NS "num_node_sets" /* # of node sets            */
#define DIM_NUM_NOD_NS(num) ex_catstr("num_nod_ns", num)
/* # of nodes in node set    */
/*   num                     */
#define DIM_NUM_DF_NS(num) ex_catstr("num_df_ns", num)
/* # of distribution factors */
/* in node set num           */
#define VAR_NS_STAT "ns_status" /* node set status           */
#define VAR_NS_IDS VAR_NS_PROP(1) /* node set id properties    */
#define VAR_NODE_NS(num) ex_catstr("node_ns", num)
/* list of nodes in node set */
/*   num                     */
#define VAR_FACT_NS(num) ex_catstr("dist_fact_ns", num)
/* list of distribution      */
/*   factors in node set num */
#define VAR_NS_PROP(num) ex_catstr("ns_prop", num)
/* list of the numth property*/
/*   for all node sets       */
#define DIM_NUM_QA "num_qa_rec" /* # of QA records           */
#define VAR_QA_TITLE "qa_records" /* QA records                */
#define DIM_NUM_INFO "num_info" /* # of information records  */
#define VAR_INFO "info_records" /* information records       */
#define VAR_HIS_TIME "time_hist" /* obsolete                  */
#define VAR_WHOLE_TIME "time_whole" /* simulation times for whole*/
/*   time steps              */
#define VAR_ELEM_TAB "elem_var_tab" /* element variable truth    */
/*   table                   */
#define DIM_NUM_GLO_VAR "num_glo_var" /* # of global variables     */
#define VAR_NAME_GLO_VAR "name_glo_var" /* names of global variables */
#define VAR_GLO_VAR "vals_glo_var" /* values of global variables*/
#define DIM_NUM_NOD_VAR "num_nod_var" /* # of nodal variables      */
#define VAR_NAME_NOD_VAR "name_nod_var" /* names of nodal variables  */
#define VAR_NOD_VAR "vals_nod_var" /* values of nodal variables */
#define DIM_NUM_ELE_VAR "num_elem_var" /* # of element variables    */
#define VAR_NAME_ELE_VAR "name_elem_var" /* names of element variables*/
#define VAR_ELEM_VAR(num1, num2) ex_catstr2("vals_elem_var", num1, "eb", num2)
/* values of element variable*/
/*   num1 in element block   */
/*   num2                    */
#define DIM_NUM_HIS_VAR "num_his_var" /* obsolete                  */
#define VAR_NAME_HIS_VAR "name_his_var" /* obsolete                  */
#define VAR_HIS_VAR "vals_his_var" /* obsolete                  */
#define DIM_STR "len_string" /* general dimension of      */
/*   length MAX_STR_LENGTH   */
/*   used for name lengths   */
#define DIM_LIN "len_line" /* general dimension of      */
/*   length MAX_LINE_LENGTH  */
/*   used for long strings   */
#define DIM_N4 "four" /* general dimension of      */
/*   length 4                */
#define DIM_TIME "time_step" /* unlimited (expandable)    */
/*   dimension for time steps*/
#define DIM_HTIME "hist_time_step" /* obsolete                  */
#define VAR_ELEM_NUM_MAP "elem_num_map" /* element numbering map     */
#define VAR_NODE_NUM_MAP "node_num_map" /* node numbering map        */

#define TRIANGLE 1 /* Triangle entity */
#define QUAD 2 /* Quad entity */
#define HEX 3 /* Hex entity */
#define WEDGE 4 /* Wedge entity */
#define TETRA 5 /* Tetra entity */
#define TRUSS 6 /* Truss entity */
#define BEAM 7 /* Beam entity */
#define SHELL 8 /* Shell entity */
#define SPHERE 9 /* Sphere entity */
#define CIRCLE 10 /* Circle entity */

/* Internal structure declarations */

struct list_item
{ /* for use with ex_get_file_item */

    int exo_id;
    int value;
    struct list_item *next;
};

/* declare function prototypes.  recall that PROTO_ARGS() is a macro that
 * puts argument list in prototype if this is ANSI C.
 */

char *ex_catstr PROTO_ARGS((char *, int));
char *ex_catstr2 PROTO_ARGS((char *, int, char *, int));

enum convert_task
{
    RTN_ADDRESS,
    READ_CONVERT,
    WRITE_CONVERT
};
typedef int convert_task;

int ex_conv_ini PROTO_ARGS((int, int *, int *, int));
void ex_conv_exit PROTO_ARGS((int));
nc_type nc_flt_code PROTO_ARGS((int));
int ex_comp_ws PROTO_ARGS((int));
void *ex_conv_array PROTO_ARGS((int, convert_task, void *, int));

void ex_rm_file_item_eb PROTO_ARGS((int));
void ex_rm_file_item_ns PROTO_ARGS((int));
void ex_rm_file_item_ss PROTO_ARGS((int));

extern struct list_item *eb_ctr_list;
extern struct list_item *ns_ctr_list;
extern struct list_item *ss_ctr_list;

int ex_get_file_item PROTO_ARGS((int, struct list_item **));
int ex_inc_file_item PROTO_ARGS((int, struct list_item **));
void ex_rm_file_item PROTO_ARGS((int, struct list_item **));

#endif
