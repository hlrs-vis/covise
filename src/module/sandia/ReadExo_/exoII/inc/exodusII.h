/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*****************************************************************************
*
* exodusII.h - Exodus II include file, for general use
*
* author - Sandia National Laboratories
*          Larry A. Schoof - Original
*          James A. Schutt - 8 byte float and standard C definitions
*          Vic Yarberry    - Added headers and error logging
*
*          
* environment - UNIX
*
* exit conditions - 
*
* revision history - 
*
*  $Id: exodusII.h,v 1.19 1998/03/31 17:21:48 laschoo Exp $
*  $Log: exodusII.h,v $
*  Revision 1.19  1998/03/31 17:21:48  laschoo
*  changes for netCDF 3.4
*    added MAX_HEADER_SIZE constant;  the ex_put_init () function will not
*    allocate more than this for the header
*
*    modified EXODUS error codes to conform to (new) netCDF 3.4 error conventions;
*    positive errors are bad (fatal); negative are informative (warnings);
*    application codes are isolated from this behavior since the EXODUS functions
*    still return EX_OK (=0) for success, EX_FATAL (=-1) for failure, and
*    EX_WARN (=1) for success with warning;  the EXODUS error codes are printed
*    by the ex_err () function
*
*  Revision 1.18  1997/05/13 14:02:55  laschoo
*  added function prototype for ex_get_side_set_node_list
*
*  Revision 1.17  1996/12/24 19:46:03  laschoo
*  modified to allow multiple node and element maps;
*  changed API version to 2.09 and file version to 2.03
*
*  Revision 1.16  1996/08/12 16:24:05  laschoo
*  modified itol and ltoi function prototypes to use nclong for netcdf 2.4.2
*
*  Revision 1.15  1996/07/09 22:03:03  laschoo
*  changed version to 2.05
*
*  Revision 1.14  1995/09/20 17:37:31  mksmith
*  Upgrade to version 2.03
*
 * Revision 1.13  1994/08/24  15:29:49  laschoo
 * new API version 2.02 which includes Alpha support
 *
 * Revision 1.12  1994/03/30  15:37:42  vryarbe
 * Updated for V2.01
 *
 * Revision 1.11  1993/11/18  18:54:16  vryarbe
 * changed name of options flag to exoptval
 *
 * Revision 1.10  1993/10/18  19:46:34  vryarbe
 * removed const declaration from passed strings
 *
 * Revision 1.9  1993/09/24  21:08:45  vryarbe
 * added definitions for new inquire parameters
 *
 * Revision 1.8  1993/09/13  20:39:45  vryarbe
 * added new parameters for inquiry and a new error code
 *
 * Revision 1.6  1993/08/30  16:23:29  vryarbe
 * added return codes EX_WARN and EX_OK
 *
 * Revision 1.5  1993/08/27  22:52:52  vryarbe
 * modfied for use with property functions
 *
 * Revision 1.4  1993/08/23  20:44:33  vryarbe
 * Added CONST and some new error msg definitions.
 *
 * Revision 1.3  1993/07/08  21:49:06  vryarbe
 * bug fixes to date
 *
 * Revision 1.2  1993/07/01  22:27:41  vryarbe
 * updated header
 *
*
*****************************************************************************/
#include "netcdf.h"
#ifndef TRUE
#define TRUE -1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef EXODUS_II_HDR
#define EXODUS_II_HDR

/* NOTE:
 *	as of 21 August 1992, the EXODUS II library, C binding version, is
 *	compiled under C only.  However, this header file is designed so that
 *	it can be included into, and the library linked with, a C++ program.
 */

#if defined __STDC__ || defined __cplusplus
#define PROTO_ARGS(proto) proto
#define CONST_CHAR (const char *)
#define VOID_PTR (void *)
#else
#define PROTO_ARGS(proto) ()
#define CONST_CHAR
#define VOID_PTR
#endif

/* need following extern if this include file is used in a C++ program, to
 * keep the C++ compiler from mangling the function names.
 */
#ifdef __cplusplus
extern "C" {
#endif

/*
 * The following are miscellaneous constants used in the EXODUS II API.
 */

#define EX_NOCLOBBER 0
#define EX_CLOBBER 1

#define EX_READ 0
#define EX_WRITE 1

#define EX_INQ_FILE_TYPE 1 /* inquire EXODUS II file type*/
#define EX_INQ_API_VERS 2 /* inquire API version number */
#define EX_INQ_DB_VERS 3 /* inquire database version   */
/*   number                   */
#define EX_INQ_TITLE 4 /* inquire database title     */
#define EX_INQ_DIM 5 /* inquire number of          */
/*   dimensions               */
#define EX_INQ_NODES 6 /* inquire number of nodes    */
#define EX_INQ_ELEM 7 /* inquire number of elements */
#define EX_INQ_ELEM_BLK 8 /* inquire number of element  */
/*   blocks                   */
#define EX_INQ_NODE_SETS 9 /* inquire number of node sets*/
#define EX_INQ_NS_NODE_LEN 10 /* inquire length of node set */
/*   node list                */
#define EX_INQ_SIDE_SETS 11 /* inquire number of side sets*/
#define EX_INQ_SS_NODE_LEN 12 /* inquire length of side set */
/*   node list                */
#define EX_INQ_SS_ELEM_LEN 13 /* inquire length of side set */
/*   element list             */
#define EX_INQ_QA 14 /* inquire number of QA       */
/*   records                  */
#define EX_INQ_INFO 15 /* inquire number of info     */
/*   records                  */
#define EX_INQ_TIME 16 /* inquire number of time     */
/*   steps in the database    */
#define EX_INQ_EB_PROP 17 /* inquire number of element  */
/*   block properties         */
#define EX_INQ_NS_PROP 18 /* inquire number of node set */
/*   properties               */
#define EX_INQ_SS_PROP 19 /* inquire number of side set */
#define EX_INQ_NS_DF_LEN 20 /* inquire length of node set */
/*   distribution factor  list*/
#define EX_INQ_SS_DF_LEN 21 /* inquire length of node set */
/*   distribution factor  list*/
#define EX_INQ_LIB_VERS 22 /* inquire API Lib vers number*/
#define EX_INQ_EM_PROP 23 /* inquire number of element  */
/*   map properties           */
#define EX_INQ_NM_PROP 24 /* inquire number of node     */
/*   map properties           */
#define EX_INQ_ELEM_MAP 25 /* inquire number of element  */
/*   maps                     */
#define EX_INQ_NODE_MAP 26 /* inquire number of node     */
/*   maps                     */

/*   properties               */
#define EX_ELEM_BLOCK 1 /* element block property code*/
#define EX_NODE_SET 2 /* node set property code     */
#define EX_SIDE_SET 3 /* side set property code     */
#define EX_ELEM_MAP 4 /* element map property code  */
#define EX_NODE_MAP 5 /* node map property code     */

/*   max string lengths; constants that are used as netcdf dimensions must be
     of type long       */
#define MAX_STR_LENGTH 32L
#define MAX_VAR_NAME_LENGTH 20
#define MAX_LINE_LENGTH 80L
#define MAX_ERR_LENGTH 256

/*   for netCDF 3.4, we estimate the size of the header; 
     if estimate is larger than this max, set the estimate to this max;
     I've never measured a header larger than 20K   */
#define MAX_HEADER_SIZE 30000

/* declare function prototypes.  recall that PROTO_ARGS() is a macro that
 * puts argument list in prototype if this is ANSI C or C++.
 */

/* routines for file initialization i/o */

extern int ex_create PROTO_ARGS((const char *,
                                 int, int *, int *));
extern int ex_open PROTO_ARGS((const char *,
                               int, int *, int *, float *));
extern int ex_close PROTO_ARGS((int));
extern void ex_err PROTO_ARGS((char *, char *, int));
extern void ex_opts PROTO_ARGS((int));
extern int ex_update PROTO_ARGS((int));

extern int ex_put_init PROTO_ARGS((int, const char *, int, int,
                                   int, int, int, int));
extern int ex_get_init PROTO_ARGS((int, char *, int *, int *,
                                   int *, int *, int *, int *));

extern int ex_put_qa PROTO_ARGS((int, int, char *[][4]));
extern int ex_get_qa PROTO_ARGS((int, char *[][4]));

extern int ex_put_info PROTO_ARGS((int, int, char *[]));
extern int ex_get_info PROTO_ARGS((int, char *[]));

/* routines for model description i/o */

extern int ex_put_coord PROTO_ARGS((int, void *, void *,
                                    void *));
extern int ex_get_coord PROTO_ARGS((int, void *, void *,
                                    void *));

extern int ex_put_coord_names PROTO_ARGS((int, char *[]));
extern int ex_get_coord_names PROTO_ARGS((int, char *[]));

extern int ex_put_map PROTO_ARGS((int, int *));
extern int ex_get_map PROTO_ARGS((int, int *));

extern int ex_put_elem_block PROTO_ARGS((int, int, const char *, int,
                                         int, int));
extern int ex_get_elem_block PROTO_ARGS((int, int, char *, int *,
                                         int *, int *));

extern int ex_get_elem_blk_ids PROTO_ARGS((int, int *));

extern int ex_put_elem_conn PROTO_ARGS((int, int, int *));
extern int ex_get_elem_conn PROTO_ARGS((int, int, int *));

extern int ex_put_elem_attr PROTO_ARGS((int, int, void *));
extern int ex_get_elem_attr PROTO_ARGS((int, int, void *));

extern int ex_put_node_set_param PROTO_ARGS((int, int, int, int));
extern int ex_get_node_set_param PROTO_ARGS((int, int, int *, int *));

extern int ex_put_node_set PROTO_ARGS((int, int, int *));
extern int ex_get_node_set PROTO_ARGS((int, int, int *));

extern int ex_put_node_set_dist_fact PROTO_ARGS((int, int, void *));
extern int ex_get_node_set_dist_fact PROTO_ARGS((int, int, void *));

extern int ex_get_node_set_ids PROTO_ARGS((int, int *));

extern int ex_put_concat_node_sets PROTO_ARGS((int, int *, int *, int *,
                                               int *, int *, int *, void *));
extern int ex_get_concat_node_sets PROTO_ARGS((int, int *, int *, int *,
                                               int *, int *, int *, void *));

extern int ex_put_side_set_param PROTO_ARGS((int, int, int, int));
extern int ex_get_side_set_param PROTO_ARGS((int, int, int *, int *));

extern int ex_put_side_set PROTO_ARGS((int, int, int *, int *));
extern int ex_get_side_set PROTO_ARGS((int, int, int *, int *));
extern int ex_put_side_set_dist_fact PROTO_ARGS((int, int, void *));
extern int ex_get_side_set_dist_fact PROTO_ARGS((int, int, void *));
extern int ex_get_side_set_ids PROTO_ARGS((int, int *));
extern int ex_get_side_set_node_list PROTO_ARGS((int, int, int *, int *));

extern int ex_put_prop_names PROTO_ARGS((int, int, int, char **));
extern int ex_get_prop_names PROTO_ARGS((int, int, char **));

extern int ex_put_prop PROTO_ARGS((int, int, int, char *,
                                   int));
extern int ex_get_prop PROTO_ARGS((int, int, int, char *,
                                   int *));

extern int ex_put_prop_array PROTO_ARGS((int, int, char *, int *));
extern int ex_get_prop_array PROTO_ARGS((int, int, char *, int *));

extern int ex_put_concat_side_sets PROTO_ARGS((int, int *, int *, int *,
                                               int *, int *, int *, int *,
                                               void *));
extern int ex_get_concat_side_sets PROTO_ARGS((int, int *, int *, int *,
                                               int *, int *, int *, int *,
                                               void *));
extern int ex_cvt_nodes_to_sides PROTO_ARGS((int, int *, int *, int *,
                                             int *, int *, int *, int *));

/* routines for analysis results i/o */

extern int ex_put_var_param PROTO_ARGS((int, const char *, int));
extern int ex_get_var_param PROTO_ARGS((int, char *, int *));

extern int ex_put_var_names PROTO_ARGS((int, const char *, int,
                                        char *[]));
extern int ex_get_var_names PROTO_ARGS((int, char *, int,
                                        char *[]));

extern int ex_put_var_name PROTO_ARGS((int, const char *, int,
                                       char *));
extern int ex_get_var_name PROTO_ARGS((int, char *, int,
                                       char *));

extern int ex_put_elem_var_tab PROTO_ARGS((int, int, int, int *));
extern int ex_get_elem_var_tab PROTO_ARGS((int, int, int, int *));

extern int ex_put_glob_vars PROTO_ARGS((int, int, int, void *));
extern int ex_get_glob_vars PROTO_ARGS((int, int, int, void *));

extern int ex_get_glob_var_time PROTO_ARGS((int, int, int, int,
                                            void *));

extern int ex_put_nodal_var PROTO_ARGS((int, int, int, int,
                                        void *));
extern int ex_get_nodal_var PROTO_ARGS((int, int, int, int,
                                        void *));

extern int ex_get_nodal_var_time PROTO_ARGS((int, int, int, int, int,
                                             void *));

extern int ex_put_elem_var PROTO_ARGS((int, int, int, int, int,
                                       void *));
extern int ex_get_elem_var PROTO_ARGS((int, int, int, int, int,
                                       void *));

extern int ex_get_elem_var_time PROTO_ARGS((int, int, int, int, int,
                                            void *));

extern int ex_put_time PROTO_ARGS((int, int, void *));
extern int ex_get_time PROTO_ARGS((int, int, void *));

extern int ex_get_all_times PROTO_ARGS((int, void *));

extern int ex_inquire PROTO_ARGS((int, int, int *, void *,
                                  char *));
extern int ex_put_elem_num_map PROTO_ARGS((int, int *));
extern int ex_get_elem_num_map PROTO_ARGS((int, int *));

extern int ex_put_node_num_map PROTO_ARGS((int, int *));
extern int ex_get_node_num_map PROTO_ARGS((int, int *));

extern int ex_put_map_param PROTO_ARGS((int, int, int));
extern int ex_get_map_param PROTO_ARGS((int, int *, int *));

extern int ex_put_elem_map PROTO_ARGS((int, int, int *));
extern int ex_get_elem_map PROTO_ARGS((int, int, int *));

extern int ex_put_node_map PROTO_ARGS((int, int, int *));
extern int ex_get_node_map PROTO_ARGS((int, int, int *));

extern nclong *itol PROTO_ARGS((int *, int));

extern int ltoi PROTO_ARGS((nclong *, int *, int));

extern int ex_copy PROTO_ARGS((int, int));

extern int cpy_att PROTO_ARGS((int, int, int, int));

extern int cpy_var_def PROTO_ARGS((int, int, int, char *));

extern int cpy_var_val PROTO_ARGS((int, int, char *));

#ifdef __cplusplus
} /* close brackets on extern "C" declaration */
#endif

#endif

/* ERROR CODE DEFINITIONS AND STORAGE                                       */
extern int exerrval; /* shared error return value                */
extern int exoptval; /* error reporting flag (default is quiet)  */

/* ex_opts function codes - codes are OR'ed into exopts                     */
#define EX_VERBOSE 1 /* verbose mode message flag                */
#define EX_DEBUG 2 /* debug mode def                           */
#define EX_ABORT 4 /* abort mode flag def                      */

/* Exodus error return codes - function return values:                      */
#define EX_FATAL -1 /* fatal error flag def                     */
#define EX_OK 0 /* no error flag def                        */
#define EX_WARN 1 /* warning flag def                         */

/* Exodus error return codes - exerrval return values:                      */
#define EX_MEMFAIL 1000 /* memory allocation failure flag def       */
#define EX_BADFILEMODE 1001 /* bad file mode def                        */
#define EX_BADFILEID 1002 /* bad file id def                          */
#define EX_WRONGFILETYPE 1003 /* wrong file type for function             */
#define EX_LOOKUPFAIL 1004 /* id table lookup failed                   */
#define EX_BADPARAM 1005 /* bad parameter passed                     */
#define EX_NULLENTITY -1006 /* null entity found                        */
#define EX_MSG -1000 /* message print code - no error implied    */
#define EX_PRTLASTMSG -1001 /* print last error message msg code        */
