/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/************************************************************************
 *                  CFDRC Data Transfer Format Library                   *
 *                    (c) CFD Research Corporation                       *
 *                        Huntsville, Alabama, USA.                      *
 *                               2000.                                   *
 ************************************************************************/

/************************************************************************
 *       FILE:   dtf.h		                                        *
 *                                                                       *
 *       Public header file for DTF library				*
 *                                                                       *
 *        William J. Coirier / David M. Fricker                          *
 ************************************************************************/

#ifndef DTF_H
#define DTF_H

#ifdef __cplusplus
extern "C" {
#endif

#include <sys/types.h>

/***********************************************************************
    * Protection for non-ANSI C compilers which may not support const
    ***********************************************************************/

#define COMPILER_SUPPORTS_CONST 1

#ifdef COMPILER_SUPPORTS_CONST
#define CONST const
#else
#define CONST
#endif

/***********************************************************************/
/* Flags to pass to dtf_file_contents                                  */
/***********************************************************************/

#define CONTENTS_VERBOSE -1
#define CONTENTS_NOT_VERBOSE 10

/***********************************************************************/
/* basic variable types used throughout the library                    */
/***********************************************************************/

#define DTF_MAX_STRING 80
#define DTF_MAX_UNITS 32

typedef char dtf_string[DTF_MAX_STRING];
typedef dtf_string *PDTFSTRING;
typedef char dtf_units[DTF_MAX_UNITS];
typedef int dtf_int;
typedef float dtf_single;
typedef double dtf_double;

typedef dtf_int dtf_handle;

#if defined(_DEC_) || (_MIPS_SIM == _NABI32) || (_MIPS_SIM == _MIPS_SIM_ABI64) || defined(_SUN64_)
typedef unsigned int dtf_long;
#else
typedef long dtf_long;
#endif

#if defined(_HPUX11_) || defined(_SUN64_)
typedef dtf_int dtf_time;
#else
typedef time_t dtf_time;
#endif

/***********************************************************************/
/* error codes that the library may return                             */
/***********************************************************************/

typedef enum dtf_errcode
{
    DTF_ERROR = -1,
    DTF_OK = 0,
    DTF_UNSUPPORTED_ID,
    DTF_BAD_ZONETYPE,
    DTF_BAD_FH,
    DTF_BAD_SIMNUM,
    DTF_BAD_ZONENUM,
    DTF_BAD_SWAPSIZE,
    DTF_BAD_MODE,
    DTF_BAD_HANDLE,
    DTF_FILE_IS_NOT_DTF,
    DTF_BAD_DATANUM,
    DTF_CANT_OPEN,
    DTF_CANT_CREATE,
    DTF_BAD_FACENUM,
    DTF_BAD_CELLNUM,
    DTF_BAD_DATATYPE,
    DTF_BAD_TOPOTYPE,
    DTF_BAD_BCNUM,
    DTF_BAD_SIZES,
    DTF_BAD_STRUCT_DIMS,
    DTF_BAD_CELL_GROUPNUM,
    DTF_BAD_BLOCKNUM,
    DTF_UNDER_CONSTRUCTION,
    DTF_NO_USER_FACE_DATA,
    DTF_NOT_A_BC_FACE,
    DTF_NOT_A_INTERFACE_FACE,
    DTF_2D_NOT_DEFINED,
    DTF_OUT_OF_MEMORY,
    DTF_BAD_CATEGORY_NUMBER,
    DTF_BAD_BCVAL_NUMBER,
    DTF_BAD_BCVAL_NAME,
    DTF_BAD_BCVAL_INT_NAME,
    DTF_BAD_BCVAL_REAL_NAME,
    DTF_BAD_BCVAL_STRING_NAME,
    DTF_BAD_ELEMENT_INDEX,
    DTF_BAD_VCNUM,
    DTF_BAD_VCVAL_NUMBER,
    DTF_BAD_VCVAL_NAME,
    DTF_BAD_VCVAL_INT_NAME,
    DTF_BAD_VCVAL_REAL_NAME,
    DTF_BAD_VCVAL_STRING_NAME,
    DTF_BAD_VOLUME_CONDITIONNUM,
    DTF_NO_STRING_ARRAYS_IN_FORTRAN,
    DTF_BAD_PATCHNUM,
    DTF_CORRUPTED_OLD_VC_DATA,
    DTF_OFFSET_NOT_FOUND,
    DTF_BAD_F2N_DATA,
    DTF_BAD_F2C_DATA,
    DTF_BAD_OFFSET,
    DTF_BAD_CONNECTIVITY_TABLE,
    DTF_FILE_IS_LOCKED,
    DTF_BAD_F2C_STRING,
    DTF_BAD_C2F_STRING,
    DTF_NO_ELEMENTAL_READ_VZONE,
    DTF_NEWER_FILE_VERSION,
    DTF_LHG,
    DTF_BAD_PATCH,
    DTF_BAD_INTERFACE_FACE,
    DTF_BAD_HASH,
    DTF_BAD_NODES_TO_CELL_MAP,
    DTF_NO_BC_DATA,
    DTF_BAD_FGNUM,
    DTF_CANT_UPDATE_FGS,
    DTF_BAD_SCNUM,
    DTF_CANT_UPDATE_SCS,
    DTF_BAD_INTERIOR_FACE,
    DTF_BFACE_NO_BCREC,
    DTF_XFACE_NO_BCREC,
    DTF_BAD_ZINUM,
    DTF_BAD_MALLOC_N,
    DTF_BAD_MALLOC_SZ,
    DTF_BAD_DATA_ARB_PROC_MODE,
    DTF_BAD_PERT_MODE_THIN_WALL,
    DTF_ASSERTION,
    DTF_BAD_ARB_INT_PAIR,
    DTF_BAD_STRUCT_INDICES,
    DTF_BAD_DESTRUCT_MODE,
    DTF_NO_BLANKING_DATA,
    DTF_BAD_BLANKING_UPDATE,
    DTF_NULL_BLANKING_ARRAY,
    DTF_NELEMENTS_MISMATCH,
    DTF_ARBINT_IN_DEGRADE,
    DTF_WRITING_ERROR,
    DTF_BAD_FACE_MAP
} dtf_errcode;

/***********************************************************************/
/* permissible datatypes, used by dtf_add_data, dtf_read_data, etc     */
/***********************************************************************/

typedef enum dtf_datatype
{
    DTF_INT_DATA = 1,
    DTF_DOUBLE_DATA = 2,
    DTF_SINGLE_DATA = 3,
    DTF_STRING_DATA = 4
} dtf_datatype;

/***********************************************************************/
/* permissible topotypes, used by dtf_add_data, dtf_read_data, etc     */
/***********************************************************************/

typedef enum dtf_topotype
{
    DTF_GNRL_TOPO = 0,
    DTF_NODE_TOPO = 1,
    DTF_EDGE_TOPO = 2,
    DTF_FACE_TOPO = 3,
    DTF_CELL_TOPO = 4
} dtf_topotype;

/***********************************************************************/
/* type of a zone, returned by dtf_query_zonetype                      */
/***********************************************************************/

typedef enum dtf_zonetype
{
    DTF_STRUCTURED_ZONE = 1,
    DTF_CARTESIAN_ZONE = 2,
    DTF_UNSTRUCTURED_ZONE = 3,
    DTF_POINT_NET_ZONE = 4,
    DTF_POLY_ZONE = 5,
    DTF_VZ_MODE_ZONE = 6
} dtf_zonetype;

/***********************************************************************/
/* some constant definitions for unstructured grids                    */
/***********************************************************************/

/* faces:
    *     Edge,
    *     Triangle,
    *     Quadrilateral,
    *     Edge-Quadratic,
    *     Triangle-Quadratic,
    *     Quadrilateral-Quadratic,
    *     Poly-Face
    */
#define DTF_NFACETYPES 7
#define DTF_NFACEKINDS DTF_NFACETYPES * 3
#define DTF_UNST_NFACETYPES 3
#define DTF_UNST_NFACEKINDS DTF_UNST_NFACETYPES * 3
extern const char *DTF_FACETYPE_TO_NAME[];
/* { 2, 3, 4, 3, 6, 8, -1 } */
extern const int DTF_FACETYPE_TO_NNODES[];

typedef enum dtf_poly_face_type
{
    DTF_INVALID_FACE = (-1),
    DTF_EDGE_FACE = 0,
    DTF_TRI_FACE = 1,
    DTF_QUAD_FACE = 2,
    DTF_EDGE_Q_FACE = 3,
    DTF_TRI_Q_FACE = 4,
    DTF_QUAD_Q_FACE = 5,
    DTF_POLY_FACE = 6
} dtf_poly_face_type;

/* cells:
    *     Triangle,
    *     Quadrilateral,
    *     Tetrahedron,
    *     Pyramid,
    *     Prism,
    *     Hexahedron,
    *     Triangle-Quadratic,
    *     Quadrilateral-Quadratic,
    *     Tetrahedron-Quadratic,
    *     Hexahedron-Quadratic,
    *     Poly-Cell
    */
#define DTF_NCELLTYPES 11
#define DTF_UNST_NCELLTYPES 6
extern const char *DTF_CELLTYPE_TO_NAME[];
/* { 3, 4, 4, 5, 6, 8, 6, 8, 10, 20, -1 } */
extern const int DTF_CELLTYPE_TO_NNODES[];
/* { 3, 4, 4, 5, 5, 6, 3, 4, 4, 6, -1 } */
extern const int DTF_CELLTYPE_TO_NFACES[];

typedef enum dtf_poly_cell_type
{
    DTF_INVALID_CELL = (-1),
    DTF_TRI_CELL = 0,
    DTF_QUAD_CELL = 1,
    DTF_TET_CELL = 2,
    DTF_PYR_CELL = 3,
    DTF_PRISM_CELL = 4,
    DTF_HEX_CELL = 5,
    DTF_TRI_Q_CELL = 6,
    DTF_QUAD_Q_CELL = 7,
    DTF_TET_Q_CELL = 8,
    DTF_HEX_Q_CELL = 9,
    DTF_POLY_CELL = 10
} dtf_poly_cell_type;

/***********************************************************************/
/* face types                                                          */
/***********************************************************************/

typedef enum dtf_facetype
{
    DTF_INTERIOR_FACE = 1,
    DTF_INTERFACE_FACE = 2,
    DTF_BOUNDARY_FACE = 3
} dtf_facetype;

/***********************************************************************/
/* TRUE/FALSE constants                                                */
/***********************************************************************/

#define DTF_TRUE 1
#define DTF_FALSE 0

/***********************************************************************/
/* function prototypes                                                 */
/*                                                                     */
/* please note the meaning of the following suffixes:                  */
/*    _d   - double precision                                          */
/*    _s   - single precision                                          */
/*                                                                     */
/* all functions return -1 (DTF_ERROR) in case of some error           */
/* use dtf_last_error to get the error code of the last error          */
/***********************************************************************/

/* __BEGIN__ */

/***********************************************************************/
/* VERSION CONTROL                                                     */
/***********************************************************************/

/* get DTF version of the library */
dtf_int dtf_query_dtf_version(/* error code */
                              dtf_string dtf_version /* o: DTF version */
                              );

/* get DTF version of the file */
dtf_int dtf_query_file_version(/* error code */
                               CONST dtf_handle *fh, /* i: file handle */
                               dtf_string dtf_version /* o: DTF version */
                               );

/***********************************************************************/
/* ERROR HANDLING                                                      */
/***********************************************************************/

/* find out whether there was an error */
dtf_int dtf_ok(/* DTF_TRUE or DTF_FALSE */
               void /* n/a */
               );

/* get the last error code */
dtf_errcode dtf_last_error(/* error code */
                           void /* n/a */
                           );

/* return the error string and code corresponding to the last error */
dtf_errcode dtf_info_last_error(/* error code */
                                dtf_string error_string /* o: A description about the last error */
                                );

/* clear error code flag */
dtf_int dtf_clear_error(/* error code */
                        void /* n/a */
                        );

/***********************************************************************/
/* GENERAL FILE OPERATIONS                                             */
/***********************************************************************/

/* create new file */
dtf_handle dtf_new_file(/* new file handle */
                        CONST char *filename /* i: name of the file to create */
                        );

/* open existing file */
dtf_handle dtf_open_file(/* new file handle */
                         CONST char *filename /* i: name of the file to open */
                         );

/* close opened/created file */
dtf_int dtf_close_file(/* error code */
                       CONST dtf_handle *fh /* i: name of the file to close */
                       );

/* close all open files */
dtf_int dtf_close_all_files(/* error code */
                            );

/* set scaling factor */
dtf_int dtf_set_scaling_d(/* error code */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_double *scaling /* i: scaling factor */
                          );

dtf_int dtf_set_scaling_s(/* error code */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_single *scaling /* i: scaling factor */
                          );

/* get scaling factor from the file */
dtf_double dtf_query_scaling_d(/* scaling factor */
                               CONST dtf_handle *fh /* i: file handle */
                               );

dtf_single dtf_query_scaling_s(/* scaling factor */
                               CONST dtf_handle *fh /* i: file handle */
                               );

/* set application name */
dtf_int dtf_set_application(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_string application /* i: application name */
                            );

/* get application name from the file */
dtf_int dtf_query_application(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              dtf_string application /* o: application name */
                              );

/* set version of the application */
dtf_int dtf_set_appversion(/* error code */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_string appversion /* i: application version */
                           );

/* get application version from the file */
dtf_int dtf_query_appversion(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             dtf_string appversion /* o: application version */
                             );

/* set file title */
dtf_int dtf_set_title(/* error code */
                      CONST dtf_handle *fh, /* i: file handle */
                      CONST dtf_string title /* i: file title */
                      );

/* get file title from the file */
dtf_int dtf_query_title(/* error code */
                        CONST dtf_handle *fh, /* i: file handle */
                        dtf_string title /* o: file title */
                        );

/* set file origin */
dtf_int dtf_set_origin(/* error code */
                       CONST dtf_handle *fh, /* i: file handle */
                       CONST dtf_string origin /* i: file origin */
                       );

/* get file origin from the file */
dtf_int dtf_query_origin(/* error code */
                         CONST dtf_handle *fh, /* i: file handle */
                         dtf_string origin /* o: file origin */
                         );

/* get creation time */
dtf_time dtf_query_cretime(/* creation time */
                           CONST dtf_handle *fh /* i: file handle */
                           );

/* get last modification time */
dtf_time dtf_query_modtime(/* last modification time */
                           CONST dtf_handle *fh /* i: file handle */
                           );

/* check file for duplicate patches */
dtf_int dtf_check_patches(/* error code */
                          CONST char *filename /* i: name of the DTF file */
                          );

/* print human-readable file info to the stdout */
dtf_int dtf_file_info(/* error code */
                      CONST char *filename /* i: name of the DTF file */
                      );

/* print file contents (all individual sections) to the stdout */
dtf_int dtf_file_contents(/* error code */
                          CONST char *filename, /* i: name of the DTF file */
                          dtf_int max_array_print, /* i: maximum number of array elements printed (-1 for all elements) */
                          dtf_int is_html /* i: DTF_TRUE or DTF_FALSE, whether to print as HTML */
                          );

/* test given file for corrupt section references */
dtf_int dtf_test_file(/* error code */
                      CONST char *filename /* i: name of the DTF file */
                      );

/* test given sim in given file for validity */
dtf_int dtf_test_validity(/* error code */
                          CONST char *filename, /* i: name of the DTF file */
                          CONST dtf_int *sim /* i: sim to test (0=>all) */
                          );

/***********************************************************************/
/* SIMULATION OPERATIONS                                               */
/***********************************************************************/

/* get number of simulations from the file */
dtf_int dtf_query_nsims(/* number of simulations */
                        CONST dtf_handle *fh /* i: file handle */
                        );

/* add new simulation to the file */
dtf_int dtf_add_sim(/* number of the simulation added */
                    CONST dtf_handle *fh, /* i: file handle */
                    CONST dtf_string descr /* i: description of the simulation */
                    );

/* create a link of an existing simulation */
dtf_int dtf_copy_sim(/* number of the simulation linked */
                     CONST dtf_handle *fh, /* i: file handle */
                     CONST dtf_int *simnum, /* i: simulation to link */
                     CONST dtf_string descr /* i: description of the simulation */
                     );

/* delete a simulation from the file */
dtf_int dtf_delete_sim(/* error code */
                       CONST dtf_handle *fh, /* i: file handle */
                       CONST dtf_int *simnum /* i: simulation number */
                       );

/* update simulation information */
dtf_int dtf_update_simdescr(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_string descr /* i: description of the simulation */
                            );

/* get simulation description */
dtf_int dtf_query_simdescr(/* error code */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           dtf_string descr /* o: description of the simulation */
                           );

/* get grid's xyz range of the whole simulation */
dtf_int dtf_query_minmax_sim_d(/* error code */
                               CONST dtf_handle *fh, /* i: file handle */
                               CONST dtf_int *simnum, /* i: simulation number */
                               dtf_double *minmax /* o: grid xyz range */
                               );

dtf_int dtf_query_minmax_sim_s(/* error code */
                               CONST dtf_handle *fh, /* i: file handle */
                               CONST dtf_int *simnum, /* i: simulation number */
                               dtf_single *minmax /* o: grid xyz range */
                               );

/***********************************************************************/
/* ZONE OPERATIONS                                                     */
/***********************************************************************/

/* get number of zones in the simulation */
dtf_int dtf_query_nzones(/* number of zones in the simulation */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum /* i: simulation number */
                         );

/* find out the type of the zone */
dtf_int dtf_query_zonetype(/* zone type */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *zonenum /* i: zone number */
                           );

/* find out the type of the zone */
dtf_int dtf_get_zonetype(/* zone type */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         dtf_zonetype *ztype);

/* query whether the zone is structured */
dtf_int dtf_query_isstruct(/* DTF_TRUE or DTF_FALSE */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *zonenum /* i: zone number */
                           );

/* query whether the zone is cartesian */
dtf_int dtf_query_iscartesian(/* DTF_TRUE or DTF_FALSE */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum /* i: zone number */
                              );

/* query whether the zone is unstructured */
dtf_int dtf_query_isunstruct(/* DTF_TRUE or DTF_FALSE */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum /* i: zone number */
                             );

/* query whether the zone is point net */
dtf_int dtf_query_ispoint(/* DTF_TRUE or DTF_FALSE */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum, /* i: simulation number */
                          CONST dtf_int *zonenum /* i: zone number */
                          );

/* query whether the zone is a polyzone */
dtf_int dtf_query_ispoly(/* DTF_TRUE or DTF_FALSE */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum /* i: zone number */
                         );

/* check if the zone is 2D */
dtf_int dtf_query_is_2d(/* Boolean: True if zone is 2D */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum, /* i: sim number */
                        CONST dtf_int *zonenum /* i: zone number */
                        );

/* delete specified zone */
dtf_int dtf_delete_zone(/* error code */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum, /* i: simulation number */
                        CONST dtf_int *zonenum /* i: zone number */
                        );

/* determine if the zone has blanking information */
dtf_int dtf_is_blanking_data_present(/* o: DTF_TRUE or DTF_FALSE */
                                     CONST dtf_handle *fh, /* i: file handle */
                                     CONST dtf_int *simnum, /* i: simulation number */
                                     CONST dtf_int *zonenum /* i: zone number */
                                     );

/* get zone dimensions (IJK for structured zone) */
dtf_int dtf_query_dims(/* error code */
                       CONST dtf_handle *fh, /* i: file handle */
                       CONST dtf_int *simnum, /* i: simulation number */
                       CONST dtf_int *zonenum, /* i: zone number */
                       dtf_int *dim /* o: IJK dimensions of the grid */
                       );

/* get number of nodes in the zone */
dtf_int dtf_query_nnodes(/* number of nodes */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum /* i: zone number */
                         );

/* get number of nodes in a structured zone */
dtf_int dtf_query_nnodes_struct(/* number of nodes */
                                CONST dtf_handle *fh, /* i: file handle */
                                CONST dtf_int *simnum, /* i: simulation number */
                                CONST dtf_int *zonenum /* i: zone number */
                                );

/* Update the coordinates and blanking in a zone */
dtf_int dtf_update_grid_d(/* error code */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum, /* i: simulation number */
                          CONST dtf_int *zonenum, /* i: zone number */
                          CONST dtf_double *x, /* i: array of X coordinates */
                          CONST dtf_double *y, /* i: array of Y coordinates */
                          CONST dtf_double *z, /* i: array of Z coordinates */
                          CONST dtf_int *blanking /* i: iblank array  */
                          );

/* Update the coordinates and blanking in a zone */
dtf_int dtf_update_grid_s(/* error code */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum, /* i: simulation number */
                          CONST dtf_int *zonenum, /* i: zone number */
                          CONST dtf_single *x, /* i: array of X coordinates */
                          CONST dtf_single *y, /* i: array of Y coordinates */
                          CONST dtf_single *z, /* i: array of Z coordinates */
                          CONST dtf_int *blanking /* i: iblank array  */
                          );

/* read xyz coordinates of the grid */
dtf_int dtf_read_grid_d(/* error code */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum, /* i: simulation number */
                        CONST dtf_int *zonenum, /* i: zone number */
                        CONST dtf_int *nodenum, /* i: node number (< 1 means all) */
                        dtf_double *x, /* o: array of X coordinates */
                        dtf_double *y, /* o: array of Y coordinates */
                        dtf_double *z, /* o: array of Z coordinates */
                        dtf_int *blanking /* o: iblank array  */
                        );

dtf_int dtf_read_grid_s(/* error code */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum, /* i: simulation number */
                        CONST dtf_int *zonenum, /* i: zone number */
                        CONST dtf_int *nodenum, /* i: node number (< 1 means all) */
                        dtf_single *x, /* o: array of X coordinates */
                        dtf_single *y, /* o: array of Y coordinates */
                        dtf_single *z, /* o: array of Z coordinates */
                        dtf_int *blanking /* o: iblank array  */
                        );

/* get grid's xyz range of the zone */
dtf_int dtf_query_minmax_zone_d(/* error code */
                                CONST dtf_handle *fh, /* i: file handle */
                                CONST dtf_int *simnum, /* i: simulation number */
                                CONST dtf_int *zonenum, /* i: zone number */
                                dtf_double *minmax /* o: grid xyz range */
                                );

dtf_int dtf_query_minmax_zone_s(/* error code */
                                CONST dtf_handle *fh, /* i: file handle */
                                CONST dtf_int *simnum, /* i: simulation number */
                                CONST dtf_int *zonenum, /* i: zone number */
                                dtf_single *minmax /* o: grid xyz range */
                                );

/* New functions to access blanking data directly */
dtf_int dtf_read_blanking(/* error code */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum, /* i: simulation number */
                          CONST dtf_int *zonenum, /* i: zone number */
                          CONST dtf_int *nodenum, /* i: node number (< 1 means all) */
                          dtf_int *blanking /* o: iblank array  */
                          );

dtf_int dtf_update_blanking(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *zonenum, /* i: zone number */
                            CONST dtf_int *blanking /* i: iblank array  */
                            );

/***********************************************************************/
/* STRUCTURED-SPECIFIC FUNCTIONS                                       */
/***********************************************************************/

/* add structured grid to the simulation */
dtf_int dtf_add_struct_d(/* zone number */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *dim, /* i: IJK dimensions */
                         CONST dtf_double *x, /* i: array of X coordinates */
                         CONST dtf_double *y, /* i: array of Y coordinates */
                         CONST dtf_double *z, /* i: array of Z coordinates */
                         CONST dtf_int *blanking /* i: iblank array */
                         );

dtf_int dtf_add_struct_s(/* error code */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *dim, /* i: IJK dimensions */
                         CONST dtf_single *x, /* i: array of X coordinates */
                         CONST dtf_single *y, /* i: array of Y coordinates */
                         CONST dtf_single *z, /* i: array of Z coordinates */
                         CONST dtf_int *blanking /* i: iblank array */
                         );

dtf_int dtf_update_struct_double(/* error code */
                                 CONST dtf_handle *fh, /* i: file handle */
                                 CONST dtf_int *simnum, /* i: simulation number */
                                 CONST dtf_int *zonenum, /* i: zone number */
                                 CONST dtf_int *dim, /* i: IJK dimensions */
                                 CONST dtf_double *x, /* i: array of X coordinates */
                                 CONST dtf_double *y, /* i: array of Y coordinates */
                                 CONST dtf_double *z, /* i: array of Z coordinates */
                                 CONST dtf_int *blanking /* i: iblank array */
                                 );

dtf_int dtf_update_struct_single(/* error code */
                                 CONST dtf_handle *fh, /* i: file handle */
                                 CONST dtf_int *simnum, /* i: simulation number */
                                 CONST dtf_int *zonenum, /* i: zone number */
                                 CONST dtf_int *dim, /* i: IJK dimensions */
                                 CONST dtf_single *x, /* i: array of X coordinates */
                                 CONST dtf_single *y, /* i: array of Y coordinates */
                                 CONST dtf_single *z, /* i: array of Z coordinates */
                                 CONST dtf_int *blanking /* i: iblank array */
                                 );

/***********************************************************************/
/* UNSTRUCTURED-SPECIFIC FUNCTIONS                                     */
/***********************************************************************/

/* add unstructured grid to the simulation */
dtf_int dtf_add_unstruct_d(/* zone number of the newly added zone */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *nnodes, /* i: number of nodes */
                           CONST dtf_double *x, /* i: array of X coordinates */
                           CONST dtf_double *y, /* i: array of Y coordinates */
                           CONST dtf_double *z, /* i: array of Z coordinates */
                           CONST dtf_int *ncells, /* i: number of cells sorted by celltype */
                           CONST dtf_int *cells /* i: cell-to-node connectivity */
                           );

dtf_int dtf_add_unstruct_s(/* error code */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *nnodes, /* i: number of nodes */
                           CONST dtf_single *x, /* i: array of X coordinates */
                           CONST dtf_single *y, /* i: array of Y coordinates */
                           CONST dtf_single *z, /* i: array of Z coordinates */
                           CONST dtf_int *ncells, /* i: number of cells sorted by celltype */
                           CONST dtf_int *cells /* i: cell-to-node connectivity */
                           );

/* update unstructured grid with new xyz coordinates */
dtf_int dtf_update_unstruct_double(/* zone number of the newly added zone */
                                   CONST dtf_handle *fh, /* i: file handle */
                                   CONST dtf_int *simnum, /* i: simulation number */
                                   CONST dtf_int *zonenum, /* i: zone number */
                                   CONST dtf_int *nnodes, /* i: number of nodes */
                                   CONST dtf_double *x, /* i: array of X coordinates */
                                   CONST dtf_double *y, /* i: array of Y coordinates */
                                   CONST dtf_double *z, /* i: array of Z coordinates */
                                   CONST dtf_int *ncells, /* i: number of cells sorted by celltype */
                                   CONST dtf_int *cells /* i: cell-to-node connectivity */
                                   );

dtf_int dtf_update_unstruct_single(/* error code */
                                   CONST dtf_handle *fh, /* i: file handle */
                                   CONST dtf_int *simnum, /* i: simulation number */
                                   CONST dtf_int *zonenum, /* i: zone number */
                                   CONST dtf_int *nnodes, /* i: number of nodes */
                                   CONST dtf_single *x, /* i: array of X coordinates */
                                   CONST dtf_single *y, /* i: array of Y coordinates */
                                   CONST dtf_single *z, /* i: array of Z coordinates */
                                   CONST dtf_int *ncells, /* i: number of cells sorted by celltype */
                                   CONST dtf_int *cells /* i: cell-to-node connectivity */
                                   );

/***********************************************************************/
/* POLYZONE-SPECIFIC FUNCTIONS                                         */
/***********************************************************************/

/* add a general polyhedral grid to the simulation */
dtf_int dtf_add_poly_d(/* zone number */
                       CONST dtf_handle *fh, /* i: file handle */
                       CONST dtf_int *simnum, /* i: simulation number */
                       CONST dtf_int *nnodes, /* i: number of nodes */
                       CONST dtf_double *x, /* i: array of X coordinates */
                       CONST dtf_double *y, /* i: array of Y coordinates */
                       CONST dtf_double *z, /* i: array of Z coordinates */
                       CONST dtf_int *n_faces_total, /* i: total number of faces */
                       CONST dtf_int *n_nodes_per_face, /* i: array containing number of nodes for each face */
                       CONST dtf_int *len_f2n, /* i: length of face->node array */
                       CONST dtf_int *f2n, /* i: face->node array */
                       CONST dtf_int *len_f2c, /* i: length of face->cell array */
                       CONST dtf_int *f2c, /* i: face->cell array */
                       CONST dtf_int *is_2D /* i: Mesh is 2D/3D */
                       );

dtf_int dtf_add_poly_s(/* zone number */
                       CONST dtf_handle *fh, /* i: file handle */
                       CONST dtf_int *simnum, /* i: simulation number */
                       CONST dtf_int *nnodes, /* i: number of nodes */
                       CONST dtf_single *x, /* i: array of X coordinates */
                       CONST dtf_single *y, /* i: array of Y coordinates */
                       CONST dtf_single *z, /* i: array of Z coordinates */
                       CONST dtf_int *n_faces_total, /* i: total number of faces */
                       CONST dtf_int *n_nodes_per_face, /* i: array containing number of nodes for each face */
                       CONST dtf_int *len_f2n, /* i: length of face->node array */
                       CONST dtf_int *f2n, /* i: face->node array */
                       CONST dtf_int *len_f2c, /* i: length of face->cell array */
                       CONST dtf_int *f2c, /* i: face->cell array */
                       CONST dtf_int *is_2D /* i: Mesh is 2D/3D */
                       );

/* update a general polyhedral grid to the simulation */
dtf_int dtf_update_poly_double(/* zone number */
                               CONST dtf_handle *fh, /* i: file handle */
                               CONST dtf_int *simnum, /* i: simulation number */
                               CONST dtf_int *zonenum, /* i: zone number */
                               CONST dtf_int *nnodes, /* i: number of nodes */
                               CONST dtf_double *x, /* i: array of X coordinates */
                               CONST dtf_double *y, /* i: array of Y coordinates */
                               CONST dtf_double *z, /* i: array of Z coordinates */
                               CONST dtf_int *n_faces_total, /* i: total number of faces */
                               CONST dtf_int *n_nodes_per_face, /* i: array containing number of nodes for each face */
                               CONST dtf_int *len_f2n, /* i: length of face->node array */
                               CONST dtf_int *f2n, /* i: face->node array */
                               CONST dtf_int *len_f2c, /* i: length of face->cell array */
                               CONST dtf_int *f2c, /* i: face->cell array */
                               CONST dtf_int *is_2D /* i: Mesh is 2D/3D */
                               );

dtf_int dtf_update_poly_single(/* zone number */
                               CONST dtf_handle *fh, /* i: file handle */
                               CONST dtf_int *simnum, /* i: simulation number */
                               CONST dtf_int *zonenum, /* i: zone number */
                               CONST dtf_int *nnodes, /* i: number of nodes */
                               CONST dtf_single *x, /* i: array of X coordinates */
                               CONST dtf_single *y, /* i: array of Y coordinates */
                               CONST dtf_single *z, /* i: array of Z coordinates */
                               CONST dtf_int *n_faces_total, /* i: total number of faces */
                               CONST dtf_int *n_nodes_per_face, /* i: array containing number of nodes for each face */
                               CONST dtf_int *len_f2n, /* i: length of face->node array */
                               CONST dtf_int *f2n, /* i: face->node array */
                               CONST dtf_int *len_f2c, /* i: length of face->cell array */
                               CONST dtf_int *f2c, /* i: face->cell array */
                               CONST dtf_int *is_2D /* i: Mesh is 2D/3D */
                               );

/* query whether the poly zone's connectivity data has been sorted according to the master list */
dtf_int dtf_query_ispoly_sorted(/* DTF_TRUE or DTF_FALSE */
                                CONST dtf_handle *fh, /* i: file handle */
                                CONST dtf_int *simnum, /* i: simulation number */
                                CONST dtf_int *zonenum /* i: zone number */
                                );

/* Sort the poly zone according to the master list */
dtf_int dtf_sort_poly(/* error code */
                      CONST dtf_handle *fh, /* i: file handle */
                      CONST dtf_int *simnum, /* i: simulation number */
                      CONST dtf_int *zonenum /* i: zone number */
                      );

/* Query the poly zone for some sizing specific data */
dtf_int dtf_query_poly_sizes(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum, /* i: zone number */
                             dtf_int *nnodes, /* o: number of nodes */
                             dtf_int *n_faces_total, /* o: number of faces */
                             dtf_int *n_bfaces_total, /* o: number of boundary faces */
                             dtf_int *n_xfaces_total, /* o: number of interface faces */
                             dtf_int *n_cells_total, /* o: number of cells */
                             dtf_int *len_f2n, /* o: number of elements in f2n array */
                             dtf_int *len_f2c, /* o: number of elements in f2c array */
                             dtf_int *len_c2n, /* o: number of elements in c2n array */
                             dtf_int *len_c2f /* o: number of elements in c2f array */
                             );

/***********************************************************************/
/* WORKING WITH FACES                                                  */
/***********************************************************************/

/* get total number of faces */
dtf_int dtf_query_nfaces(/* total number of faces */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum /* i: zone number */
                         );

/* get total number of faces for a structured zone (no degeneracies) */
dtf_int dtf_query_nfaces_struct(/* total number of faces */
                                CONST dtf_handle *fh, /* i: file handle */
                                CONST dtf_int *simnum, /* i: simulation number */
                                CONST dtf_int *zonenum /* i: zone number */
                                );

/* get face sizing data: number of faces total, for each type and kind */
dtf_int dtf_query_faces(/* total number of faces for this zone */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum, /* i: simulation number */
                        CONST dtf_int *zonenum, /* i: zone number */
                        dtf_int *n_faces_of_type, /* o: number of faces for each face type */
                        dtf_int *n_faces_of_kind /* o: number of faces for each face kind */
                        );

/* given face number, return offset in the face-to-node array */
dtf_int dtf_query_f2n_pos(/* return location in f2n array for start of face's f2n data */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum, /* i: simulation number */
                          CONST dtf_int *zonenum, /* i: zone number */
                          CONST dtf_int *facenum /* i: Face number of the face */
                          );

/* given face number, return face kind */
dtf_int dtf_query_facekind(/* return face kind */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *zonenum, /* i: zone number */
                           CONST dtf_int *facenum /* i: face number */
                           );

/***********************************************************************/
/* WORKING WITH CELLS                                                  */
/***********************************************************************/

/* returns total number of cells */
dtf_int dtf_query_ncells(/* total number of cells */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum /* i: zone number */
                         );

/* get cell sizing data: number of cells total, for each type */
dtf_int dtf_query_cells(/* total number of cells for this zone */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum, /* i: simulation number */
                        CONST dtf_int *zonenum, /* i: zone number */
                        dtf_int *n_cells_of_type /* o: number of cells for each master cell type */
                        );

/* given cell number, return offset in the cell-to-node array */
dtf_int dtf_query_c2n_pos(/* offset of the cell in the cell-to-node connectivity array */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum, /* i: simulation number */
                          CONST dtf_int *zonenum, /* i: zone number */
                          CONST dtf_int *cellnum /* i: cell number */
                          );

/* given cell number, return cell type*/
dtf_int dtf_query_celltype(/* return cell type */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *zonenum, /* i: zone number */
                           CONST dtf_int *cellnum /* i: cell number */
                           );

/***********************************************************************/
/* GRID CONNECTIVITY                                                   */
/***********************************************************************/

/* Node->Cell data: query for sizes */
dtf_int dtf_query_n2c(/* return size of node->cell array for node(s) */
                      CONST dtf_handle *fh, /* i: file handle */
                      CONST dtf_int *simnum, /* i: simulation number */
                      CONST dtf_int *zonenum, /* i: zone number */
                      CONST dtf_int *nodenum, /* i: node number (<1 mean all nodes) */
                      dtf_int *n_cells_per_node /* o: number of cells per node(s) */
                      );

/* Node->Cell data: read connectivity array */
dtf_int dtf_read_n2c(/* error code */
                     CONST dtf_handle *fh, /* i: file handle */
                     CONST dtf_int *simnum, /* i: simulation number */
                     CONST dtf_int *zonenum, /* i: zone number */
                     CONST dtf_int *nodenum, /* i: node number (<1 mean all nodes) */
                     dtf_int *n2c /* o: node->cell connectivity array */
                     );

/* Face->Node data: query for sizes */
dtf_int dtf_query_f2n(/* return size of Face->Node array for face(s) */
                      CONST dtf_handle *fh, /* i: file handle */
                      CONST dtf_int *simnum, /* i: simulation number */
                      CONST dtf_int *zonenum, /* i: zone number */
                      CONST dtf_int *facenum, /* i: face number (< 1 means all faces) */
                      dtf_int *n_nodes_per_face /* o: number of nodes per face(s) */
                      );

/* Face->Node data: read connectivity array */
dtf_int dtf_read_f2n(/* error code */
                     CONST dtf_handle *fh, /* i: file handle */
                     CONST dtf_int *simnum, /* i: simulation number */
                     CONST dtf_int *zonenum, /* i: zone number */
                     CONST dtf_int *facenum, /* i: face number (< 1 means all faces) */
                     dtf_int *f2n /* o: face->node connectivity array */
                     );

/* Face->Cell data: query for sizes */
dtf_int dtf_query_f2c(/* return size of face->cell array */
                      CONST dtf_handle *fh, /* i: file handle */
                      CONST dtf_int *simnum, /* i: simulation number */
                      CONST dtf_int *zonenum, /* i: zone number */
                      CONST dtf_int *facenum /* i: face number (< 1 means all faces) */
                      );

/* Face->Cell data: read connectivity array */
dtf_int dtf_read_f2c(/* error code */
                     CONST dtf_handle *fh, /* i: file handle */
                     CONST dtf_int *simnum, /* i: simulation number */
                     CONST dtf_int *zonenum, /* i: zone number */
                     CONST dtf_int *facenum, /* i: face number (< 1 mean all faces) */
                     dtf_int *f2c /* o: face->cell connectivity array */
                     );

/* Cell->Node data: query for sizes */
dtf_int dtf_query_c2n(/* return size of cell->node array for cell(s) */
                      CONST dtf_handle *fh, /* i: file handle */
                      CONST dtf_int *simnum, /* i: simulation number */
                      CONST dtf_int *zonenum, /* i: zone number */
                      CONST dtf_int *cellnum, /* i: cell number (< 1 means all cells) */
                      dtf_int *n_nodes_per_cell /* o: number of nodes per cell(s) */
                      );

/* Cell->Node data: read connectivity array */
dtf_int dtf_read_c2n(/* error code */
                     CONST dtf_handle *fh, /* i: file handle */
                     CONST dtf_int *simnum, /* i: simulation number */
                     CONST dtf_int *zonenum, /* i: zone number */
                     CONST dtf_int *cellnum, /* i: cell number (< 1 means all cells) */
                     dtf_int *c2n /* o: cell->node connectivity array */
                     );

/* Cell->Face data: query for sizes */
dtf_int dtf_query_c2f(/* return size of cell->face array for cell(s) */
                      CONST dtf_handle *fh, /* i: file handle */
                      CONST dtf_int *simnum, /* i: simulation number */
                      CONST dtf_int *zonenum, /* i: zone number */
                      CONST dtf_int *cellnum, /* i: cell number (< 1 means all cells) */
                      dtf_int *n_faces_per_cell /* o: number of faces per cell(s) */
                      );

/* Cell->Face data: read connectivity array */
dtf_int dtf_read_c2f(/* error code */
                     CONST dtf_handle *fh, /* i: file handle */
                     CONST dtf_int *simnum, /* i: simulation number */
                     CONST dtf_int *zonenum, /* i: zone number */
                     CONST dtf_int *cellnum, /* i: cell number (< 1 means all cells) */
                     dtf_int *c2f /* o: cell->face connectivity array */
                     );

/***********************************************************************/
/* VIRTUAL ZONE API */
/***********************************************************************/

/* given zone number, return an array of corresponding virtual node numbers for this zone */
dtf_int dtf_read_virtual_nodenums(/* error code */
                                  CONST dtf_handle *fh, /* i: file handle */
                                  CONST dtf_int *simnum, /* i: simulation number */
                                  CONST dtf_int *zonenum, /* i: zone number */
                                  dtf_int *v_nodenums /* o: an array of virtual zone node numbers */
                                  );

/* given zone number, return an array of corresponding virtual node numbers for this zone  ONLY FOR STRUCT ZONES!!!*/
dtf_int dtf_read_struct_zone_virtual_nodenums(/* error code */
                                              CONST dtf_handle *fh, /* i: file handle */
                                              CONST dtf_int *simnum, /* i: simulation number */
                                              CONST dtf_int *zonenum, /* i: zone number */
                                              dtf_int *v_nodenums /* o: an array of virtual zone node numbers */
                                              );

/* given zone number, return an array of corresponding virtual face numbers for this zone. */
dtf_int dtf_read_virtual_facenums(/* error code */
                                  CONST dtf_handle *fh, /* i: file handle */
                                  CONST dtf_int *simnum, /* i: simulation number */
                                  CONST dtf_int *zonenum, /* i: zone number */
                                  dtf_int *v_facenums /* o: an array of virtual zone face numbers */
                                  );

/* given zone number, return an array of corresponding virtual cell numbers for this zone. */
dtf_int dtf_read_virtual_cellnums(/* error code */
                                  CONST dtf_handle *fh, /* i: file handle */
                                  CONST dtf_int *simnum, /* i: simulation number */
                                  CONST dtf_int *zonenum, /* i: zone number */
                                  dtf_int *v_cellnums /* o: an array of virtual zone cell numbers */
                                  );

/* get number of data arrays attached to virtual zone with the given topological type */
dtf_int dtf_query_nvzds_of_topotype(/* number of arrays with the given topotype */
                                    CONST dtf_handle *fh, /* i: file handle */
                                    CONST dtf_int *simnum, /* i: simulation number */
                                    CONST dtf_topotype *topotype /* i: topological type */
                                    );

/* get number of data arrays attached to virtual zone */
dtf_int dtf_query_nvzds(/* number of arrays with the given topotype */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum /* i: simulation number */
                        );

/* Given a topotype, fill an array of vzd datanums */
dtf_int dtf_read_vzdnums_of_topotype(/* number of arrays with the given topotype */
                                     CONST dtf_handle *fh, /* i: file handle */
                                     CONST dtf_int *simnum, /* i: simulation number */
                                     CONST dtf_topotype *topotype, /* i: topological type */
                                     dtf_int *vzdnums /* o: array of vzd nums */
                                     );

/* Given a vzd data num, return info about it */
dtf_int dtf_query_vzd_by_num(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *num, /* i: instance # of the vzd */
                             dtf_string name, /* o: name of data */
                             dtf_int *n, /* o: number of elements in the data array */
                             dtf_datatype *datatype, /* o: datatype of the elements */
                             dtf_units units, /* o: units of the data array */
                             dtf_topotype *topotype /* o: topological type of the data array */
                             );

/* Given a vzd data name, return info about it */
dtf_int dtf_query_vzd_by_name(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_string name, /* i: name of the data array */
                              dtf_int *n, /* o: number of elements in the data array */
                              dtf_datatype *datatype, /* o: datatype of the elements */
                              dtf_units units, /* o: units of the data array */
                              dtf_topotype *topotype /* o: topological type of the data array */
                              );

/* Given a vzd data num, read it in */
dtf_int dtf_read_vzd_by_num(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *num, /* i: instance # of the data array in the virtual zone */
                            void *data, /* o: elements of the data array */
                            CONST dtf_datatype *datatype /* i: datatype of the elements */
                            );

/* Given a vzd data name, read it in */
dtf_int dtf_read_vzd_by_name(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_string name, /* i: name of the data array */
                             void *data, /* o: elements of the data array */
                             CONST dtf_datatype *datatype /* i: datatype of the elements */
                             );

/* Update the given vzd given by num with the data */
dtf_int dtf_update_vzd_by_num(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *num, /* i: instance # of zd in virutal zone */
                              CONST void *data, /* i: data array itself */
                              CONST dtf_datatype *datatype /* i: datatype of the array elements */
                              );

/* Update the given vzd given by name with the data */
dtf_int dtf_update_vzd_by_name(/* error code */
                               CONST dtf_handle *fh, /* i: file handle */
                               CONST dtf_int *simnum, /* i: simulation number */
                               CONST dtf_string name, /* i: old name of the data array */
                               CONST void *data, /* i: data array itself */
                               CONST dtf_datatype *datatype /* i: datatype of the array elements */
                               );

/* Given the bcrecord_num in a zone, determine it's number in the virtual zone */
dtf_int dtf_query_vz_bcrec_num(/* VZ BC Record Number */
                               dtf_handle *fh, /* i: file handle */
                               dtf_int *simnum, /* i: simulation number */
                               dtf_int *zonenum, /* i: zone number */
                               dtf_int *bcrec_num /* i: BC Record number */
                               );

/***********************************************************************/
/* BC CONNECTIVITY                                                     */
/***********************************************************************/

/* Query number of boundary faces */
dtf_int dtf_query_bf2bcr(/* Return number of boundary faces */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         dtf_int *nbfaces_of_type /* o: Number of boundary faces of each type */
                         );

/* Read all boundary face connectivity data */
dtf_int dtf_read_bf2bcr(/* error code */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum, /* i: simulation number */
                        CONST dtf_int *zonenum, /* i: zone number */
                        CONST dtf_int *facenum, /* i: boundary face number (< 1 means all boundary faces) */
                        dtf_int *bf2f, /* o: Boundary face to global face connectivity */
                        dtf_int *bf2r /* o: Boundary face to BC Record number connectivity */
                        );

/* Read boundary face conn
   ectivity data in the form of face->node. */
dtf_int dtf_read_bf2nbcr(/* error code */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         CONST dtf_int *facenum, /* i: boundary face number (< 1 means all boundary faces) */
                         dtf_int *bf2n, /* o: Boundary face to node connectivity */
                         dtf_int *bf2r /* o: Boundary face to BC Record number connectivity */
                         );

/* Update all boundary face connectivity data */
dtf_int dtf_update_bf2bcr(/* error code */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum, /* i: simulation number */
                          CONST dtf_int *zonenum, /* i: zone number */
                          CONST dtf_int *nboundary_faces, /* i: Number of boundary faces */
                          CONST dtf_int *bf2f, /* i: Boundary face to global face connectivity */
                          CONST dtf_int *bf2r /* i: Boundary face to BC Record number connectivity */
                          );

/* Update all boundary face connectivity data using face-to-node connectivity */
dtf_int dtf_update_bf2n_bf2bcr(/* error code */
                               CONST dtf_handle *fh, /* i: file handle */
                               CONST dtf_int *simnum, /* i: simulation number */
                               CONST dtf_int *zonenum, /* i: zone number */
                               CONST dtf_int *nboundary_faces, /* i: Number of boundary faces of each type */
                               CONST dtf_int *bf2n, /* i: Boundary face to global face connectivity */
                               CONST dtf_int *bf2r /* i: Boundary face to BC Record number connectivity */
                               );

/* Query number of interface faces */
dtf_int dtf_query_xf2bcr(/* Return number of interface faces */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         dtf_int *nxfaces_of_type /* o: Number of interface faces of each type */
                         );

/* Read all interface face connectivity data */
dtf_int dtf_read_xf2bcr(/* error code */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum, /* i: simulation number */
                        CONST dtf_int *zonenum, /* i: zone number */
                        CONST dtf_int *facenum, /* i: interface face number (< 1 means all interface faces) */
                        dtf_int *xf2f, /* o: Interface face to global face connectivity */
                        dtf_int *xf2r /* o: Interface face to BC Record number connectivity */
                        );

/* Read all interface face connectivity data in the form of face->node. */
dtf_int dtf_read_xf2nbcr(/* error code */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         CONST dtf_int *facenum, /* i: interace face number (< 1 means all interface faces) */
                         dtf_int *xf2n, /* o: Interface face to node connectivity */
                         dtf_int *xf2r /* o: Interface face to BC Record number connectivity */
                         );

/* Update all interface face connectivity data */
dtf_int dtf_update_xf2bcr(/* error code */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum, /* i: simulation number */
                          CONST dtf_int *zonenum, /* i: zone number */
                          CONST dtf_int *xface_faces, /* i: Number of interface faces */
                          CONST dtf_int *xf2f, /* i: Interface face to global face connectivity */
                          CONST dtf_int *xf2r /* i: Interface face to BC Record number connectivity */
                          );

/* Update all interface face connectivity data using face-to-node connectivity */
dtf_int dtf_update_xf2n_xf2bcr(/* error code */
                               CONST dtf_handle *fh, /* i: file handle */
                               CONST dtf_int *simnum, /* i: simulation number */
                               CONST dtf_int *zonenum, /* i: zone number */
                               CONST dtf_int *nboundary_faces, /* i: number of interface faces of each type */
                               CONST dtf_int *xf2n, /* i: interface face to global face connectivity */
                               CONST dtf_int *xf2r /* i: interface face to BC record number connectivity */
                               );

/***********************************************************************/
/* BOUNDARY AND INTERFACE FACES                                        */
/***********************************************************************/

/* find the index in the boundary record_id/face_id array corresponding to the global face index */
dtf_int dtf_query_bfnum_by_fnum(/* index of the face in the record_id/bface_id array*/
                                CONST dtf_handle *fh, /* i: file handle */
                                CONST dtf_int *simnum, /* i: simulation number */
                                CONST dtf_int *zonenum, /* i: zone number */
                                CONST dtf_int *global_face_index /* i: index of the face in the global face array */
                                );

/* find the index in the interface record_id/face_id array corresponding to the global face index */
dtf_int dtf_query_ifnum_by_fnum(/* index of the face in the record_id/face_id array*/
                                CONST dtf_handle *fh, /* i: file handle */
                                CONST dtf_int *simnum, /* i: simulation number */
                                CONST dtf_int *zonenum, /* i: zone number */
                                CONST dtf_int *global_face_index /* i: index of the face in the global face array */
                                );

/***********************************************************************/
/* DATA READING/WRITING/MANIPULATING                                   */
/***********************************************************************/

/* add a data array to a zone */
dtf_int dtf_add_zd(/* number of the data array added */
                   CONST dtf_handle *fh, /* i: file handle */
                   CONST dtf_int *simnum, /* i: simulation number */
                   CONST dtf_int *zonenum, /* i: zone number */
                   CONST dtf_string name, /* i: name of the data array */
                   CONST dtf_int *n, /* i: number of elements in the data array */
                   CONST void *data, /* i: data array itself */
                   CONST dtf_datatype *datatype, /* i: datatype of the array elements */
                   CONST dtf_units units, /* i: units of the data array */
                   CONST dtf_topotype *topotype /* i: topological type of the data array */
                   );

/* delete a data array attached to a zone */
dtf_int dtf_delete_zd_by_num(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum, /* i: zone number */
                             CONST dtf_int *datanum /* i: number of the data array */
                             );

dtf_int dtf_delete_zd_by_name(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_string name /* i: name of the data array */
                              );

/* update a zonal data array by datanum */
dtf_int dtf_update_zd_by_num(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum, /* i: zone number */
                             CONST dtf_int *datanum, /* i: number of the data array */
                             CONST dtf_string name, /* i: new name of the data array */
                             CONST dtf_int *n, /* i: number of elements in the data array */
                             CONST void *data, /* i: data array itself */
                             CONST dtf_datatype *datatype, /* i: datatype of the array elements */
                             CONST dtf_units units, /* i: units of the data array */
                             CONST dtf_topotype *topotype /* i: topological type of the data array */
                             );

/* update a zonal data array of name=name */
dtf_int dtf_update_zd_by_name(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_string name, /* i: old name of the data array */
                              CONST dtf_string newname, /* i: new name of the data array */
                              CONST dtf_int *n, /* i: number of elements in the data array */
                              CONST void *data, /* i: data array itself */
                              CONST dtf_datatype *datatype, /* i: datatype of the array elements */
                              CONST dtf_units units, /* i: units of the data array */
                              CONST dtf_topotype *topotype /* i: topological type of the data array */
                              );

/* get number of data arrays attached to a zone */
dtf_int dtf_query_nzds(/* number of data arrays atached to the zone */
                       CONST dtf_handle *fh, /* i: file handle */
                       CONST dtf_int *simnum, /* i: simulation number */
                       CONST dtf_int *zonenum /* i: zone number */
                       );

/* get number of data arrays attached to a zone with the given topological type */
dtf_int dtf_query_nzds_of_topotype(/* number of arrays with the given topotype */
                                   CONST dtf_handle *fh, /* i: file handle */
                                   CONST dtf_int *simnum, /* i: simulation number */
                                   CONST dtf_int *zonenum, /* i: zone number */
                                   CONST dtf_topotype *topotype /* i: topological type */
                                   );

/* read an array of numbers of data arrays with given topotype */
dtf_int dtf_read_zdnums_of_topotype(/* error code */
                                    CONST dtf_handle *fh, /* i: file handle */
                                    CONST dtf_int *simnum, /* i: simulation number */
                                    CONST dtf_int *zonenum, /* i: zone number */
                                    CONST dtf_topotype *topotype, /* i: topological type */
                                    dtf_int *nums /* o: array of data array numbers */
                                    );

/* get info about a data array attached to a zone */
dtf_int dtf_query_zd_by_num(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *zonenum, /* i: zone number */
                            CONST dtf_int *datanum, /* i: number of the data array */
                            dtf_string name, /* o: name of the data array */
                            dtf_int *n, /* o: number of elements in the data array */
                            dtf_datatype *datatype, /* o: datatype of the elements */
                            dtf_units units, /* o: units of the data array */
                            dtf_topotype *topotype /* o: topological type of the data array */
                            );

dtf_int dtf_query_zd_by_name(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum, /* i: zone number */
                             CONST dtf_string name, /* i: name of the data array */
                             dtf_int *n, /* o: number of elements in the data array */
                             dtf_datatype *datatype, /* o: datatype of the elements */
                             dtf_units units, /* o: units of the data array */
                             dtf_topotype *topotype /* o: topological type of the data array */
                             );

/* read data array attached to a zone */
dtf_int dtf_read_zd_by_num(/* error code */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *zonenum, /* i: zone number */
                           CONST dtf_int *datanum, /* i: number of the data array */
                           CONST dtf_int *element_num, /* i: Element of this array to read: element_num<=0 to read entire array */
                           void *data, /* o: elements of the data array */
                           CONST dtf_datatype *datatype /* i: datatype of the elements */
                           );

dtf_int dtf_read_zd_by_name(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *zonenum, /* i: zone number */
                            CONST dtf_string name, /* i: name of the data array */
                            CONST dtf_int *element_num, /* i: Element of this array to read: element_num<=0 to read entire array */
                            void *data, /* o: elements of the data array */
                            CONST dtf_datatype *datatype /* i: datatype of the elements */
                            );

/* get minmax array for data field */
dtf_int dtf_query_zd_minmax_by_name(/* error code */
                                    CONST dtf_handle *fh, /* i: file handle */
                                    CONST dtf_int *simnum, /* i: simulation number */
                                    CONST dtf_int *zonenum, /* i: zone number */
                                    CONST dtf_string name, /* i: name of the data array */
                                    void *data, /* o: minmax of the data array */
                                    CONST dtf_datatype *datatype /* i: datatype of the elements */
                                    );

/* Delete all zd of given topotype */
dtf_int dtf_delete_all_zd_of_topotype(
    CONST dtf_handle *fh, /* i: file handle */
    CONST dtf_int *simnum, /* i: simulation number */
    CONST dtf_int *zonenum, /* i: zone number */
    CONST dtf_topotype *topotype /* i: topological type of the data array */
    );

/* add a data array to a simulation */
dtf_int dtf_add_sd(/* number of the data array added */
                   CONST dtf_handle *fh, /* i: file handle */
                   CONST dtf_int *simnum, /* i: simulation number */
                   CONST dtf_string name, /* i: name of the data array */
                   CONST dtf_int *n, /* i: number of elements in the data array */
                   CONST void *data, /* i: data array itself */
                   CONST dtf_datatype *datatype, /* i: datatype of the array elements */
                   CONST dtf_units units, /* i: units of the data array */
                   CONST dtf_topotype *topotype /* i: topological type of the data array */
                   );

/* delete a data array attached to a simulation */
dtf_int dtf_delete_sd_by_num(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *datanum /* i: number of the data array */
                             );

dtf_int dtf_delete_sd_by_name(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_string name /* i: name of the data array */
                              );

/* Update simulation data by datanum */
dtf_int dtf_update_sd_by_num(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *datanum, /* i: number of the data array */
                             CONST dtf_string name, /* i: new name of the data array */
                             CONST dtf_int *n, /* i: number of elements in the data array */
                             CONST void *data, /* i: data array itself */
                             CONST dtf_datatype *datatype, /* i: datatype of the array elements */
                             CONST dtf_units units, /* i: units of the data array */
                             CONST dtf_topotype *topotype /* i: topological type of the data array */
                             );

/* Update simulation data by name */
dtf_int dtf_update_sd_by_name(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_string name, /* i: old name of the data array */
                              CONST dtf_string newname, /* i: new name of the data array */
                              CONST dtf_int *n, /* i: number of elements in the data array */
                              CONST void *data, /* i: data array itself */
                              CONST dtf_datatype *datatype, /* i: datatype of the array elements */
                              CONST dtf_units units, /* i: units of the data array */
                              CONST dtf_topotype *topotype /* i: topological type of the data array */
                              );

/* get number of data arrays attached to a simulation */
dtf_int dtf_query_nsds(/* number of data arrays atached to the simulation */
                       CONST dtf_handle *fh, /* i: file handle */
                       CONST dtf_int *simnum /* i: simulation number */
                       );

/* get number of data arrays attached to a simulation with the given topological type */
dtf_int dtf_query_nsds_of_topotype(/* number of arrays with the given topotype */
                                   CONST dtf_handle *fh, /* i: file handle */
                                   CONST dtf_int *simnum, /* i: simulation number */
                                   CONST dtf_topotype *topotype /* i: topological type */
                                   );

/* read an array of numbers of data arrays with given topotype */
dtf_int dtf_read_sdnums_of_topotype(/* error code */
                                    CONST dtf_handle *fh, /* i: file handle */
                                    CONST dtf_int *simnum, /* i: simulation number */
                                    CONST dtf_topotype *topotype, /* i: topological type */
                                    dtf_int *nums /* o: array of data array numbers */
                                    );

/* get info about a data array attached to a simulation */
dtf_int dtf_query_sd_by_num(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *datanum, /* i: number of the data array */
                            dtf_string name, /* o: name of the data array */
                            dtf_int *n, /* o: number of elements in the data array */
                            dtf_datatype *datatype, /* o: datatype of the elements */
                            dtf_units units, /* o: units of the data array */
                            dtf_topotype *topotype /* o: topological type of the data array */
                            );

dtf_int dtf_query_sd_by_name(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_string name, /* i: name of the data array */
                             dtf_int *n, /* o: number of elements in the data array */
                             dtf_datatype *datatype, /* o: datatype of the elements */
                             dtf_units units, /* o: units of the data array */
                             dtf_topotype *topotype /* o: topological type of the data array */
                             );

/* read data array attached to a simulation */
dtf_int dtf_read_sd_by_num(/* error code */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *datanum, /* i: number of the data array */
                           CONST dtf_int *element_num, /* i: Element of this array to read: element_num<=0 to read entire array */
                           void *data, /* o: elements of the data array */
                           CONST dtf_datatype *datatype /* i: datatype of the elements */
                           );
dtf_int dtf_read_sd_by_name(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_string name, /* i: name of the data array */
                            CONST dtf_int *element_num, /* i: Element of this array to read: element_num<=0 to read entire array */
                            void *data, /* o: elements of the data array */
                            CONST dtf_datatype *datatype /* i: datatype of the elements */
                            );

/* get minmax array for data field */
dtf_int dtf_query_sd_minmax_by_name(/* error code */
                                    CONST dtf_handle *fh, /* i: file handle */
                                    CONST dtf_int *simnum, /* i: simulation number */
                                    CONST dtf_string name, /* i: name of the data array */
                                    void *data, /* o: minmax of the data array */
                                    CONST dtf_datatype *datatype /* i: datatype of the elements */
                                    );

/***********************************************************************/
/* WORKING WITH BOUNDARY CONDITION RECORDS                             */
/***********************************************************************/

/* get number of boundary condition records attached to the zone */
dtf_int dtf_query_nbcrecords(/* number of boundary conditions records attached to the zone */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum /* i: zone number */
                             );

/* get information about a BC record given its number */
dtf_int dtf_query_bcrecord(/* error code */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *zonenum, /* i: zone number */
                           CONST dtf_int *bcnum, /* i: record number */
                           dtf_int *key, /* o: key */
                           dtf_string type, /* o: type */
                           dtf_string name, /* o: name */
                           dtf_int *n_categories, /* o: number of categories */
                           dtf_int *n_bcvals /* o: number of BC values */
                           );

/* add a new BC record or update key, type and name of an existing record */
dtf_int dtf_update_bcrecord(/* actual number of the record added or updated */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *zonenum, /* i: zone number */
                            CONST dtf_int *bcnum, /* i: record number */
                            CONST dtf_int *key, /* i: key */
                            CONST dtf_string type, /* i: type */
                            CONST dtf_string name /* i: name */
                            );

/* Copy the bcrecord in one (sim,zone) to another (sim,zone) (not a link) */
dtf_int dtf_copy_bcrecord(/* Number of bcrecord copied (=nbcrecords if copy all records) */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum_from, /* i: simulation number */
                          CONST dtf_int *zonenum_from, /* i: zone number */
                          CONST dtf_int *bcnum, /* i: record number (overloaded: use (-1) to copy all) */
                          CONST dtf_int *simnum_to, /* i: simulation number */
                          CONST dtf_int *zonenum_to /* i: zone number */
                          );

/* delete a BC record */
dtf_int dtf_delete_bcrecord(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *zonenum, /* i: zone number */
                            CONST dtf_int *bcnum /* i: record number */
                            );

/* get category name and value by its number */
dtf_int dtf_query_bc_category(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *bcnum, /* i: record number */
                              CONST dtf_int *catnum, /* i: category number */
                              dtf_string name, /* o: category name */
                              dtf_string value /* o: category value */
                              );

/* get category value by its name */
dtf_int dtf_query_bc_category_value(/* error code */
                                    CONST dtf_handle *fh, /* i: file handle */
                                    CONST dtf_int *simnum, /* i: simulation number */
                                    CONST dtf_int *zonenum, /* i: zone number */
                                    CONST dtf_int *bcnum, /* i: record number */
                                    CONST dtf_string name, /* i: category name */
                                    dtf_string value /* o: category value */
                                    );

/* update all category names and values in one shot */
dtf_int dtf_update_bc_categories(/* error code */
                                 CONST dtf_handle *fh, /* i: file handle */
                                 CONST dtf_int *simnum, /* i: simulation number */
                                 CONST dtf_int *zonenum, /* i: zone number */
                                 CONST dtf_int *bcnum, /* i: record number */
                                 CONST dtf_int *n_categories, /* i: number of categories */
                                 CONST PDTFSTRING name, /* i: array of category names */
                                 CONST PDTFSTRING value /* i: array of category values */
                                 );

/* get names of BC values attached to a BC record */
dtf_int dtf_query_bcval_name(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum, /* i: zone number */
                             CONST dtf_int *bcnum, /* i: record number */
                             CONST dtf_int *bcvalnum, /* i: BC value number */
                             dtf_string name /* o: BC value name */
                             );

/* get evaluation method of a BC value */
dtf_int dtf_query_bcval_eval_method(/* error code */
                                    CONST dtf_handle *fh, /* i: file handle */
                                    CONST dtf_int *simnum, /* i: simulation number */
                                    CONST dtf_int *zonenum, /* i: zone number */
                                    CONST dtf_int *bcnum, /* i: record number */
                                    CONST dtf_string name, /* i: BC value name */
                                    dtf_string eval_method /* o: evaluation method */
                                    );

/* get number of evaluation data of a BC value */
dtf_int dtf_query_bcval_eval_data(/* error code */
                                  CONST dtf_handle *fh, /* i: file handle */
                                  CONST dtf_int *simnum, /* i: simulation number */
                                  CONST dtf_int *zonenum, /* i: zone number */
                                  CONST dtf_int *bcnum, /* i: record number */
                                  CONST dtf_string name, /* i: BC value name */
                                  dtf_int *nints, /* o: number of integers */
                                  dtf_int *nreals, /* o: number of reals */
                                  dtf_int *nstrings /* o: number of strings */
                                  );

/* read all evaluation data of a BC value */
dtf_int dtf_read_bcval_eval_data_d(/* error code */
                                   CONST dtf_handle *fh, /* i: file handle */
                                   CONST dtf_int *simnum, /* i: simulation number */
                                   CONST dtf_int *zonenum, /* i: zone number */
                                   CONST dtf_int *bcnum, /* i: record number */
                                   CONST dtf_string name, /* i: BC value name */
                                   dtf_string *var_ints, /* o: array of integer names */
                                   dtf_int *ints, /* o: array of integers */
                                   dtf_string *var_reals, /* o: array of real names */
                                   dtf_double *reals, /* o: array of reals */
                                   dtf_string *var_strings, /* o: array of string names */
                                   dtf_string *strings /* o: array of strings */
                                   );

dtf_int dtf_read_bcval_eval_data_s(/* error code */
                                   CONST dtf_handle *fh, /* i: file handle */
                                   CONST dtf_int *simnum, /* i: simulation number */
                                   CONST dtf_int *zonenum, /* i: zone number */
                                   CONST dtf_int *bcnum, /* i: record number */
                                   CONST dtf_string name, /* i: BC value name */
                                   dtf_string *var_ints, /* o: array of integer names */
                                   dtf_int *ints, /* o: array of integers */
                                   dtf_string *var_reals, /* o: array of real names */
                                   dtf_single *reals, /* o: array of reals */
                                   dtf_string *var_strings, /* o: array of string names */
                                   dtf_string *strings /* o: array of strings */
                                   );

/* given its name, read an integer from evaluation data of a BC value */
dtf_int dtf_read_bcval_int(/* error code */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *zonenum, /* i: zone number */
                           CONST dtf_int *bcnum, /* i: record number */
                           CONST dtf_string name, /* i: BC value name */
                           CONST dtf_string int_name, /* i: integer name */
                           dtf_int *int_value /* o: integer value */
                           );

/* given its name, read a real from evaluation data of a BC value */
dtf_int dtf_read_bcval_real_d(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *bcnum, /* i: record number */
                              CONST dtf_string name, /* i: BC value name */
                              CONST dtf_string real_name, /* i: real name */
                              dtf_double *real_value /* o: real value */
                              );

dtf_int dtf_read_bcval_real_s(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *bcnum, /* i: record number */
                              CONST dtf_string name, /* i: BC value name */
                              CONST dtf_string real_name, /* i: real name */
                              dtf_single *real_value /* o: real value */
                              );

/* given its name, read a string from evaluation data of a BC value */
dtf_int dtf_read_bcval_string(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *bcnum, /* i: record number */
                              CONST dtf_string name, /* i: BC value name */
                              CONST dtf_string string_name, /* i: string name */
                              dtf_string string_value /* o: string value */
                              );

/* add a new BC value one or update name and evaluation method of an existing one */
dtf_int dtf_update_bcval(/* error code */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         CONST dtf_int *bcnum, /* i: record number */
                         CONST dtf_string name, /* i: BC value name */
                         CONST dtf_string eval_method /* i: BC value evaluation method */
                         );

/* update integer evaluation data of a BC value */
dtf_int dtf_update_bcval_ints(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *bcnum, /* i: record number */
                              CONST dtf_string name, /* i: BC value name */
                              CONST dtf_int *nints, /* i: number of integers */
                              CONST PDTFSTRING var_ints, /* i: array of integer names */
                              CONST dtf_int *ints /* i: array of integers */
                              );

/* update real evaluation data of a BC value */
dtf_int dtf_update_bcval_reals_d(/* error code */
                                 CONST dtf_handle *fh, /* i: file handle */
                                 CONST dtf_int *simnum, /* i: simulation number */
                                 CONST dtf_int *zonenum, /* i: zone number */
                                 CONST dtf_int *bcnum, /* i: record number */
                                 CONST dtf_string name, /* i: BC value name */
                                 CONST dtf_int *nreals, /* i: number of reals */
                                 CONST PDTFSTRING var_reals, /* i: array of real names */
                                 CONST dtf_double *reals /* i: array of reals */
                                 );

dtf_int dtf_update_bcval_reals_s(/* error code */
                                 CONST dtf_handle *fh, /* i: file handle */
                                 CONST dtf_int *simnum, /* i: simulation number */
                                 CONST dtf_int *zonenum, /* i: zone number */
                                 CONST dtf_int *bcnum, /* i: record number */
                                 CONST dtf_string name, /* i: BC value name */
                                 CONST dtf_int *nreals, /* i: number of reals */
                                 CONST PDTFSTRING var_reals, /* i: array of real names */
                                 CONST dtf_single *reals /* i: array of reals */
                                 );

/* update string evaluation data of a BC value */
dtf_int dtf_update_bcval_strings(/* error code */
                                 CONST dtf_handle *fh, /* i: file handle */
                                 CONST dtf_int *simnum, /* i: simulation number */
                                 CONST dtf_int *zonenum, /* i: zone number */
                                 CONST dtf_int *bcnum, /* i: record number */
                                 CONST dtf_string name, /* i: BC value name */
                                 CONST dtf_int *nstrings, /* i: number of strings */
                                 CONST PDTFSTRING var_strings, /* i: array of string names */
                                 CONST PDTFSTRING strings /* i: array of strings */
                                 );

/* update integer element of a BC value by name */
dtf_int dtf_update_bcval_int_by_name(/* error code */
                                     CONST dtf_handle *fh, /* i: file handle */
                                     CONST dtf_int *simnum, /* i: simulation number */
                                     CONST dtf_int *zonenum, /* i: zone number */
                                     CONST dtf_int *bcnum, /* i: record number */
                                     CONST dtf_string val_name, /* i: BC value name */
                                     CONST dtf_string elem_name, /* i: element name */
                                     CONST dtf_int *elem_value /* i: element value */
                                     );

/* update real (double) element of a BC value by name */
dtf_int dtf_update_bcval_real_d_by_name(/* error code */
                                        CONST dtf_handle *fh, /* i: file handle */
                                        CONST dtf_int *simnum, /* i: simulation number */
                                        CONST dtf_int *zonenum, /* i: zone number */
                                        CONST dtf_int *bcnum, /* i: record number */
                                        CONST dtf_string val_name, /* i: BC value name */
                                        CONST dtf_string elem_name, /* i: element name */
                                        CONST dtf_double *elem_value /* i: element value */
                                        );

/* update real (single) element of a BC value by name */
dtf_int dtf_update_bcval_real_s_by_name(/* error code */
                                        CONST dtf_handle *fh, /* i: file handle */
                                        CONST dtf_int *simnum, /* i: simulation number */
                                        CONST dtf_int *zonenum, /* i: zone number */
                                        CONST dtf_int *bcnum, /* i: record number */
                                        CONST dtf_string val_name, /* i: BC value name */
                                        CONST dtf_string elem_name, /* i: element name */
                                        CONST dtf_single *elem_value /* i: element value */
                                        );

/* update string element of a BC value by name */
dtf_int dtf_update_bcval_string_by_name(/* error code */
                                        CONST dtf_handle *fh, /* i: file handle */
                                        CONST dtf_int *simnum, /* i: simulation number */
                                        CONST dtf_int *zonenum, /* i: zone number */
                                        CONST dtf_int *bcnum, /* i: record number */
                                        CONST dtf_string val_name, /* i: BC value name */
                                        CONST dtf_string elem_name, /* i: element name */
                                        CONST dtf_string elem_value /* i: element value */
                                        );

/* delete a BC value */
dtf_int dtf_delete_bcval(/* error code */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         CONST dtf_int *bcnum, /* i: record number */
                         CONST dtf_string name /* i: BC value name */
                         );

/* delete all BC values */
dtf_int dtf_delete_all_bcvals(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *bcnum /* i: record number */
                              );

/***********************************************************************/
/* PATCHES                                                             */
/***********************************************************************/

/* get number of patches */
dtf_int dtf_query_npatches(/* number of patches attached to the zone */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *zonenum /* i: zone number */
                           );

/* get info about a patch */
dtf_int dtf_query_patch(/* error code */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum, /* i: simulation number */
                        CONST dtf_int *zonenum, /* i: zone number */
                        CONST dtf_int *patchnum, /* i: patch number */
                        dtf_int *imin, /* o: minimum index in first plane */
                        dtf_int *imax, /* o: maximum index in first plane */
                        dtf_int *jmin, /* o: minimum index in second plane */
                        dtf_int *jmax, /* o: maximum index in second plane */
                        dtf_int *kmin, /* o: minimum index in third plane */
                        dtf_int *kmax /* o: maximum index in third plane */
                        );

/* read array of records attached to this patch */
dtf_int dtf_read_patch(/* error code */
                       CONST dtf_handle *fh, /* i: file handle */
                       CONST dtf_int *simnum, /* i: simulation number */
                       CONST dtf_int *zonenum, /* i: zone number */
                       CONST dtf_int *patchnum, /* i: patch number */
                       dtf_int *records /* o: records attached to the plane */
                       );

/* modify existing patch or add a new one */
dtf_int dtf_update_patch(/* error code */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         CONST dtf_int *patchnum, /* i: patch number */
                         CONST dtf_int *imin, /* i: minimum index in first plane */
                         CONST dtf_int *imax, /* i: maximum index in first plane */
                         CONST dtf_int *jmin, /* i: minumum index in second plane */
                         CONST dtf_int *jmax, /* i: maximum index in second plane */
                         CONST dtf_int *kmin, /* i: minimum index in third plane */
                         CONST dtf_int *kmax, /* i: maximum index in third plane */
                         CONST dtf_int *records /* i: array of records attached to the patch */
                         );

/* delete patch */
dtf_int dtf_delete_patch(/* error code */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         CONST dtf_int *patchnum /* i: patch number */
                         );

/* get number of records in a patch */
dtf_int dtf_query_nrecords_in_patch(/* number of records for this patch */
                                    CONST dtf_int *imin, /* i: minimum index in first plane */
                                    CONST dtf_int *imax, /* i: maximum index in first plane */
                                    CONST dtf_int *jmin, /* i: minumum index in second plane */
                                    CONST dtf_int *jmax, /* i: maximum index in second plane */
                                    CONST dtf_int *kmin, /* i: minimum index in third plane */
                                    CONST dtf_int *kmax /* i: maximum index in third plane */
                                    );

/***********************************************************************/
/* FACE GROUPS                                                         */
/***********************************************************************/

/* get number of face_groups */
dtf_int dtf_query_nface_groups(/* number of face groups attached to the zone */
                               CONST dtf_handle *fh, /* i: file handle */
                               CONST dtf_int *simnum, /* i: simulation number */
                               CONST dtf_int *zonenum /* i: zone number */
                               );

/* get info about a face_group */
dtf_int dtf_query_face_group(/* number of faces in the group number */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum, /* i: zone number */
                             CONST dtf_int *face_groupnum, /* i: face group number */
                             dtf_int *key /* o: key */
                             );

/* read faces of the face_group */
dtf_int dtf_read_face_group(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *zonenum, /* i: zone number */
                            CONST dtf_int *face_groupnum, /* i: face group number */
                            dtf_int *faces /* o: array of face numbers */
                            );

/***********************************************************************/
/* SURFACE CONDITIONS                                                  */
/***********************************************************************/

/* get number of surface_conditions */
dtf_int dtf_query_nsurface_conditions(/* number of surface conditions attached to the zone */
                                      CONST dtf_handle *fh, /* i: file handle */
                                      CONST dtf_int *simnum, /* i: simulation number */
                                      CONST dtf_int *zonenum /* i: zone number */
                                      );

/* get info about a surface_condition */
dtf_int dtf_query_surface_condition(/* error code */
                                    CONST dtf_handle *fh, /* i: file handle */
                                    CONST dtf_int *simnum, /* i: simulation number */
                                    CONST dtf_int *zonenum, /* i: zone number */
                                    CONST dtf_int *surface_conditionnum, /* i: surface condition number */
                                    dtf_int *sc_group_num, /* o: surface condition group number */
                                    dtf_int *bc_record_num /* o: surface condition record number */
                                    );

/***********************************************************************/
/* WORKING WITH VOLUME CONDITION RECORDS                             */
/***********************************************************************/

/* get number of volume condition records attached to the zone */
dtf_int dtf_query_nvcrecords(/* number of volume conditions records attached to the zone */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum /* i: zone number */
                             );

/* get information about a VC record given its number */
dtf_int dtf_query_vcrecord(/* error code */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *zonenum, /* i: zone number */
                           CONST dtf_int *vcnum, /* i: record number */
                           dtf_string category, /* o: category */
                           dtf_string name, /* o: name */
                           dtf_int *n_vcvals /* o: number of VC values */
                           );

/* add a new VC record or update key, type and name of an existing record */
dtf_int dtf_update_vcrecord(/* actual number of the record added or updated */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *zonenum, /* i: zone number */
                            CONST dtf_int *vcnum, /* i: record number */
                            CONST dtf_string category, /* i: category */
                            CONST dtf_string name /* i: type */
                            );

/* Copy the vcrecord in one (sim,zone) to another (sim,zone) (not a link) */
dtf_int dtf_copy_vcrecord(/* Number of vcrecord copied (=nvcrecords if copy all records) */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum_from, /* i: simulation number */
                          CONST dtf_int *zonenum_from, /* i: zone number */
                          CONST dtf_int *vcnum, /* i: record number (overloaded: use (-1) to copy all) */
                          CONST dtf_int *simnum_to, /* i: simulation number */
                          CONST dtf_int *zonenum_to /* i: zone number */
                          );

/* delete a VC record */
dtf_int dtf_delete_vcrecord(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *zonenum, /* i: zone number */
                            CONST dtf_int *vcnum /* i: record number */
                            );

/* get names of VC values attached to a VC record */
dtf_int dtf_query_vcval_name(/* error code */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum, /* i: zone number */
                             CONST dtf_int *vcnum, /* i: record number */
                             CONST dtf_int *vcvalnum, /* i: VC value number */
                             dtf_string name /* o: VC value name */
                             );

/* get evaluation method of a VC value */
dtf_int dtf_query_vcval_eval_method(/* error code */
                                    CONST dtf_handle *fh, /* i: file handle */
                                    CONST dtf_int *simnum, /* i: simulation number */
                                    CONST dtf_int *zonenum, /* i: zone number */
                                    CONST dtf_int *vcnum, /* i: record number */
                                    CONST dtf_string name, /* i: VC value name */
                                    dtf_string eval_method /* o: evaluation method */
                                    );

/* get number of evaluation data of a VC value */
dtf_int dtf_query_vcval_eval_data(/* error code */
                                  CONST dtf_handle *fh, /* i: file handle */
                                  CONST dtf_int *simnum, /* i: simulation number */
                                  CONST dtf_int *zonenum, /* i: zone number */
                                  CONST dtf_int *vcnum, /* i: record number */
                                  CONST dtf_string name, /* i: VC value name */
                                  dtf_int *nints, /* o: number of integers */
                                  dtf_int *nreals, /* o: number of reals */
                                  dtf_int *nstrings /* o: number of strings */
                                  );

/* read all evaluation data of a VC value */
dtf_int dtf_read_vcval_eval_data_d(/* error code */
                                   CONST dtf_handle *fh, /* i: file handle */
                                   CONST dtf_int *simnum, /* i: simulation number */
                                   CONST dtf_int *zonenum, /* i: zone number */
                                   CONST dtf_int *vcnum, /* i: record number */
                                   CONST dtf_string name, /* i: VC value name */
                                   dtf_string *var_ints, /* o: array of integer names */
                                   dtf_int *ints, /* o: array of integers */
                                   dtf_string *var_reals, /* o: array of real names */
                                   dtf_double *reals, /* o: array of reals */
                                   dtf_string *var_strings, /* o: array of string names */
                                   dtf_string *strings /* o: array of strings */
                                   );

dtf_int dtf_read_vcval_eval_data_s(/* error code */
                                   CONST dtf_handle *fh, /* i: file handle */
                                   CONST dtf_int *simnum, /* i: simulation number */
                                   CONST dtf_int *zonenum, /* i: zone number */
                                   CONST dtf_int *vcnum, /* i: record number */
                                   CONST dtf_string name, /* i: VC value name */
                                   dtf_string *var_ints, /* o: array of integer names */
                                   dtf_int *ints, /* o: array of integers */
                                   dtf_string *var_reals, /* o: array of real names */
                                   dtf_single *reals, /* o: array of reals */
                                   dtf_string *var_strings, /* o: array of string names */
                                   dtf_string *strings /* o: array of strings */
                                   );

/* given its name, read an integer from evaluation data of a VC value */
dtf_int dtf_read_vcval_int(/* error code */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *zonenum, /* i: zone number */
                           CONST dtf_int *vcnum, /* i: record number */
                           CONST dtf_string name, /* i: VC value name */
                           CONST dtf_string int_name, /* i: integer name */
                           dtf_int *int_value /* o: integer value */
                           );

/* given its name, read a real from evaluation data of a VC value */
dtf_int dtf_read_vcval_real_d(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *vcnum, /* i: record number */
                              CONST dtf_string name, /* i: VC value name */
                              CONST dtf_string real_name, /* i: real name */
                              dtf_double *real_value /* o: real value */
                              );

dtf_int dtf_read_vcval_real_s(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *vcnum, /* i: record number */
                              CONST dtf_string name, /* i: VC value name */
                              CONST dtf_string real_name, /* i: real name */
                              dtf_single *real_value /* o: real value */
                              );

/* given its name, read a string from evaluation data of a VC value */
dtf_int dtf_read_vcval_string(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *vcnum, /* i: record number */
                              CONST dtf_string name, /* i: VC value name */
                              CONST dtf_string string_name, /* i: string name */
                              dtf_string string_value /* o: string value */
                              );

/* add a new VC value one or update name and evaluation method of an existing one */
dtf_int dtf_update_vcval(/* error code */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         CONST dtf_int *vcnum, /* i: record number */
                         CONST dtf_string name, /* i: VC value name */
                         CONST dtf_string eval_method /* i: VC value evaluation method */
                         );

/* update integer evaluation data of a VC value */
dtf_int dtf_update_vcval_ints(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *vcnum, /* i: record number */
                              CONST dtf_string name, /* i: VC value name */
                              CONST dtf_int *nints, /* i: number of integers */
                              CONST PDTFSTRING var_ints, /* i: array of integer names */
                              CONST dtf_int *ints /* i: array of integers */
                              );

/* update real evaluation data of a VC value */
dtf_int dtf_update_vcval_reals_d(/* error code */
                                 CONST dtf_handle *fh, /* i: file handle */
                                 CONST dtf_int *simnum, /* i: simulation number */
                                 CONST dtf_int *zonenum, /* i: zone number */
                                 CONST dtf_int *vcnum, /* i: record number */
                                 CONST dtf_string name, /* i: VC value name */
                                 CONST dtf_int *nreals, /* i: number of reals */
                                 CONST PDTFSTRING var_reals, /* i: array of real names */
                                 CONST dtf_double *reals /* i: array of reals */
                                 );

dtf_int dtf_update_vcval_reals_s(/* error code */
                                 CONST dtf_handle *fh, /* i: file handle */
                                 CONST dtf_int *simnum, /* i: simulation number */
                                 CONST dtf_int *zonenum, /* i: zone number */
                                 CONST dtf_int *vcnum, /* i: record number */
                                 CONST dtf_string name, /* i: VC value name */
                                 CONST dtf_int *nreals, /* i: number of reals */
                                 CONST PDTFSTRING var_reals, /* i: array of real names */
                                 CONST dtf_single *reals /* i: array of reals */
                                 );

/* update string evaluation data of a VC value */
dtf_int dtf_update_vcval_strings(/* error code */
                                 CONST dtf_handle *fh, /* i: file handle */
                                 CONST dtf_int *simnum, /* i: simulation number */
                                 CONST dtf_int *zonenum, /* i: zone number */
                                 CONST dtf_int *vcnum, /* i: record number */
                                 CONST dtf_string name, /* i: VC value name */
                                 CONST dtf_int *nstrings, /* i: number of strings */
                                 CONST PDTFSTRING var_strings, /* i: array of string names */
                                 CONST PDTFSTRING strings /* i: array of strings */
                                 );

/* update integer element of a VC value by name */
dtf_int dtf_update_vcval_int_by_name(/* error code */
                                     CONST dtf_handle *fh, /* i: file handle */
                                     CONST dtf_int *simnum, /* i: simulation number */
                                     CONST dtf_int *zonenum, /* i: zone number */
                                     CONST dtf_int *vcnum, /* i: record number */
                                     CONST dtf_string val_name, /* i: VC value name */
                                     CONST dtf_string elem_name, /* i: element name */
                                     CONST dtf_int *elem_value /* i: element value */
                                     );

/* update real (double) element of a VC value by name */
dtf_int dtf_update_vcval_real_d_by_name(/* error code */
                                        CONST dtf_handle *fh, /* i: file handle */
                                        CONST dtf_int *simnum, /* i: simulation number */
                                        CONST dtf_int *zonenum, /* i: zone number */
                                        CONST dtf_int *vcnum, /* i: record number */
                                        CONST dtf_string val_name, /* i: VC value name */
                                        CONST dtf_string elem_name, /* i: element name */
                                        CONST dtf_double *elem_value /* i: element value */
                                        );

/* update real (single) element of a VC value by name */
dtf_int dtf_update_vcval_real_s_by_name(/* error code */
                                        CONST dtf_handle *fh, /* i: file handle */
                                        CONST dtf_int *simnum, /* i: simulation number */
                                        CONST dtf_int *zonenum, /* i: zone number */
                                        CONST dtf_int *vcnum, /* i: record number */
                                        CONST dtf_string val_name, /* i: VC value name */
                                        CONST dtf_string elem_name, /* i: element name */
                                        CONST dtf_single *elem_value /* i: element value */
                                        );

/* update string element of a VC value by name */
dtf_int dtf_update_vcval_string_by_name(/* error code */
                                        CONST dtf_handle *fh, /* i: file handle */
                                        CONST dtf_int *simnum, /* i: simulation number */
                                        CONST dtf_int *zonenum, /* i: zone number */
                                        CONST dtf_int *vcnum, /* i: record number */
                                        CONST dtf_string val_name, /* i: VC value name */
                                        CONST dtf_string elem_name, /* i: element name */
                                        CONST dtf_string elem_value /* i: element value */
                                        );

/* delete a VC value */
dtf_int dtf_delete_vcval(/* error code */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         CONST dtf_int *vcnum, /* i: record number */
                         CONST dtf_string name /* i: VC value name */
                         );

/* delete all VC values */
dtf_int dtf_delete_all_vcvals(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *vcnum /* i: record number */
                              );

/***********************************************************************/
/* BLOCKS                                                              */
/***********************************************************************/

/* get number of blocks */
dtf_int dtf_query_nblocks(/* number of blocks attached to the zone */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum, /* i: simulation number */
                          CONST dtf_int *zonenum /* i: zone number */
                          );

/* get info about a block */
dtf_int dtf_query_block(/* number of cells in the block */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum, /* i: simulation number */
                        CONST dtf_int *zonenum, /* i: zone number */
                        CONST dtf_int *blocknum, /* i: block number */
                        dtf_int *key, /* o: key */
                        dtf_int *imin, /* o: minimum index in first plane */
                        dtf_int *imax, /* o: maximum index in first plane */
                        dtf_int *jmin, /* o: minimum index in second plane */
                        dtf_int *jmax, /* o: maximum index in second plane */
                        dtf_int *kmin, /* o: minimum index in third plane */
                        dtf_int *kmax /* o: maximum index in third plane */
                        );

/* modify existing block or add a new one */
dtf_int dtf_update_block(/* error code */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         CONST dtf_int *blocknum, /* i: block number */
                         CONST dtf_int *key, /* i: key */
                         CONST dtf_int *imin, /* i: minimum index in first plane */
                         CONST dtf_int *imax, /* i: maximum index in first plane */
                         CONST dtf_int *jmin, /* i: minumum index in second plane */
                         CONST dtf_int *jmax, /* i: maximum index in second plane */
                         CONST dtf_int *kmin, /* i: minimum index in third plane */
                         CONST dtf_int *kmax /* i: maximum index in third plane */
                         );

/* Delete a block */
dtf_int dtf_delete_block(/* Number of blocks in the zone after deletion */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         CONST dtf_int *blocknum /* i: block number */
                         );

/***********************************************************************/
/* CELL GROUPS                                                         */
/***********************************************************************/

/* get number of cell_groups */
dtf_int dtf_query_ncell_groups(/* number of cell groups attached to the zone */
                               CONST dtf_handle *fh, /* i: file handle */
                               CONST dtf_int *simnum, /* i: simulation number */
                               CONST dtf_int *zonenum /* i: zone number */
                               );

/* get info about a cell_group */
dtf_int dtf_query_cell_group(/* number of cells in the group number */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum, /* i: zone number */
                             CONST dtf_int *cell_groupnum, /* i: cell group number */
                             dtf_int *key /* o: key */
                             );

/* read cells of the cell_group */
dtf_int dtf_read_cell_group(/* error code */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *zonenum, /* i: zone number */
                            CONST dtf_int *cell_groupnum, /* i: cell group number */
                            dtf_int *cells /* o: array of cell numbers */
                            );

/* modify existing cell_group or add a new one */
dtf_int dtf_update_cell_group(/* error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *cell_groupnum, /* i: cell group number */
                              CONST dtf_int *key, /* i: key */
                              CONST dtf_int *ncells, /* i: number of cells in the group */
                              CONST dtf_int *cells /* i: array of cell numbers */
                              );

/* delete a cell_group */
dtf_int dtf_delete_cell_group(/* Number of cell groups in the zone after deletion */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum, /* i: zone number */
                              CONST dtf_int *cell_groupnum /* i: cell group number */
                              );

/***********************************************************************/
/* VOLUME CONDITIONS                                                   */
/***********************************************************************/

/* get number of volume_conditions */
dtf_int dtf_query_nvolume_conditions(/* number of volume conditions attached to the zone */
                                     CONST dtf_handle *fh, /* i: file handle */
                                     CONST dtf_int *simnum, /* i: simulation number */
                                     CONST dtf_int *zonenum /* i: zone number */
                                     );

/* get info about a volume_condition */
dtf_int dtf_query_volume_condition(/* error code */
                                   CONST dtf_handle *fh, /* i: file handle */
                                   CONST dtf_int *simnum, /* i: simulation number */
                                   CONST dtf_int *zonenum, /* i: zone number */
                                   CONST dtf_int *volume_conditionnum, /* i: volume condition number */
                                   dtf_int *vc_group_num, /* o: volume condition group number */
                                   dtf_int *vc_record_num /* o: volume condition record number */
                                   );

/* modify existing volume_condition or add a new one */
dtf_int dtf_update_volume_condition(/* error code */
                                    CONST dtf_handle *fh, /* i: file handle */
                                    CONST dtf_int *simnum, /* i: simulation number */
                                    CONST dtf_int *zonenum, /* i: zone number */
                                    CONST dtf_int *volume_conditionnum, /* i: volume condition number */
                                    CONST dtf_int *vc_group_num, /* i: volume condition group number */
                                    CONST dtf_int *vc_record_num /* i: volume condition record number */
                                    );

/* Delete an existing volume condition */
dtf_int dtf_delete_volume_condition(/* Number of volume conditions in the zone after deletion */
                                    CONST dtf_handle *fh, /* i: file handle */
                                    CONST dtf_int *simnum, /* i: simulation number */
                                    CONST dtf_int *zonenum, /* i: zone number */
                                    CONST dtf_int *volume_conditionnum /* i: volume condition number */
                                    );

/***********************************************************************/
/* MISC UTILITY FUNCTIONS                                              */
/***********************************************************************/

/* Special connectivity repairing routine that checks for duplicated nodes,
      and if they exist, repair all the c->n and bf->n and if->n connectivity.
      Works only for single zone mesh */

dtf_int dtf_check_rep_dup_nodes(/* return error code */
                                CONST dtf_handle *fh, /* i: file handle */
                                CONST dtf_int *simnum, /* i: simulation number */
                                CONST dtf_int *zonenum /* i: zone number */
                                );

/* Another special thing: Make the add_p3d_to_sim function public */

dtf_int dtf_add_p3d_to_sim(/* return error code */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simnum */
                           CONST char *filenamePFG, /* i: filename */
                           CONST dtf_int *has_blanking, /* i: Boolean for blanking data */
                           dtf_int *nzones_added /* o: number of zones added */
                           );

dtf_int dtf_strcasecmp(/* Returns standard C strcasecmp return value */
                       const char *s1, /* i: pointer to string 1 */
                       const char *s2 /* i: pointer to string 2 */
                       );

/* Zonal Interface Data */

/* Return total number of zonal interfaces for this (fh,sim) */
dtf_int dtf_query_nzi(/* Number of zonal interface groups for this (fh,sim) */
                      CONST dtf_handle *fh, /* i: file handle */
                      CONST dtf_int *simnum /* i: simulation number */
                      );

/* Find out information about this zonal interface  */
dtf_int dtf_query_zi(/* Find out information about this zonal group */
                     CONST dtf_handle *fh, /* i: file handle */
                     CONST dtf_int *simnum, /* i: simulation number */
                     CONST dtf_int *zinum, /* i: zonal interface number */
                     dtf_int *zone_L, /* o: "Left" zone */
                     dtf_int *zone_R, /* o: "Right" zone */
                     dtf_int *nfaces /* o: Number of faces for this zi */
                     );

/* Read in this zonal interface's facenums */
dtf_int dtf_read_zi(/* Read in this zonal interface's facenums */
                    CONST dtf_handle *fh, /* i: file handle */
                    CONST dtf_int *simnum, /* i: simulation number */
                    CONST dtf_int *zinum, /* i: zonal interface number */
                    dtf_int *facenum_L, /* o: face numbers in "Left" zone */
                    dtf_int *facenum_R /* o: 1face numbers in "Right" zone */
                    );

/* Return the total number of zonal interface groups for this (fh,sim,zone) */
dtf_int dtf_query_nzi_zone(/* Number of zonal interface groups for this (fh,sim,zone) */
                           CONST dtf_handle *fh, /* i: file handle */
                           CONST dtf_int *simnum, /* i: simulation number */
                           CONST dtf_int *zonenum /* i: zone number */
                           );

/* Find out information about this zonal interface  */
dtf_int dtf_query_zi_zone(/* Find out information about this zonal group */
                          CONST dtf_handle *fh, /* i: file handle */
                          CONST dtf_int *simnum, /* i: simulation number */
                          CONST dtf_int *zonenum, /* i: zone number */
                          CONST dtf_int *zinum, /* i: zonal interface number */
                          dtf_int *zone_L, /* o: "Left" zone */
                          dtf_int *zone_R, /* o: "Right" zone */
                          dtf_int *nfaces /* o: Number of faces for this zi */
                          );

/* Read in this zonal interface's facenums */
dtf_int dtf_read_zi_zone(/* Read in this zonal interface's facenums */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum, /* i: simulation number */
                         CONST dtf_int *zonenum, /* i: zone number */
                         CONST dtf_int *zinum, /* i: zonal interface number */
                         dtf_int *facenum_L, /* o: face numbers in "Left" zone */
                         dtf_int *facenum_R /* o: 1face numbers in "Right" zone */
                         );

/* Structured/Structured Zonal Interface Data */

/* The following api are specifically designed for structured/structured grid interfaces. */

/* Return total number of zonal interfaces for this (fh,sim) */
dtf_int dtf_query_nzi_ss(/* Number of zonal interface groups for this (fh,sim) */
                         CONST dtf_handle *fh, /* i: file handle */
                         CONST dtf_int *simnum /* i: simulation number */
                         );

/* Find out information about this zonal interface  */
dtf_int dtf_query_zi_ss(/* Find out information about this zonal group */
                        CONST dtf_handle *fh, /* i: file handle */
                        CONST dtf_int *simnum, /* i: simulation number */
                        CONST dtf_int *zinum, /* i: zonal interface number */
                        dtf_int *zone_L, /* o: "Left" zone */
                        dtf_int *zone_R, /* o: "Right" zone */
                        dtf_int *nfaces /* o: Number of faces for this zi */
                        );

/* Read in this zonal interface's facenums */
dtf_int dtf_read_zi_ss(/* Read in this zonal interface's facenums */
                       CONST dtf_handle *fh, /* i: file handle */
                       CONST dtf_int *simnum, /* i: simulation number */
                       CONST dtf_int *zinum, /* i: zonal interface number */
                       dtf_int *facenum_L, /* o: face numbers in "Left" zone */
                       dtf_int *facenum_R, /* o: 1face numbers in "Right" zone */
                       dtf_int *patchnum_L, /* o: patch numbers in "Left" zone */
                       dtf_int *patchnum_R /* o: patch numbers in "Right" zone */
                       );

/* Return the total number of zonal interface groups for this (fh,sim,zone) */
dtf_int dtf_query_nzi_zone_ss(/* Number of zonal interface groups for this (fh,sim,zone) */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *simnum, /* i: simulation number */
                              CONST dtf_int *zonenum /* i: zone number */
                              );

/* Find out information about this zonal interface  */
dtf_int dtf_query_zi_zone_ss(/* Find out information about this zonal group */
                             CONST dtf_handle *fh, /* i: file handle */
                             CONST dtf_int *simnum, /* i: simulation number */
                             CONST dtf_int *zonenum, /* i: zone number */
                             CONST dtf_int *zinum, /* i: zonal interface number */
                             dtf_int *zone_L, /* o: "Left" zone */
                             dtf_int *zone_R, /* o: "Right" zone */
                             dtf_int *nfaces /* o: Number of faces for this zi */
                             );

/* Read in this zonal interface's facenums */
dtf_int dtf_read_zi_zone_ss(/* Read in this zonal interface's facenums */
                            CONST dtf_handle *fh, /* i: file handle */
                            CONST dtf_int *simnum, /* i: simulation number */
                            CONST dtf_int *zonenum, /* i: zone number */
                            CONST dtf_int *zinum, /* i: zonal interface number */
                            dtf_int *facenum_L, /* o: face numbers in "Left" zone */
                            dtf_int *facenum_R, /* o: face numbers in "Right" zone */
                            dtf_int *patchnum_L, /* o: patch numbers in "Left" zone */
                            dtf_int *patchnum_R /* o: patch numbers in "Right" zone */
                            );

/****** Explicit manipulation of the connectivity table */

/* Clear out all the connectivity tables for this file */
dtf_int dtf_clear_file_connectivity_table(/* Return error code */
                                          CONST dtf_handle *fh /* i: file handle */
                                          );

/* Clear out this sims entries in the connectivity table */
dtf_int dtf_clear_sim_connectivity_table(/* Return error code */
                                         CONST dtf_handle *fh, /* i: file handle */
                                         CONST dtf_int *sim /* i: Simulation number */
                                         );

/* Clear out this zone's entries in the connectivity table */
dtf_int dtf_clear_zone_connectivity_table(/* Return error code */
                                          CONST dtf_handle *fh, /* i: file handle */
                                          CONST dtf_int *sim, /* i: Simulation number */
                                          CONST dtf_int *zone /* i: Zone number */
                                          );

/* Create this zone's entry in the connectivity table */
dtf_int dtf_create_zone_connectivity_table(/* Return error code */
                                           CONST dtf_handle *fh, /* i: file handle */
                                           CONST dtf_int *sim, /* i: Simulation number */
                                           CONST dtf_int *zone /* i: Zone number */
                                           );

/* Flush this files cached data */
dtf_int dtf_flush_file_cache(/* Return error code */
                             CONST dtf_handle *fh /* i: file handle */
                             );

/* Unadvertised public api */

/* Set the mode for destructing structured grids zones. */
dtf_int dtf_set_destruct_mode(/* Return error code */
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *sim, /* i: Simulation number */
                              CONST dtf_int *zone, /* i: Zone number */
                              CONST dtf_int *mode /* i: Destruct mode: DTF_TRUE or DTF_FALSE */
                              );

/* Get the mode for destructing structured grids zones. */
dtf_int dtf_get_destruct_mode(/* Return destruct mode*/
                              CONST dtf_handle *fh, /* i: file handle */
                              CONST dtf_int *sim, /* i: Simulation number */
                              CONST dtf_int *zone /* i: Zone number */
                              );

/***********************************************************************/

/***********************************************************************/

/* Very ACE+ specific functions. Not for the general public */

dtf_int dtf_map_nodes(dtf_double *xmap, dtf_double *ymap, dtf_double *zmap, dtf_int *nnodes_map, dtf_int *nodenum, dtf_int *map, dtf_double *tolerance);

/***********************************************************************/

/* Very CFD-VisCart specific functions. Not for the general public */

/* Create an empty poly zone in the file. Must fill up with following api before it is usable. */
dtf_int dtf_create_empty_poly_zone(/* Return zone # */
                                   CONST dtf_handle *fh, /* i: file handle */
                                   CONST dtf_int *sim, /* i: Simulation number */
                                   CONST dtf_int *nnodes, /* i: Number of nodes to add */
                                   CONST dtf_int *n_faces_total, /* i: total number of faces */
                                   CONST dtf_int *is_2D /* i: Mesh is 2D/3D */
                                   );

/* Add x coordinates to an empty poly_zone */
dtf_int dtf_add_x_to_empty_poly_zone(/* Return error code */
                                     CONST dtf_handle *fh, /* i: file handle */
                                     CONST dtf_int *sim, /* i: Simulation number */
                                     CONST dtf_int *zone, /* i: Zone number */
                                     CONST dtf_double *x /* i: array of X coordinates */
                                     );

/* Add y coordinates to an empty poly_zone */
dtf_int dtf_add_y_to_empty_poly_zone(/* Return error code */
                                     CONST dtf_handle *fh, /* i: file handle */
                                     CONST dtf_int *sim, /* i: Simulation number */
                                     CONST dtf_int *zone, /* i: Zone number */
                                     CONST dtf_double *y /* i: array of Y coordinates */
                                     );

/* Add z coordinates to an empty poly_zone */
dtf_int dtf_add_z_to_empty_poly_zone(/* Return error code */
                                     CONST dtf_handle *fh, /* i: file handle */
                                     CONST dtf_int *sim, /* i: Simulation number */
                                     CONST dtf_int *zone, /* i: Zone number */
                                     CONST dtf_double *z /* i: array of Z coordinates */
                                     );

/* Add nodes to an empty poly_zone */
dtf_int dtf_add_f2n_to_empty_poly_zone(/* Return error code */
                                       CONST dtf_handle *fh, /* i: file handle */
                                       CONST dtf_int *sim, /* i: Simulation number */
                                       CONST dtf_int *zone, /* i: Zone number */
                                       CONST dtf_int *len_f2n, /* i: length of face->node array */
                                       CONST dtf_int *f2n /* i: face->node array */
                                       );

/* Add #nodes/face to an empty poly_zone */
/* Return error code */
dtf_int dtf_add_n_nodes_per_face_to_empty_poly_zone(
    CONST dtf_handle *fh, /* i: file handle */
    CONST dtf_int *sim, /* i: Simulation number */
    CONST dtf_int *zone, /* i: Zone number */
    CONST dtf_int *n_nodes_per_face /* i: array containing number of nodes for each face */
    );

/* Add f2c to an empty poly_zone */
dtf_int dtf_add_f2c_to_empty_poly_zone(/* Return error code */
                                       CONST dtf_handle *fh, /* i: file handle */
                                       CONST dtf_int *sim, /* i: Simulation number */
                                       CONST dtf_int *zone, /* i: Zone number */
                                       CONST dtf_int *f2c /* i: face->cell connectivity array */
                                       );
/***********************************************************************/

/* Very DTFOL specific functions. Not for the general public */

dtf_int dtf_perturb_node(dtf_double *px, dtf_int *plev0);

/***********************************************************************/

/* __END__ */

#ifdef __cplusplus
}
#endif
#endif
