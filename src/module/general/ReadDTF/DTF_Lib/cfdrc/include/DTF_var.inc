C/************************************************************************
C*                  CFDRC Data Transfer Format Library                   *
C*                    (c) CFD Research Corporation                       *
C*                        Huntsville, Alabama, USA.                      *
C*                               1996.                                   *
C************************************************************************/

C/************************************************************************
C*       FILE:   dtf.h		                                        *
C*                                                                       *
C*       Public header file for DTF library				*
C*                                                                       *
C*       Freddy Golos / William Coirier                                  *
c*     
c* This file contributed by Steve Scherr, AFRL/VAAC
C************************************************************************/

C/***********************************************************************/
C/* Flags to pass to dtf_file_contents                                  */
C/***********************************************************************/

	integer CONTENTS_VERBOSE,CONTENTS_NOT_VERBOSE
	parameter(	CONTENTS_VERBOSE =	-1)
	parameter(	CONTENTS_NOT_VERBOSE =	10)

C/***********************************************************************/
C/* basic variable types used throughout the library                    */
C/***********************************************************************/

	integer DTF_MAX_STRING,DTF_MAX_UNITS
	parameter(	DTF_MAX_STRING =	80)
	parameter(	DTF_MAX_UNITS =	32)


C/***********************************************************************/
C/* error codes that the library may return                             */
C/***********************************************************************/

	integer DTF_ERROR, DTF_OK , DTF_UNSUPPORTED_ID
	integer DTF_BAD_ZONETYPE, DTF_BAD_FH, DTF_BAD_SIMNUM
	integer DTF_BAD_ZONENUM, DTF_BAD_SWAPSIZE, DTF_BAD_MODE
	integer DTF_BAD_HANDLE, DTF_FILE_IS_NOT_DTF, DTF_BAD_DATANUM
	integer DTF_CANT_OPEN, DTF_CANT_CREATE, DTF_BAD_FACENUM
	integer DTF_BAD_CELLNUM, DTF_BAD_DATATYPE, DTF_BAD_TOPOTYPE
	integer DTF_BAD_BCNUM, DTF_BAD_SIZES, DTF_BAD_STRUCT_DIMS
	integer DTF_BAD_CELL_GROUPNUM, DTF_BAD_BLOCKNUM
	integer DTF_UNDER_CONSTRUCTION, DTF_NO_USER_FACE_DATA
	integer DTF_NOT_A_BC_FACE, DTF_NOT_A_INTERFACE_FACE
	integer DTF_2D_NOT_DEFINED, DTF_OUT_OF_MEMORY
	integer DTF_BAD_CATEGORY_NUMBER, DTF_BAD_BCVAL_NUMBER
	integer DTF_BAD_BCVAL_NAME, DTF_BAD_BCVAL_INT_NAME
	integer DTF_BAD_BCVAL_REAL_NAME, DTF_BAD_BCVAL_STRING_NAME
        integer DTF_BAD_ELEMENT_INDEX, DTF_BAD_VCNUM
	integer DTF_BAD_VCVAL_NUMBER, DTF_BAD_VCVAL_NAME
	integer DTF_BAD_VCVAL_INT_NAME, DTF_BAD_VCVAL_REAL_NAME
	integer DTF_BAD_VCVAL_STRING_NAME, DTF_BAD_VOLUME_CONDITIONNUM
	integer DTF_NO_STRING_ARRAYS_IN_FORTRAN, DTF_BAD_PATCHNUM
	integer DTF_CORRUPTED_OLD_VC_DATA, DTF_OFFSET_NOT_FOUND
	integer DTF_BAD_F2N_DATA, DTF_BAD_F2C_DATA

	parameter( DTF_ERROR		=	-1)
        parameter( DTF_OK 			=	 0)
	parameter( DTF_UNSUPPORTED_ID	=  1)
	parameter( DTF_BAD_ZONETYPE	=  2)
	parameter( DTF_BAD_FH	=  3)
	parameter( DTF_BAD_SIMNUM	=  4)
	parameter( DTF_BAD_ZONENUM	=  5)
	parameter( DTF_BAD_SWAPSIZE	=  6)
	parameter( DTF_BAD_MODE	=  7)
	parameter( DTF_BAD_HANDLE	=  8)
	parameter( DTF_FILE_IS_NOT_DTF	=  9)
	parameter( DTF_BAD_DATANUM	= 10)
	parameter( DTF_CANT_OPEN	= 11)
	parameter( DTF_CANT_CREATE	= 12)
	parameter( DTF_BAD_FACENUM	= 13)
	parameter( DTF_BAD_CELLNUM	= 14)
	parameter( DTF_BAD_DATATYPE	= 15)
	parameter( DTF_BAD_TOPOTYPE	= 16)
	parameter( DTF_BAD_BCNUM	= 17)
	parameter( DTF_BAD_SIZES	= 18)
	parameter( DTF_BAD_STRUCT_DIMS	= 19)
	parameter( DTF_BAD_CELL_GROUPNUM	= 20)
	parameter( DTF_BAD_BLOCKNUM	= 21)
	parameter( DTF_UNDER_CONSTRUCTION	= 22)
	parameter( DTF_NO_USER_FACE_DATA	= 23)
	parameter( DTF_NOT_A_BC_FACE	= 24)
	parameter( DTF_NOT_A_INTERFACE_FACE	= 25)
	parameter( DTF_2D_NOT_DEFINED	= 26)
	parameter( DTF_OUT_OF_MEMORY	= 27)
	parameter( DTF_BAD_CATEGORY_NUMBER	= 28)
	parameter( DTF_BAD_BCVAL_NUMBER	= 29)
	parameter( DTF_BAD_BCVAL_NAME	= 30)
	parameter( DTF_BAD_BCVAL_INT_NAME	= 31)
	parameter( DTF_BAD_BCVAL_REAL_NAME	= 32)
	parameter( DTF_BAD_BCVAL_STRING_NAME	= 33)
        parameter( DTF_BAD_ELEMENT_INDEX	= 34)
        parameter( DTF_BAD_VCNUM	= 35)
	parameter( DTF_BAD_VCVAL_NUMBER	= 36)
	parameter( DTF_BAD_VCVAL_NAME	= 37)
	parameter( DTF_BAD_VCVAL_INT_NAME	= 38)
	parameter( DTF_BAD_VCVAL_REAL_NAME	= 39)
	parameter( DTF_BAD_VCVAL_STRING_NAME	= 40)
	parameter( DTF_BAD_VOLUME_CONDITIONNUM	= 41)
	parameter( DTF_NO_STRING_ARRAYS_IN_FORTRAN	= 42)
	parameter( DTF_BAD_PATCHNUM	= 43)
	parameter( DTF_CORRUPTED_OLD_VC_DATA	= 44)
	parameter( DTF_OFFSET_NOT_FOUND	= 45)
	parameter( DTF_BAD_F2N_DATA	= 46)
	parameter( DTF_BAD_F2C_DATA	= 47)

C/***********************************************************************/
C/* permissible datatypes, used by dtf_add_data, dtf_read_data, etc     */
C/***********************************************************************/

	integer DTF_INT_DATA, DTF_DOUBLE_DATA
	integer DTF_SINGLE_DATA, DTF_STRING_DATA
	parameter( DTF_INT_DATA		= 1)
	parameter( DTF_DOUBLE_DATA		= 2)
	parameter( DTF_SINGLE_DATA		= 3)
	parameter( DTF_STRING_DATA		= 4)

C/***********************************************************************/
C/* permissible topotypes, used by dtf_add_data, dtf_read_data, etc     */
C/***********************************************************************/

	integer DTF_GNRL_TOPO, DTF_NODE_TOPO, DTF_EDGE_TOPO
	integer DTF_FACE_TOPO, DTF_CELL_TOPO
	parameter( DTF_GNRL_TOPO		= 0)
	parameter( DTF_NODE_TOPO		= 1)
	parameter( DTF_EDGE_TOPO		= 2)
	parameter( DTF_FACE_TOPO		= 3)
	parameter( DTF_CELL_TOPO		= 4)


C/***********************************************************************/
C/* type of a zone, returned by dtf_query_zonetype                      */
C/***********************************************************************/

	integer DTF_STRUCTURED_ZONE, DTF_CARTESIAN_ZONE
	integer DTF_UNSTRUCTURED_ZONE, DTF_POINT_NET_ZONE
	integer DTF_POLY_ZONE, DTF_VZ_MODE_ZONE
	parameter( DTF_STRUCTURED_ZONE	= 1)
	parameter( DTF_CARTESIAN_ZONE	= 2)
	parameter( DTF_UNSTRUCTURED_ZONE	= 3)
	parameter( DTF_POINT_NET_ZONE	= 4)
	parameter( DTF_POLY_ZONE	        = 5)
	parameter( DTF_VZ_MODE_ZONE        = 6)

C/***********************************************************************/
C/* some constant definitions for unstructured grids                    */
C/***********************************************************************/

C/* faces:
C *     Edge,
C *     Triangle,
C *     Quadrilateral,
C *     Edge-Quadratic,
C *     Triangle-Quadratic,
C *     Quadrilateral-Quadratic,
C *     Poly-Face
C */
	integer DTF_NFACETYPES,DTF_NFACEKINDS
	integer DTF_UNST_NFACETYPES,DTF_UNST_NFACEKINDS
	parameter( DTF_NFACETYPES = 7)
	parameter( DTF_NFACEKINDS = DTF_NFACETYPES*3)
	parameter( DTF_UNST_NFACETYPES = 3)
	parameter( DTF_UNST_NFACEKINDS = DTF_UNST_NFACETYPES*3)
C extern const int        DTF_FACETYPE_TO_NNODES[];  /* { 2, 3, 4, 3, 6, 8, -1 } */

	integer DTF_EDGE_FACE, DTF_TRI_FACE, DTF_QUAD_FACE
	integer DTF_EDGE_Q_FACE, DTF_TRI_Q_FACE, DTF_QUAD_Q_FACE
	integer DTF_POLY_FACE
	parameter( DTF_EDGE_FACE     = 0)
	parameter( DTF_TRI_FACE      = 1)
	parameter( DTF_QUAD_FACE     = 2)
	parameter( DTF_EDGE_Q_FACE   = 3)
	parameter( DTF_TRI_Q_FACE    = 4)
	parameter( DTF_QUAD_Q_FACE   = 5)
	parameter( DTF_POLY_FACE     = 6)

C/* cells:
C *     Triangle,
C *     Quadrilateral,
C *     Tetrahedron,
C *     Pyramid,
C *     Prism,
C *     Hexahedron,
C *     Triangle-Quadratic,
C *     Quadrilateral-Quadratic,
C *     Tetrahedron-Quadratic,
C *     Hexahedron-Quadratic,
C *     Poly-Cell
C */
	integer DTF_NCELLTYPES,DTF_UNST_NCELLTYPES
	parameter( DTF_NCELLTYPES = 11)
	parameter( DTF_UNST_NCELLTYPES = 6)
C extern const int        DTF_CELLTYPE_TO_NNODES[];       /* { 3, 4, 4, 5, 6, 8, 6, 8, 10, 20, -1 } */
C extern const int        DTF_CELLTYPE_TO_NFACES[];  /* { 3, 4, 4, 5, 5, 6, 3, 4, 4, 6, -1 } */

	integer DTF_TRI_CELL, DTF_QUAD_CELL, DTF_TET_CELL
	integer DTF_PYR_CELL, DTF_PRISM_CELL, DTF_HEX_CELL
	integer DTF_TRI_Q_CELL, DTF_QUAD_Q_CELL, DTF_TET_Q_CELL
	integer DTF_HEX_Q_CELL, DTF_POLY_CELL
	parameter( DTF_TRI_CELL     = 0)
	parameter( DTF_QUAD_CELL    = 1)
	parameter( DTF_TET_CELL     = 2)
	parameter( DTF_PYR_CELL     = 3)
	parameter( DTF_PRISM_CELL   = 4)
	parameter( DTF_HEX_CELL     = 5)
	parameter( DTF_TRI_Q_CELL   = 6)
	parameter( DTF_QUAD_Q_CELL  = 7)
	parameter( DTF_TET_Q_CELL   = 8)
	parameter( DTF_HEX_Q_CELL   = 9)
	parameter( DTF_POLY_CELL    = 10)

C/***********************************************************************/
C/* face types                                                          */
C/***********************************************************************/

        integer DTF_INTERIOR_FACE, DTF_INTERFACE_FACE, DTF_BOUNDARY_FACE
        parameter( DTF_INTERIOR_FACE   = 1)
        parameter( DTF_INTERFACE_FACE  = 2)
        parameter( DTF_BOUNDARY_FACE   = 3)

C/***********************************************************************/
C/* TRUE/FALSE constants                                                */
C/***********************************************************************/

	integer DTF_TRUE,DTF_FALSE
	parameter( DTF_TRUE  = 1)
	parameter( DTF_FALSE = 0)

