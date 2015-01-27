/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Copyright (c)     1996-2005.  ANSYS Europe Ltd.
File Description: Volume Mesh Import API.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

#ifndef _cfxIMPORT_H_
#define _cfxIMPORT_H_

#include <stddef.h>

#include "cfxids.h"

/*
--------------------------------------------------------------------------------
Allowable element types
--------------------------------------------------------------------------------
*/
typedef enum
{
    cfxELEM_BAD = 0,
    cfxELEM_TET = 4, /* Tetrahedral element    (4 nodes)   */
    cfxELEM_PYR = 5, /* Pyramidal element      (5 nodes)   */
    cfxELEM_WDG = 6, /* Wedge or Prism element (6 nodes)   */
    cfxELEM_HEX = 8 /* Hexahedral element     (8 nodes)   */
} cfxImpElemType_t;

/*
--------------------------------------------------------------------------------
Allowable region types
--------------------------------------------------------------------------------
*/

typedef enum
{
    cfxImpREG_ERROR = 0,
    cfxImpREG_NODES,
    cfxImpREG_FACES,
    cfxImpREG_ELEMS
} cfxImpRegType_t;

/*
--------------------------------------------------------------------------------
Allowable region source
--------------------------------------------------------------------------------
*/

typedef enum
{
    cfxImpRS_PRIMITIVE_FROM_FILE = 0,
    cfxImpRS_COMPOSITE_FROM_FILE,
    cfxImpRS_GENERATED
} cfxImpRegSource_t;

/*
--------------------------------------------------------------------------------
Total count entries
--------------------------------------------------------------------------------
*/

enum cfxImpCounts
{
    cfxImpCNT_NODE = 0, /* number of nodes                */
    cfxImpCNT_ELEMENT, /* number of elements             */
    cfxImpCNT_REGION, /* number of regions              */
    cfxImpCNT_UNUSED, /* number of unused nodes         */
    cfxImpCNT_DUP, /* number of duplicate nodes      */
    cfxImpCNT_TET, /* number of tetrahedral elements */
    cfxImpCNT_PYR, /* number of ptramid elements     */
    cfxImpCNT_WDG, /* number of wedge elements       */
    cfxImpCNT_HEX, /* number of hexhedral elements   */
    cfxImpCNT_SIZE /* size of count array            */
};

/*
--------------------------------------------------------------------------------
Function prototypes
--------------------------------------------------------------------------------
*/

/* For backwards compatability */
#define cfxFACEID(a, b) cfxImportFaceID(a, b)

#if defined(__cplusplus)
extern "C" {
#endif
/* Process Control */
/* =============== */

/* Initialize for import process */
void
cfxImportInit(void);

void
cfxImportSetup(const size_t nodeCount,
               const size_t elementCount,
               const size_t regionCount);

/* Test import routine */
int
cfxImportTest(const char *filename);

/* Geometry Units */
int
cfxImportUnits(
    const char *units /* Length units of imported mesh */
    );

/* Terminate import process and write data to CFX-5 or test file. */
long
cfxImportDone(void);

/* Checks if import process is connected to the CFX-5 */
int
cfxImportStatus(void);

/* Define callback for errors */
void
cfxImportError(
    void (*callback)(char *errmsg) /* User supplied function */
    );

/* Issue a warning */
void
cfxImportWarning(
    char *wanmsg /* Warning message to issue */
    );

/* Terminate with error message */
void
cfxImportFatal(
    const char *errmsg /* Error message to handle. */
    );

/* Return totals of objects imported */
long
cfxImportTotals(
    size_t counts[cfxImpCNT_SIZE] /* See "Total count entries" */
    );

/* Nodes */
/* ===== */

/* Import a node */
ID_t
cfxImportNode(
    const ID_t nodeid, /* node identifier (ID) */
    const double x, /* x coordinate */
    const double y, /* y coordinate */
    const double z /* z coordinate */
    );

/* Get coordinates of a node with a given ID */
ID_t
cfxImportGetNode(
    const ID_t nodeid, /* node ID */
    double *x, /* returned coordinates */
    double *y,
    double *z);

/* Return the list of all imported node ID's */
ID_t *
cfxImportNodeList(void);

/*
   * Explicitly map 2 nodes to each other (i.e. mark as duplicated)
   */
int
cfxImportMap(
    const ID_t nodeid, /* reference node id */
    const ID_t mapid /* node id to map */
    );

/* 
   * Get the bounding box of nodes
   */
int
cfxImportRange(
    double *xmin, /* lower x limit */
    double *ymin, /* lower y limit */
    double *zmin, /* lower z limit */
    double *xmax, /* upper x limit */
    double *ymax, /* upper y limit */
    double *zmax /* upper z limit */
    );

/* Elements */
/* ======== */

/* Import an element */
ID_t
cfxImportElement(
    const ID_t elemid, /* element identifier (ID) */
    const cfxImpElemType_t elemtype, /* See "Allowable Element Types"  */
    const ID_t *nodelist /* node IDs defining the element */
    );

/* Get the node ID's at the element vertices */
int
cfxImportGetElement(
    const ID_t elemid, /* element ID */
    ID_t nodeid[] /* returned node ID's */
    );

/* Return the list of all imported element ID's */
ID_t *
cfxImportElementList(void);

/* Get a faces ID from the element and relative face of an element */
ID_t
cfxImportFaceID(
    const ID_t elemid, /* element ID */
    const int facenum /* relative face number */
    );

/* Get the node ID's of the relative face of an element */
int
cfxImportGetFace(
    const ID_t elemid, /* element ID */
    const int facenum, /* relative face number */
    ID_t nodeid[] /* face nodes */
    );

/* Return the relative face number of an element given a set of nodes */
int
cfxImportFindFace(
    const ID_t elemid, /* element ID */
    const int nnodes, /* number of nodes */
    const ID_t nodeid[] /* face nodes */
    );

/* Simple Regions */
/* ============== */

/* Begin to define a region specification */
size_t
cfxImportBegReg(
    const char *name, /* region name */
    const cfxImpRegType_t regtype /* See "Allowable region types" above */
    );

/* Add a number of object ID's to a region specification. */
size_t
cfxImportAddReg(
    const size_t numobjs, /* number of objects to add */
    ID_t *objlist /* object list for region */
    );

/* Finish defining a region specification */
size_t
cfxImportEndReg(void);

/* Define a region specification, add object ID's, finish defining a region */
size_t
cfxImportRegion(
    const char *name, /* region name */
    const cfxImpRegType_t regtype, /* region type */
    const size_t numobjs, /* number of objects */
    ID_t *objlist /* object ID's in region */
    );

/* Rename a region */
int
cfxImportSetRegionName(
    const char *regName, /* Original region name */
    const char *newName /* New region name */
    );

/* Get the list of all imported region names */
char **
cfxImportRegionList(void);

/* Get the type of a region or cfxImpREG_ERROR if it doesn't exist */
cfxImpRegType_t
cfxImportRegionExists(
    const char *name /* region name to query*/
    );

/* Get object ID's defined used by region */
ID_t *
cfxImportGetRegion(
    const char *name, /* Name of the region to query */
    cfxImpRegType_t *type /* Type of the region */
    );

/* Composite Regions */
/* ================= */

/* Begin defining a composite region name */
int
cfxImportBegCompReg(
    const char *name /* Name of the composite region */
    );

/* Add composite region components to a composite region being defined */
int
cfxImportAddCompRegComponents(
    const size_t count, /* Number of components */
    const char **components /* Component names */
    );

/* Finish defining a composite region */
int
cfxImportEndCompReg(void);

/* Query whether a composite region is defined exists */
int
cfxImportCompositeExists(
    const char *name /* Composite region name */
    );

/* Define a composite region with a given set of components */
int
cfxImportCompositeRegion(
    const char *name, /* Composite region name */
    const size_t count, /* Number of components */
    const char **components /* Component names */
    );

int
cfxImportAddCompositeCCL(
    const size_t len, /* Length of CCL string */
    char *cclString /* CCL String */
    );

/* 
   * Specify the source of a region: whether it was  was created by the import 
   * process or read from the original mesh file.
   */
int
cfxImportSetRegionSource(const char *name, const cfxImpRegSource_t source);

#ifdef __cplusplus
}
#endif

#endif /* _cfxIMPORT_H_ */

/*
================================================================================
*/
