/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CFX_EXPORT_LIB
#define CFX_EXPORT_LIB

#define REQ_TYPE_ALL 0
#define REQ_TYPE_TETRA 4 // must be cfxELEM_TET
#define REQ_TYPE_PYRAM 5 // must be cfxELEM_PYR
#define REQ_TYPE_WEDGE 6 // must be cfxELEM_WDG
#define REQ_TYPE_HEXA 8 // must be cfxELEM_HEX

int cfx_getInfo(const char *file_name, int level_of_interest, int zone,
                int crop, float cropXMin, float cropXMax,
                float cropYMin, float cropYMax,
                float cropZMin, float cropZMax,
                int timestep, int timestep_by_idx,
                int *num_tetra, int *num_pyra, int *num_wedge, int *num_hexa,
                int *nnodes,
                char *components_to_read, char *components_to_read_delimiter,
                int output_zone_id,
                int *node_veclen, int *num_node_components,
                int output_boundaryNodes, int *num_boundaryies,
                float *time_val,
                int *timeStepCnt,
                int allow_zone_rotation,
                char *ucdName,
                int exportInit, int exportDone);

int cfx_getData(const char *file_name, int level_of_interest, int zone,
                int crop, float cropXMin, float cropXMax,
                float cropYMin, float cropYMax,
                float cropZMin, float cropZMax,
                int timestep, int timestep_by_idx,
                float *node_x, float *node_y, float *node_z,
                int required_cell_type,
                int *node_list,
                unsigned char *cell_types,
                char *components_to_read, char *components_to_read_delimiter,
                float *node_data,
                int *node_components,
                char *node_component_labels,
                int fix_boundary_nodes,
                int output_zone_id,
                int output_boundaryNodes, char *boundary_node_label, char *boundary_node_labels,
                char *search_string,
                int allowZoneRotation,
                int output_boundaries, int output_regions,
                int remove_faces, double faceNormal[3], int **geomNodes, int *geomNodeCnt, int *geomObjCnt, int *geomNodeObjSizes, char geomObjNames[][256], int *geomPListCounts,
                char *ucdName,
                int exportInit, int exportDone);

void getNodeComponentLabel(char *node_component_labels, int c, char *res_label);

#endif
