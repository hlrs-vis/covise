/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*RICARDO SQA =========================================================
 * Status        : UNASSURED
 * Module Name   : v2e
 * Subject       : Vectis Phase 5 POST to Ensight Gold convertor
 * Language      : ANSI C
 * Requires      : RUtil (on little-endian platforms only)
 * Documentation : README.html
 * Filename      : v2e.h
 * Author        : RJF
 * Creation Date : Oct 2000
 * Last Modified : $Date: $
 * Version       : $Revision: $
 *======================================================================
 */

/* ---------------------------------------------------------- */

void clear_node_table(void);
void free_node_table(void);
int add_node_from_face(int ispatch, int lindex);
int node_comparison_function(const void *a, const void *b);
void sort_node_table(void);

BOOL calculate_cell_centre(int cell, node *centre);
void update_node_table_coordinates(void);
void write_patches(FILE *of);
void write_binary_patches();
void write_binary_boundaries();
void calculate_scalar_cell_connectivity(void);
void generate_node_connectivity(FILE *fo, FILE *fe);
void generate_binary_node_connectivity(FILE *fo, FILE *fe);
void write_scalar_cells(FILE *of, FILE *fc);
void write_binary_scalar_cells();
void write_droplets(FILE *of);
void write_binary_droplets();
void write_result_vars(void);
void write_binary_result_vars(void);
void write_geometry_file(FILE *f);
void write_binary_geometry_file();
void write_general_data(void);
void write_result_set(void);
void write_case(void);
fortran_header read_record_header(void);

void slurp_general(void);
void clear_result_vars(void);
void allocate_droplet_vars(void);
void free_droplet_vars(void);
BOOL slurp_result_set(void);

void free_memory(void);
void free_geometry(void);
void free_result_vars(void);
void free_result_sets(void);
void remove_temp_files(void);

void usage(void);
int translate_file(char *post_filename);
