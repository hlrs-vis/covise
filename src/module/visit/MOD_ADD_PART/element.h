/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef MAX_NODES_ELEMENT

#define MIXED 0
#define NODE 1
#define BAR_2 2
#define BAR_3 3
#define TETRA_4 4
#define TETRA_10 5
#define HEXA_8 6
#define HEXA_20 7
#define TRI_3 8
#define TRI_3_X 9
#define TRI_6 10
#define TRI_6_X 11
#define PENTA_6 12
#define PENTA_15 13
#define QUAD_4 14
#define QUAD_4_X 15
#define QUAD_8 16
#define QUAD_8_X 17
#define PYRA_5 18

#define QUAD_9 19
#define HEXA_27 20
#define PENTA_18 21
#define PYRA_14 22

#define PYRA_13 24

#define POLYGON 23

#define N_ELEMENT_TYPES 25

#define MAX_NODES_ELEMENT 27
#define MAX_EDGES_ELEMENT 12
#define MAX_FACES_ELEMENT 6
#define MAX_FACE_SIDES 10

#define EDGE_ELEMENT_MASK ((1 << BAR_2) | (1 << BAR_3))
#define SURFACE_ELEMENT_MASK ((1 << TRI_3) | (1 << TRI_6) | (1 << QUAD_4) | (1 << QUAD_8))
#define VOLUME_ELEMENT_MASK ((1 << TETRA_4) | (1 << TETRA_10) | (1 << HEXA_8) | (1 << HEXA_20) | (1 << PENTA_6) | (1 << PENTA_15) | (1 << PYRA_5) | (1 << HEXA_27) | (1 << PENTA_18) | (1 << PYRA_14) | (1 << PYRA_13))

#define LINEAR_ELEMENT_MASK ((1 << NODE) | (1 << BAR_2) | (1 << TETRA_4) | (1 << HEXA_8) | (1 << TRI_3) | (1 << PENTA_6) | (1 << QUAD_4) | (1 << PYRA_5))

#define QUADRATIC_ELEMENT_MASK ((1 << BAR_3) | (1 << TETRA_10) | (1 << HEXA_20) | (1 << TRI_6) | (1 << PENTA_15) | (1 << QUAD_8) | (1 << QUAD_9) | (1 << HEXA_27) | (1 << PENTA_18) | (1 << PYRA_14) | (1 << PYRA_13))
#endif
