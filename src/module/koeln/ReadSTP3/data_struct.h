/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef DATA_STRUCT
#define DATA_STRUCT

/* data_struct.h
 * Some basic data structures 
 * Changes:
 * 18.12.1995, UB, formatted
 * 08.07.1996  JP  changes for 16bit images 
 * 26.08.1996  KOE XYZ_F included
 */

typedef struct XY
{
    double x;
    double y;
} XY;

typedef struct XYZ
{
    double x;
    double y;
    double z;
} XYZ;

/* KOE 26.8.96 --> */
typedef struct
{
    float x;
    float y;
    float z;
} XYZ_F;
/* KOE 26.8.96 <-- */

typedef struct XYZE
{
    double x;
    double y;
    double z;
    double err;
} XYZE;

typedef XY Image_coord;

/* jp 16bit -->
* typedef unsigned char Image_pixel;
*/
typedef short int Image_pixel;

typedef union
{
    unsigned char *char_ptr;
    short int *short_ptr;
    int *int_ptr;
} Data_pointer;

typedef struct
{
    Data_pointer data; /* pointer to the pixel data */
    int type; /* sizeof data type for one pixel 
                             1 =>char, 2=>short, 4=>long*/
} Image_data;

/* macro for direct access to image pixels independend of type */
#define IMAGEPIX(IMD, IDX) (int)(                                                                                                                                                     \
    IMD.type == 1 ? IMD.data.char_ptr[IDX] : IMD.type == 2 ? IMD.data.short_ptr[IDX] : IMD.type == 4 ? IMD.data.int_ptr[IDX] : (printf("error in %s, line %d\n", __FILE__, __LINE__), \
                                                                                                                                exit(2),                                              \
                                                                                                                                0))
/* jp 16bit <-- */

typedef unsigned char Voxel;

typedef struct Plane
{
    double a;
    double b;
    double c;
    double d;
} Plane;

#endif
