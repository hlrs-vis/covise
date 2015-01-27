/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Appdata functions

// Our APP data chunk IDs
#define NORMALS_ID 0
#define INDENT_ID 1
#define FIELDS_ID 2
#define OUTPUT_LANG_ID 3
#define USE_PREFIX_ID 4
#define URL_PREFIX_ID 5
#define CAMERA_ID 6
#define MAX_POLYS_ID 7
#define MAX_SELECTED_ID 8
#define UPDIR_ID 9
#define DIGITS_ID 10
#define COORD_INTERP_ID 11
#define TFORM_SAMPLE_ID 12
#define COORD_SAMPLE_ID 13
#define TFORM_SAMPLE_RATE_ID 14
#define COORD_SAMPLE_RATE_ID 15
#define NAV_INFO_ID 16
#define TITLE_ID 17
#define INFO_ID 18
#define EXPORT_HIDDEN_ID 19
#define PRIMITIVES_ID 20
#define BACKGROUND_ID 21
#define FOG_ID 22
#define TOUCH_ID 23
#define ANCHOR_ID 24
#define POLYGON_TYPE_ID 25
#define ENABLE_PROGRESS_BAR_ID 26
#define EXPORT_PRE_LIGHT_ID 27
#define FLIP_BOOK_ID 28
#define FLIPBOOK_SAMPLE_ID 29
#define FLIPBOOK_SAMPLE_RATE_ID 30
#define CPV_SOURCE_ID 31
#define DEFUSE_ID 32
#define USELOD_ID 33
#define EXPORTLIGHTS_ID 34
#define COPYTEXTURES_ID 35
#define SKY_ID 36
#define OCCLUDER_ID 37
#define FORCE_WHITE_ID 38

extern void WriteAppData(Interface *ip, int id, TCHAR *val);
extern void GetAppData(Interface *ip, int id, TCHAR *def,
                       TCHAR *val, int len);
