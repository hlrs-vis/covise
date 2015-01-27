/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef STP_CONST_H
#define STP_CONST_H

/*      27.1.94 sk addded DIGI_MATRIX  for CBI
        21.01.97 AO CINE_ALTERNATING, ALTERFORW, ALTERBACK added for new Cine Display modes
*/

#define TRUE 1
#define FALSE 0
#define YES 'Y'
#define NO 'N'
#define LEN_BYTE 1
#define LEN_SHORT 2
#define LEN_INT 4
#define LEN_FLOAT 4
#define LEN_DOUBLE 8

/* number of subwindows per window */
#define WINDOW_1_IMAGE 1
#define WINDOW_4_IMAGE 4
#define WINDOW_16_IMAGE 16
#define WINDOW_64_IMAGE 64

/* to identify the different window pattern */
/*small windows*/
#define W_PATTERN_0 0 /* 1 image */
#define W_PATTERN_1 1 /* 4 images */
#define W_PATTERN_2 2 /* 16 images */
#define W_PATTERN_3 3
/* full size windows */
#define W_PATTERN_4 4 /* 1 image */
#define W_PATTERN_5 5
#define W_PATTERN_6 6
#define W_PATTERN_7 7 /* 64 images */

#define NUM_SMALL_WINDOW_PATTERNS 4
#define NUM_WINDOW_PATTERNS 8

#define MAX_WINDOWS 2 /* number of small windows */
#define FULL_WINDOW 2 /* number of fullsize windows (one for each small one)*/
#define MAX_ROWS_PER_WINDOW 8
#define MAX_IMAGES_PER_WINDOW 96 /* not 100! */

/* callback identifier for different windows and pattern */
#define CALLBACK_FAKTOR_WINDOW 10000
#define CALLBACK_FAKTOR_PATTERN 1000
#define CALLBACK_W1_0 10000
#define CALLBACK_W1_1 11000
#define CALLBACK_W1_2 12000
#define CALLBACK_W1_3 13000
#define CALLBACK_W1_4 14000
#define CALLBACK_W1_5 15000
#define CALLBACK_W1_6 16000
#define CALLBACK_W1_7 17000

#define CALLBACK_W2_0 20000
#define CALLBACK_W2_1 21000
#define CALLBACK_W2_2 22000
#define CALLBACK_W2_3 23000
#define CALLBACK_W2_4 24000
#define CALLBACK_W2_5 25000
#define CALLBACK_W2_6 26000
#define CALLBACK_W2_7 27000

/* identifier for image types */
typedef enum
{
    NO_IMAGE = 0,
    CT_SLICE = 100,
    CT_SECTION = 110,
    MR_SLICE = 200,
    MR_SECTION = 210,
    CT_MR_SECTION = 250,
    XR_IMAGE = 300,
    AC_IMAGE = 400,
    /* CS 19.10.95 --> */
    ATLAS_SLICE = 405,
    PET_SLICE = 410,
    PET_SECTION = 415,
    /* CS 19.10.95 <-- */
    _3D_IMAGE_GEN = 420,
    /* CS 30.7.96 --> */
    ATLAS_3D_IMAGE = 430,
    FUNC_3D_IMAGE = 440,
    /* CS 30.7.96 <-- */
    SAG_SECTION = 500,
    COR_SECTION = 510,
    OBL_SECTION = 520,
    TRANS_SECTION = 530,
    PERP_SECTION = 540,
    PARAL_SECTION = 550,

    /* TR 03.02.97 --> */
    LINEDOSE_PARAL_SECTION = 555,
    LINEDOSE_PERP_SECTION = 556,
    /* TR 03.02.97 <-- */

    /* HAS 4.10.94 SRC 307-287 */
    ARC_PLANE_SECTION = 560,
    /* HAS 4.10.94 SRC 307-287 */

    /* HAS 5.6.96 --> */
    FUNC_SAG_SECTION = 570,
    FUNC_COR_SECTION = 580,
    FUNC_AXI_SECTION = 590,
    /* HAS 5.6.96 <-- */
    LOGO = 600,
    BEV_IMAGE = 800,
    D3_CT_TRIA_IMAGE = 810,
    D3_CT_VOL_RENDER_IMAGE = 830,
    D3_CT_VOL_RAY_TRACING_IMAGE = 840,
    D3_MR_TRIA_IMAGE = 811,
    D3_MR_VOL_RENDER_IMAGE = 831,
    D3_MR_VOL_RAY_TRACING_IMAGE = 841,

    /* DADO VR --> */
    VR_CT_IMAGE = 850,
    VR_MR_IMAGE = 851,
    /* DADO VR <-- */
    DOSVOLHIS = 900,
    TRANSFORMATION_DIAGRAM_IMAGE = 910,
    PRE_SCAN = 1000,
    SCAN_IMAGE = 1100,
    NO_OF_IMAGE_TYPES = 1100 /* please set this value, if you add
                                   new image types, used in 
                                   tri_surfaceTable.c */
} Image_type;

/* CS 23.10.95 --> */
#define SAG_SERIE 0
#define COR_SERIE 1
#define AXI_SERIE 2
/* CS 23.10.95 <-- */

#define CINE_AHEAD 1
#define CINE_BACK -1
/* AO 21.01.97 --> */
#define NO_CINE 0
#define CINE_ALTERFORW 2
#define CINE_ALTERBACK 3
/* AO 21.01.97 <-- */

/* CS 19.9.96 SCR 330b-993 --> 
#define MAX_CINE_SPEED 50
*/
#define MAX_CINE_SPEED 10
/* CS 19.9.96 SCR 330b-993 <-- */

/* Has 16.6.94 steht schon in iso_struct.h
#define MAX_ISODOSES 10
*/

/* to identify the graphics input device */
#define SCREEN 'S'
#define DIGI 'D'

/* hardcopy types */
#define POSTSCRIPT_PRINTER -1
#define POSTSCRIPT_FILE 0
#define IMAGE_FILE 1

/* to identify the actual program modus */
#define FREE 0
#define EDIT_VOI 1
#define EDIT_PLAN 2
#define HARDCOPY 3

/* identifier for work_proc */
#define WORK_HARDCOPY 10
#define DIGI_POINT_LAT 5
#define DIGI_POINT_FRON 6

#define MAX_ISOCENTRES 50
/*#define MAX_VOIS 20*/ /* defined also in voi_struct.h  rb 11.3.94 */
/*#define MAX_SUB_ROIS 10 */ /* not used at all rb 16.3.94 */
/*#define LEN_VOINAME 10*/ /*defined in voi_struct.h rb 16.3.94 */
#define MAX_ROI_POINTS 500
#define MAX_XR_ROI_POINTS 10000

#define MAX_POLY_POINTS 10000

#define MAX_SERIES 4 /* reduced from 10 t0 4 11.3.94 rb */
#define MAX_ATLAS_SERIES 3 /* CS 14.2.96 */

/* maximum lenght for patient identifier */
#define DIR_STRING_LEN 40
#define DIR_STRING_COUNT 100

#define W_SIZE 512 /* size of normal graphics window */

#define PIXEL_OF_SCREEN 1024 /* maximum size for graphics windows */

#define IMAGE_SIZE 1024 /* scaling for image coordinate system \
all image coordinates will run between 0 .. IMAGE_SIZE in x and y*/

/* mouse button definition */
typedef enum
{
    DIGI_BUT = 1,
    END_BUT = 2,
    DEL_BUT = 3
} Mouse_but;

/* if the cursor is not moved > DELTA 
the active graphic window will not be updated */
#define DELTA 0.1

/* definitions for work_proc identifiers */
#define WP_DIGI_ZOOM 1
#define WP_CINE 2
#define WP_MKM 3

#define VOI_COLORS 10

#define ISOCENTER_COLORS 10
#define ISODOSE_COLORS 10
#define STEREO_COLORS 10

#define ERROR_TEXT "d:"

/* digitalization modes */
typedef enum
{
    NO_DIGI_MODE = 0,
    DIGI_POLY = 1,
    FINISHED_POLY = 2,
    DELETE_POINTS = 3,
    START_AUTOCONTOURING = 4,
    CUT_POLY = 5,
    APPEND_POLY = 6,
    SHIFT_POINTS = 7,
    DIGI_STEREO_POINT = 8,
    DIGITIZE_LANDMARKS = 9,
    TRANSFORM_DIGI_ANGIO_LOCALIZER = 10,
    TRANSFORM_DIGI_CT_LOCALIZER = 11,
    TRANSFORM_3D_DISPLAY = 12,
    DIGI_PRESCAN = 13,
    DIGI_CUBE_AREA = 14,
    DIGI_CBI_POINT = 15,
    DIGI_MKM_POINT = 16,
    DOSE_MATRIX = 17,
    DIGI_SEED_POINT = 18,
    SBP_DOSE_MATRIX = 19,
    DIGI_DISTANCE = 20,
    DIGI_ENTRY_POINT = 21,
    DIGI_REGION_GROW_POINT = 22,
    DIGI_ATLAS_POINT = 23,
    ATLAS_3D_DISPLAY = 24,
    ATLAS_PAT3D_DISPLAY = 25,
    FUNC3D_DISPLAY = 26,
    DIGI_FUNC_POINT = 27,
    /* HKO SCR 913 */
    DIGI_SURFACE_RENDERING = 28,
    /* HKO SCR 913 */
    /* CS 24.10.96 --> */
    DIGI_FUSION_SQUARE = 29,
    /* CS 24.10.96 <-- */
    /* tr 22.01.97 --> */
    DIGI_LINE_DOSE = 30
    /* CS 22.01.97 <-- */
} Digi_mode;

/* program modes, indicates active popup menu */
typedef enum
{
    NO_MODE = 0,
    DIGI_VOI = 1,
    STEREO = 3,
    CBI = 4,
    SEED = 5,
    GEN_3D = 6,
    CONTOURS = 7,
    SCAN = 8,
    TRANSFORMATION_CT = 10,
    TRANSFORMATION_MR = 11,
    TRANSFORMATION_XR = 12,
    LANDMARK = 13,
    MKM_POINT = 14,
    PAT_SELECT_MODE = 15,
    PAT_BACKUP_MODE = 16,
    PAT_RESTORE_MODE = 17,
    PAT_DELETE_MODE = 18,
    PROFILE = 20
} Prog_mode;

#define NO_POINT -99999

#define LENGTH_VOI_FILE_HEADER 2048 /* increased from 1024 rb 16.3.94 */
#define LENGTH_VOI_DSCR 1024
#define LENGTH_VOI_BLOCK 256

#define LENGTH_SNP_FILE_HEADER 512
#define LENGTH_SNP_DSCR 256
#define LENGTH_SNP_BLOCK 256

#define LENGTH_SRP_FILE_HEADER 1024
#define LENGTH_SRP_DSCR 512
#define LENGTH_SRP_BLOCK 512

#define CT_TRANS_SER_HEAD_LENGTH 512
#define CT_TRANS_BLOCK_LENGTH 1024

#define XR_TRANS_SER_HEAD_LENGTH 512
#define XR_TRANS_BLOCK_LENGTH 2048

#define LENGTH_XR_SER_HEADER 512
#define LENGTH_XR_IM_HEADER 512
#define LENGTH_XR_IMAGE 512

/* file types */
#define CT_IMAGE_FILE 1
#define MR_IMAGE_FILE 2
#define XR_IMAGE_FILE 3

#define CT_VOI_FILE 11
#define MR_VOI_FILE 12
#define XR_VOI_FILE 13

#define CT_TRANS_FILE 21
#define MR_TRANS_FILE 22
#define XR_TRANS_FILE 23

#define STEREO_FILE 30
#define CBI_FILE 30

#define CBI_FILE_31 31 /* HAS 31.3.95 neues format der CBI pläne */

/* JP 18.8.95 --> */
#define HALF_LINE_WIDTH 60
/*#define HALF_LINE_WIDTH 256*/ /* image size in pixel for which (and also for the
		smaller) the line width for drawing lines will be halvened */
/* JP 18.8.95 <-- */

typedef enum
{
    NO_TRAFO_MODE = 0,
    AUTO_MODE = 1,
    MANU_MODE = 2,
    IMA_MODE = 3,
    SECT_MODE = 4
} Transf_type_mode;

/* CS 31.5.95 from section.c */
#define SECTION_SHIFT_SCALE 200
#define SECTION_SHIFT_FINE 20

#define SECTION_ANGLE_SCALE 90
#define SECTION_ANGLE_FINE 10
/* CS 31.5.95 end */

/* jp 16bit --> */
#define NINTENSITY 4096
#define NGRAYS_VR 256
/* jp 16bit <-- */

#endif
