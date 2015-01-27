/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*********************************************************************
 *
 *  Phylogentic Tree Converter Program for Tree3D System
 *  - converts phylogenetic tree file in Newick format into a
 *    3D Inventor (VRML1.0) model file
 *  - text processing only; does not require Inventor libraries
 *  - compile with: gcc -o treeconvert treeconvert.c
 *  - usage: treeconvert <newick_input_file> > <inventor_output_file>
 *
 *  * Written by Eric A. Wernert, UITS Advanced Visualization Lab,
 *    Indiana University, October 1998
 *
 *  Copyright 1998-2003  The Trustees of Indiana University.
 *  All rights reserved. (This software is licensed under IU's open
 *  source license agreement found in the file LICENSE.txt included
 *  with this distribution.)
 *
 *********************************************************************/

#include <string.h>
#include <stdio.h>
#include <util/coviseCompat.h>
#include <api/coModule.h>
#include <math.h>

#define MAXLEN 10000
#define DISTSCALE 1
#define YSPACING 0.02
#define FONTSIZE 10
#define NUMTAXA 200
#define NUMFIELDS 5
/* #define TEXT_SCALE 0.0015 */
#define TEXT_SCALE 0.0010
#define TEXT_SCALE_2D 0.05
#define CYLINDER_RADIUS 0.002
#define SPHERE_RADIUS 0.0080
#define STACKSIZE 500
#define NUMCOLORS 18
#define DEFAULT_LENGTH .05

extern char *readTreeFile(const char *fileName, int ignore_lengths, bool useColoring, bool addNames, bool lowRes, int hostID);
extern char *readTree(int ignore_lengths, bool useColoring, const char *buffer, bool addNames, bool lowRes, int hostID);

/* prototypes */
int readString(const char *filename, char *string);
char *findTreeStart(char *string);
char *readName(char *string, char *nameString);
char *readFloat(char *string, char *floatString);
void treeheader2iv(char *string);
void treeheader2iv_multiline(char *string);
void treeheader2ivFilename(char *string, char *filename);
void convertAndPrintColorString(char *str);

#define IVBUFLEN 4000
char *ivFileBuffer = NULL;
int ivAllocLen = 0;
int ivBufLen = 0;

void writeToBuf(const char *format, ...)
{
    char buffer[10002];
    va_list args;

    va_start(args, format);

#ifdef HAS_VSNPRINTF
    vsnprintf(buffer, 10000, format, args);
#else
    vsprintf(buffer, format, args);
#endif
    va_end(args);
    int len = strlen(buffer);
    while (ivBufLen + len + 1 > ivAllocLen)
    {
        char *oldbuf = ivFileBuffer;
        ivAllocLen += IVBUFLEN;
        ivFileBuffer = new char[ivAllocLen];
        memcpy(ivFileBuffer, oldbuf, ivBufLen + 1);
    }
    strcat(ivFileBuffer + ivBufLen, buffer);
    ivBufLen += len;
}

char *readTreeFile(const char *fileName, int ignore_lengths, bool useColoring, bool addNames, bool lowRes, int hostID)
{
    char string[MAXLEN];
    int strlength = readString(fileName, string);
    if (strlength < 0)
    {
        fprintf(stderr, "Error: failed to read %s\n", fileName);
        return NULL;
    }
    return readTree(ignore_lengths, useColoring, string, addNames, lowRes, hostID);
}

float DEFAULT_DEPTH = 0.05;

char *readTree(int ignore_lengths, bool useColoring, const char *string, bool addNames, bool lowRes, int hostID)
/**********************************************************************/
{
    //char string[MAXLEN];
    char nameString[16];
    char floatString[16];
    char tabString[100], tabString2[100];
    char colorfilename[128];
    char taxaname[NUMTAXA][12];
    char taxacolor[NUMTAXA][12];
    char taxadata[NUMTAXA][NUMFIELDS][12];
    char tmpstr[100], ch;
    char *treestart, *curr, *lastchar;
    int strlength, depth, depth2;
    int i, j, a, taxa, t, str_i, f, field;
    float distance, yoffset, start_yoffset, y1, y2, ymidpoint, yheight;
    float offsetStackL[STACKSIZE], offsetStackR[STACKSIZE];
    float distanceStack[STACKSIZE];
    int offsetLRindex[STACKSIZE];
    int offsetStackTop, distanceStackTop;
    int /*ignore_lengths, */ uniform_depth, z_coding, use_2d_text, use_colorfile;
    int id = 0;
    float node_depth;
    FILE *colorfile;

    float real_colors[NUMCOLORS][3] = {
        { 1.0, 0.5, 0.5 },
        { 0.83, 0.67, 0.5 },
        { 0.67, 0.83, 0.5 },
        { 0.5, 1.0, 0.5 },
        { 0.5, 0.83, 0.67 },
        { 0.5, 0.67, 0.83 },
        { 0.5, 0.5, 1.0 },
        { 0.67, 0.5, 0.83 },
        { 0.83, 0.5, 0.67 },

        { 1.0, 0.2, 0.2 },
        { 0.73, 0.47, 0.2 },
        { 0.47, 0.73, 0.2 },
        { 0.2, 1.0, 0.2 },
        { 0.2, 0.73, 0.47 },
        { 0.2, 0.47, 0.73 },
        { 0.2, 0.2, 1.0 },
        { 0.47, 0.2, 0.73 },
        { 0.73, 0.2, 0.47 },
    };

    float colors[NUMCOLORS][3] = {
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },

        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
        { .8, .8, .8 },
    };
    if (ivFileBuffer == NULL)
    {
        ivFileBuffer = new char[IVBUFLEN];
        ivAllocLen = IVBUFLEN;
    }
    ivBufLen = 0;
    ivFileBuffer[0] = '\0';

    /*if (argc < 2) {
     fprintf(stderr, "usage: convert <treefile> [-i|-u] [-z|-Z] [-c] [-r] [-2]\n");
     fprintf(stderr, "       -i = ignore branch lengths\n");
     fprintf(stderr, "       -u = draw leaf nodes with uniform depth\n");
     fprintf(stderr, "       -z = use z-coding for depth ('z' = + dir, 'Z' = - dir)\n");
     fprintf(stderr, "       -c = use depth coloring\n");
     fprintf(stderr, "       -r = reverse drawing from left to right\n");
     fprintf(stderr, "       -2 = use 2D text instead of 3D text\n");
     fprintf(stderr, "       -f <file> = use taxa coloring file\n");
    exit(-1);
   } */
    ignore_lengths = 0;
    uniform_depth = 0;
    z_coding = 0;
    use_2d_text = 1;
    use_colorfile = 1;
    /*
   for (a = 2; a < argc; a++) {
     if (!strcasecmp(argv[a], "-i")) {
       ignore_lengths = 1;
       fprintf(stderr, "ignoring branch lengths\n");
     }
     else if (!strcasecmp(argv[a], "-u")) {
       ignore_lengths = 0;
       uniform_depth = 1;
       fprintf(stderr, "drawing leaf nodes with uniform depth\n");
     }
   else if (!strcmp(argv[a], "-Z")) {
   z_coding = 1;
   fprintf(stderr, "using z-coding (with direction reversed) for depth\n");
   DEFAULT_DEPTH *= -1.0;
   }
   else if (!strcasecmp(argv[a], "-z")) {
   z_coding = 1;
   fprintf(stderr, "using z-coding for depth\n");
   }
   else if (!strcasecmp(argv[a], "-2")) {
   use_2d_text = 1;
   fprintf(stderr, "using 2D text instead of 3D text\n");
   }
   else if (!strcasecmp(argv[a], "-c")) {
   fprintf(stderr, "using depth coloring\n");
   for (i = 0; i < NUMCOLORS; i++)
   for (j = 0; j < 3; j++)
   colors[i][j] = real_colors[i][j];
   }
   else if (!strcasecmp(argv[a], "-f")) {
   a++; */
    char *covisedir = getenv("COVISEDIR");
#if 0
   if(covisedir) covisedir=strchr(covisedir,'=');
   if(covisedir) covisedir++;
#endif

    if (covisedir)
        snprintf(colorfilename, sizeof(colorfilename), "%s/%s", covisedir, "colors_sc03.dat");
    else
        strcpy(colorfilename, "colors_sc03.dat");
    fprintf(stderr, "using taxa coloring file <%s>\n", colorfilename);
    use_colorfile = 1;
    if ((colorfile = fopen(colorfilename, "r")) == NULL)
    {
        fprintf(stderr, "cannot open color file %s\n", colorfilename);
        exit(-1);
    }
    taxa = 0;
    str_i = 0;
    field = 0;
    /* the 67 is a hack as the system seems to miss the EOF marker */
    while (((ch = getc(colorfile)) != EOF) && (taxa < 67))
    {
        if ((ch == '\n') || (ch == ','))
        {
            tmpstr[str_i] = '\0';
            strcpy(taxadata[taxa][field], tmpstr);
            str_i = 0;
            field++;
            if (ch == '\n')
            {
                field = 0;
                taxa++;
            }
        }
        else
        {
            if (ch != ' ') /* igonre spaces at end of names */
            {
                if (ch == '/')
                    ch = ' '; /*change slashes to spaces */
                tmpstr[str_i] = ch;
                str_i++;
            }
        }
        //fprintf(stderr,"<[%c],%d,%d,%d> ", ch, taxa, field, str_i);
    }
    fclose(colorfile);
    for (t = 0; t < taxa; t++)
    {
        for (f = 0; f < NUMFIELDS; f++)
        {
            fprintf(stderr, "<%s>", taxadata[t][f]);
        }
        fprintf(stderr, "\n");
    }
    /*  }
       else {
         fprintf(stderr, "unknown option: [%s]\n", argv[a]);
       }
     }*/

    offsetStackTop = -1;
    distanceStackTop = -1;

    if ((strlength = strlen(string)) <= 0)
    {
        fprintf(stderr, "Error: failed to read Tree from string\n");
        return NULL;
    }

    /*
     fprintf(stderr, "strlength = %d, strlen = %d\n", strlength, strlen(string));
     writeToBuf( "String = \n%s\n", string);
     writeToBuf( "*************************************************\n");
   */

    treestart = findTreeStart((char *)string);
    /*fprintf(stderr, "Data = \n%s...\n", treestart); */

    depth = 0;
    depth2 = 0;
    yoffset = 0;
    writeToBuf("#Inventor V2.1 ascii\n");
    writeToBuf("# ID %d\n", hostID);
    //srand(time(NULL));
    //writeToBuf( "# ID %d\n", 1+(int) (10.0*rand()/(RAND_MAX+1.0)));
    writeToBuf("# %s\n", string);
    /*writeToBuf( "SoFont { name \"Helvetica\"  size %d }\n\n", FONTSIZE);*/
    writeToBuf("SoFont {\n");
    writeToBuf("   size %d  \n", FONTSIZE);
    writeToBuf("   name \"Helvetica-Bold\"\n");
    writeToBuf("}\n\n");

    treeheader2iv_multiline((char *)string);
    yoffset += 2 * YSPACING;

    /* parse through data portion of string backwards */
    curr = (char *)&string[strlength - 1];

    while (curr >= string && *curr && *curr != ']')
    {
        switch (*curr)
        {
        case '(':
            depth--;
            if (offsetStackTop == -1)
            {
                fprintf(stderr, "Error: encountered empty stack!\n");
                return NULL;
            }
            y1 = offsetStackL[offsetStackTop];
            y2 = offsetStackR[offsetStackTop];
            ymidpoint = (y1 + y2) / 2.0;
            yheight = fabs(y2 - y1);
            offsetStackTop--;
            /*
            fprintf(stderr,"y1 = %f, y1 = %f, midpoint = %f, height = %f\n",
               y1, y2, ymidpoint, yheight);
               */

            tabString[0] = '\0';
            for (i = 0; i < (depth + depth2); i++)
                strcat(tabString, "\t");
            /* output transform and material */
            writeToBuf("%sSeparator {\n", tabString);
            writeToBuf("%s\tMaterial {diffuseColor %f %f %f}\n", tabString,
                       colors[(depth + 1) % NUMCOLORS][0],
                       colors[(depth + 1) % NUMCOLORS][1],
                       colors[(depth + 1) % NUMCOLORS][2]);
            if (z_coding == 1)
            {
                node_depth = DEFAULT_DEPTH * depth;
            }
            else
            {
                node_depth = 0;
            }
            writeToBuf("%s\tDEF V_CONNECTOR_XFORM Transform {translation\t %f %f %f }\n",
                       tabString, 0.0, ymidpoint, node_depth);

            /* and connector node */
            writeToBuf("%s\tSphere { radius %f } #connector\n",
                       tabString, SPHERE_RADIUS);

            /* transform for the vertical cylinder */
            writeToBuf("%s\tDEF V_CYL_XFORM Transform { scaleFactor %f %f %f }\n", tabString, 1.0, yheight * 0.5, 1.0);
            /* output  vertical cylinder  */
            writeToBuf("%s\tDEF V_CYL Cylinder {\n", tabString);
            writeToBuf("%s\t\tradius\t %f\n", tabString, CYLINDER_RADIUS);
            /* writeToBuf( "%s\t\theight\t %f\n", tabString, yheight); */
            writeToBuf("%s\t}\n", tabString);

            /* close off separator */
            writeToBuf("%s}\n", tabString);

            /* output horizontal connecting cylinder */
            if (distanceStackTop == -1)
            {
                fprintf(stderr, "Encountered empty distance stack!\n");
                return NULL;
            }
            distance = distanceStack[distanceStackTop];
            distanceStackTop--;
            writeToBuf("%sSeparator {\n", tabString);
            writeToBuf("%s\tMaterial {diffuseColor %f %f %f}\n", tabString,
                       colors[depth % NUMCOLORS][0],
                       colors[depth % NUMCOLORS][1],
                       colors[depth % NUMCOLORS][2]);

            if (z_coding == 1)
            {
                node_depth = DEFAULT_DEPTH * depth;
            }
            else
            {
                node_depth = 0;
            }
            writeToBuf("%s\tSeparator {\n", tabString);
            writeToBuf("%s\t\tTransform {\n", tabString, id++);
            writeToBuf("%s\t\t\trotation\t 0 0 1  1.5707963\n", tabString);
            writeToBuf("%s\t\t\ttranslation\t %f %f %f\n",
                       tabString, -distance * DISTSCALE / 2.0, ymidpoint, node_depth);
            writeToBuf("%s\t\t}\n", tabString);
            writeToBuf("%s\t\tDEF H_CYL Cylinder { \n", tabString);
            writeToBuf("%s\t\t\tradius\t %f\n", tabString, CYLINDER_RADIUS);
            writeToBuf("%s\t\t\theight\t %f\n", tabString, distance * DISTSCALE);
            writeToBuf("%s\t\t}\n", tabString);
            writeToBuf("%s\t}\n", tabString);

            if (z_coding) /* add on the little cylinders in the z direction */
            {
                writeToBuf("%s\tSeparator {\n", tabString);
                writeToBuf("%s\t\tDEF Z_CYL_XFORM Transform { \n", tabString);
                writeToBuf("%s\t\t\tscaleFactor %f %f %f \n", tabString, 1.0, DEFAULT_DEPTH * 0.5, 1.0);
                writeToBuf("%s\t\t\trotation 1 0 0 %f\n", tabString, M_PI * 0.5);
                writeToBuf("%s\t\t\ttranslation %f %f %f \n", tabString,
                           -distance * DISTSCALE, ymidpoint, node_depth - (DEFAULT_DEPTH * 0.5));
                writeToBuf("%s\t\t}\n", tabString);
                writeToBuf("%s\t\tDEF Z_CYL_INTERNAL Cylinder {radius %f}\n", tabString, CYLINDER_RADIUS);
                writeToBuf("%s\t}\n", tabString);
            }

            writeToBuf("%s}\n", tabString);

            /*close off SoSeparator*/
            writeToBuf("%s}\n", tabString);

            if (offsetLRindex[offsetStackTop] == 0)
                offsetStackL[offsetStackTop] = ymidpoint;
            else
                offsetStackR[offsetStackTop] = ymidpoint;

            curr--;
            break;

        case ')':
            /*
            for (i = 0; i < depth; i++) writeToBuf( "\t");
            writeToBuf( "%s\t\tMaterial {diffuseColor %f %f %f}\n", tabString,
               colors[depth%NUMCOLORS][0],
               colors[depth%NUMCOLORS][1],
               colors[depth%NUMCOLORS][2]);
            writeToBuf( "Separator {\n");
             */
            offsetStackTop++;
            offsetLRindex[offsetStackTop] = 0;
            offsetStackL[offsetStackTop] = -1;
            offsetStackR[offsetStackTop] = -1;
            distanceStackTop++;
            distanceStack[distanceStackTop] = distance;
            depth++;
            depth2--;
            curr--;
            break;

        case '\'':
            curr = readName(curr, nameString);
            if (use_colorfile)
            {
                fprintf(stderr, "searching for string <%s>\n", nameString);
                t = 0;
                while (strcasecmp(nameString, taxadata[t][0]) && (t < taxa))
                    t++;
                if (t < taxa)
                {
                    fprintf(stderr, "matched string <%s> with <%s>\n",
                            nameString, taxadata[t][0]);
                }
                else
                {
                    t = -1;
                    fprintf(stderr, "could not match string <%s> \n", nameString);
                }
            }

            tabString[0] = '\0';
            for (i = 0; i < (depth + depth2); i++)
                strcat(tabString, "\t");
            writeToBuf("%sSeparator { #leaf node\n", tabString);
            if (z_coding == 1)
            {
                node_depth = DEFAULT_DEPTH * depth;
            }
            else
            {
                node_depth = 0;
            }
            writeToBuf("%s\tDEF LEAF_XFORM Transform { translation\t %f %f %f}\n",
                       tabString, 0.0, yoffset, node_depth);
            if (offsetLRindex[offsetStackTop] == 0)
                offsetStackL[offsetStackTop] = yoffset;
            else
                offsetStackR[offsetStackTop] = yoffset;
            yoffset += YSPACING;
            /* the sphere */
            if (use_colorfile)
            {
                writeToBuf("%s\tMaterial {diffuseColor ", tabString);
                /*fprintf(stderr, "<t = %d>\n", t);*/
                if (t != -1)
                    convertAndPrintColorString(taxadata[t][3]);
                else
                    writeToBuf(".8 .8 .8");
                writeToBuf("}\n");
            }
            else
            {
                writeToBuf("%s\tMaterial {diffuseColor %f %f %f}\n",
                           tabString,
                           colors[depth % NUMCOLORS][0],
                           colors[depth % NUMCOLORS][1],
                           colors[depth % NUMCOLORS][2]);
            }
            writeToBuf("%s\tDEF %s Sphere { radius %f }\n",
                       tabString, nameString, SPHERE_RADIUS);
            /* the cylinder */
            writeToBuf("%s\tSeparator {\n", tabString);
            writeToBuf("%s\t\tTransform {\n", tabString);
            writeToBuf("%s\t\t\trotation\t 0 0 1  1.5707963\n", tabString);
            writeToBuf("%s\t\t\ttranslation\t %f %f %f\n",
                       tabString, -distance * DISTSCALE / 2.0, 0.0, 0.0);
            writeToBuf("%s\t\t}\n", tabString);
            writeToBuf("%s\t\tDEF H_CYL Cylinder {\n", tabString);
            writeToBuf("%s\t\t\tradius\t %f\n", tabString, CYLINDER_RADIUS);
            writeToBuf("%s\t\t\theight\t %f\n", tabString, distance * DISTSCALE);
            writeToBuf("%s\t\t}\n", tabString);
            writeToBuf("%s\t}\n", tabString);
            if (z_coding) /* add on the little cylinders in the z direction */
            {
                writeToBuf("%s\tSeparator {\n", tabString);
                writeToBuf("%s\t\tDEF Z_CYL_XFORM Transform { \n", tabString);
                writeToBuf("%s\t\t\tscaleFactor %f %f %f \n", tabString, 1.0, DEFAULT_DEPTH * 0.5, 1.0);
                writeToBuf("%s\t\t\trotation 1 0 0 %f\n", tabString, M_PI * 0.5);
                writeToBuf("%s\t\t\ttranslation %f %f %f \n", tabString,
                           -distance * DISTSCALE, 0.0, -DEFAULT_DEPTH * 0.5);
                writeToBuf("%s\t\t}\n", tabString);
                writeToBuf("%s\t\tDEF Z_CYL_LEAF Cylinder {radius %f}\n", tabString, CYLINDER_RADIUS);
                writeToBuf("%s\t}\n", tabString);
            }

            /* the text */
            if (addNames)
            {
                if (use_2d_text)
                {
                    writeToBuf("%s\tSeparator { \n", tabString);
                    writeToBuf("%s\t\tTransform {\n", tabString);
                    writeToBuf("%s\t\t\ttranslation\t %f %f %f\n",
                               tabString, 2 * SPHERE_RADIUS, 0.0, 0.0);
                    writeToBuf("%s\t\t\tscaleFactor\t %f %f %f\n",
                               tabString, TEXT_SCALE_2D, TEXT_SCALE_2D, TEXT_SCALE_2D);
                    writeToBuf("%s\t\t}\n", tabString);
                    writeToBuf("%s\t\tText2 { string \"%s\" }\n",
                               tabString, nameString);
                    writeToBuf("%s\t} \n", tabString);
                }
                else
                {
                    writeToBuf("%s\tSeparator { \n", tabString);
                    writeToBuf("%s\t\tTransform {\n", tabString);
                    writeToBuf("%s\t\t\ttranslation\t %f %f %f\n",
                               tabString, 2 * SPHERE_RADIUS, 0.0, 0.0);
                    writeToBuf("%s\t\t\tscaleFactor\t %f %f %f\n",
                               tabString, TEXT_SCALE, TEXT_SCALE, TEXT_SCALE);
                    writeToBuf("%s\t\t}\n", tabString);
                    writeToBuf("%s\t\tText3 { string \"%s\" }\n",
                               tabString, nameString);
                    writeToBuf("%s\t} \n", tabString);
                }
            }

            /* close the separators */
            writeToBuf("%s} #end leaf node \n", tabString);
            tabString2[0] = '\0';

            for (i = 0; i < (depth + depth2) - 1; i++)
                strcat(tabString2, "\t");
            writeToBuf("%s} #end horizontal length \n", tabString2);
            //depth2--;
            //if (depth2 < 0) fprintf(stderr, "Error: depth2 = %d\n", depth2);

            break;

        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            curr = readFloat(curr, floatString);
            if (ignore_lengths)
            {
                distance = DEFAULT_LENGTH;
            }
            else
            {
                sscanf(floatString, "%f", &distance);
            }
            for (i = 0; i < (depth + depth2); i++)
                writeToBuf("\t");
            writeToBuf("Separator { #horizontal length\n");
            for (i = 0; i < (depth + depth2); i++)
                writeToBuf("\t");
            writeToBuf("\tDEF H_OFFSET_XFORM Transform { translation\t %f %f %f }\n",
                       distance * DISTSCALE, 0.0, 0.0);
            //depth2++;
            break;

        case ',':
            /* comma separates left & right branches */
            offsetLRindex[offsetStackTop] += 1;
            curr--;
            break;

        case ' ':
        case ':':
        case '\n':
            /* do nothing with colons, spaces, or newlines */
            curr--;
            break;
        default:
            /*fprintf(stderr, "unknown character: [%c]\n", *curr); */
            curr--;
            break;
        }
    }
    return ivFileBuffer;
}

/**********************************************************************/

char *readName(char *string, char *nameString)
{
    /* read from string until closing quote is encountered */
    char *ptr = string, *returnptr;
    int i = 0;

    ptr--; /* step past ending quote */
    while (*ptr != '\'')
        ptr--;
    returnptr = ptr;
    returnptr--;

    ptr++; /* step past starting quote */
    while (*ptr != '\'')
    {
        if (*ptr == ' ')
            nameString[i] = '_';
        else
            nameString[i] = *ptr;
        i++;
        ptr++;
    }
    ptr++; /* step past ending quote */
    nameString[i] = '\0'; /* tidy up string */
    return returnptr;
}

/**********************************************************************/

char *readFloat(char *string, char *floatString)
{
    /* read from string backwards until space is  encountered */
    char *ptr = string, *returnptr;
    int i = 0;

    while ((*ptr != ' ') && (*ptr != ':'))
        ptr--;
    returnptr = ptr;

    ptr++; /* step past space which marks start of number */
    while ((*ptr != ',') && (*ptr != ')') && (*ptr != ';'))
    {
        floatString[i] = *ptr;
        i++;
        ptr++;
    }
    floatString[i] = '\0'; /* tidy up string */
    return returnptr;
}

/**********************************************************************/

char *findTreeStart(char *string)
{
    /* return pointer to first character after comment in square brackets */
    char *ptr = string;
    while ((*ptr != '\0') && (*ptr != ']'))
        ptr++;
    ptr++;
    return ptr;
}

/**********************************************************************/
int readString(const char *filename, char *string)
{
    int ch;
    int i;

    FILE *infile;

    if ((infile = fopen(filename, "r")) == NULL)
    {
        fprintf(stderr, "cannot open file %s\n", filename);
        return -1;
    }

    i = 0;
    while ((ch = getc(infile)) != EOF)
    {
        string[i] = ch;
        i++;
    }
    string[i] = '\0';

    fclose(infile);

    return (i);
}

/**********************************************************************/

void treeheader2iv(char *string)
{
    char header[256];
    char *tabString = "";
    char *curr = string;
    int i = 0;

    curr += 3; /* step past opening '[&&' */
    while (i < 256 && *curr != '\0' && *curr != ']')
    {
        header[i] = *curr;
        i++;
        curr++;
    }
    if (i < 256)
        header[i] = '\0';
    else
    {
        strcpy(header, "no valid header");
    }

    writeToBuf("%sSeparator {\n", tabString);

    writeToBuf("%s\tTransform {\n", tabString);
    writeToBuf("%s\t\ttranslation\t %f %f %f\n",
               tabString, 2 * SPHERE_RADIUS, 0.0, 0.0);
    writeToBuf("%s\t\tscaleFactor\t %f %f %f\n",
               tabString, TEXT_SCALE * 1.0, TEXT_SCALE * 1.0, TEXT_SCALE * 1.0);
    writeToBuf("%s\t}\n", tabString);
    writeToBuf("%s\tMaterial {diffuseColor %f %f %f}\n",
               tabString, .95, .80, .2);
    writeToBuf("%s\tText3 { string \"%s\" }\n", tabString, header);

    writeToBuf("%s}\n", tabString);
}

/**********************************************************************/

void treeheader2ivFilename(char *string, char *filename)
{
    char header[256];
    char *tabString = "";
    char *curr = string;
    int i = 0;

    curr += 3; /* step past opening '[&&' */

    while (*curr != ']')
    {
        header[i] = *curr;
        i++;
        curr++;
    }
    header[i] = '\0';

    writeToBuf("%sSeparator {\n", tabString);

    writeToBuf("%s\tTransform {\n", tabString);
    writeToBuf("%s\t\ttranslation\t %f %f %f\n",
               tabString, 2 * SPHERE_RADIUS, 0.0, 0.0);
    writeToBuf("%s\t\tscaleFactor\t %f %f %f\n",
               tabString, TEXT_SCALE * 1.0, TEXT_SCALE * 1.0, TEXT_SCALE * 1.0);
    writeToBuf("%s\t}\n", tabString);
    writeToBuf("%s\tMaterial {diffuseColor %f %f %f}\n",
               tabString, 1.0, 0.8, 0.0);
    writeToBuf("%s\tText3 { string \"%s\" }\n", tabString, header);

    writeToBuf("%s\tTransform {\n", tabString);
    writeToBuf("%s\t\ttranslation\t %f %f %f\n",
               tabString, -2 * SPHERE_RADIUS, -7000 * TEXT_SCALE, 0.0);
    writeToBuf("%s\t\tscaleFactor\t %f %f %f\n",
               tabString, 1.5, 1.5, 1.5);
    writeToBuf("%s\t}\n", tabString);
    writeToBuf("%s\tMaterial {diffuseColor %f %f %f}\n",
               tabString, 0.5, 1.0, 1.0);
    writeToBuf("%s\tText3 { string \"%s\" }\n", tabString, filename);

    writeToBuf("%s}\n", tabString);
}

/**********************************************************************/

void convertAndPrintColorString(char *str)
{
    int i, l, t;
    float f;
    char tmp[12];
    //fprintf(stderr, "converting string <%s>\n", str);
    l = strlen(str) + 1;
    i = t = 0;
    while (i < l)
    {
        if ((str[i] == ' ') || (str[i] == '\0'))
        {
            tmp[t] = '\0';
            f = atof(tmp);
            writeToBuf("%f ", f / 255.0);
            t = 0;
        }
        else
        {
            tmp[t] = str[i];
            t++;
        }
        i++;
    }
}

/**********************************************************************/

void treeheader2iv_multiline(char *string)
{
    char header[10][256];
    char *tabString = "";
    char *curr = string;
    int i = 0, j = 0;

    curr += 3; /* step past opening '[&&' */

    while (*curr != ']')
    {
        if (*curr == ',')
        {
            header[j][i] = '\0';
            j++;
            i = 0;
        }
        else
        {
            header[j][i] = *curr;
            i++;
        }
        curr++;
    }
    header[j][i] = '\0';

    writeToBuf("%sSeparator {\n", tabString);
    writeToBuf("%s\tTransform {\n", tabString);
    writeToBuf("%s\t\ttranslation\t %f %f %f\n",
               tabString, 10 * SPHERE_RADIUS, 0.0, 0.0);
    writeToBuf("%s\t\tscaleFactor\t %f %f %f\n",
               tabString, TEXT_SCALE * 1.5, TEXT_SCALE * 1.5, TEXT_SCALE * 1.5);
    writeToBuf("%s\t}\n", tabString);
    writeToBuf("%s\tMaterial {diffuseColor %f %f %f}\n",
               tabString, 1.0, 0.8, 0.0);
    writeToBuf("%s\tText3 { string [\n", tabString);
    for (i = 0; i <= j; i++)
        writeToBuf("%s\t\t\t\"%s\",\n", tabString, header[i]);
    writeToBuf("%s\t\t] }\n", tabString);
    writeToBuf("%s}", tabString);
}
