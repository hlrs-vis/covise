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
#define YSPACING 0.02f
#define FONTSIZE 10
/* #define TEXT_SCALE 0.0015 */
#define TEXT_SCALE 0.0010f
#define CYLINDER_RADIUS 0.002f
#define SPHERE_RADIUS 0.0080f
#define STACKSIZE 500
#define MYNUMCOLORS 18
#define DEFAULT_LENGTH .05f

/* prototypes */
int readString(const char *filename, char *string);
char *findTreeStart(char *string);
char *readName(char *string, char *nameString);
char *readFloat(char *string, char *floatString);
void treeheader2iv(char *string);

#define IVBUFLEN 4000
char *ivFileBuffer = NULL;
size_t ivAllocLen = 0;
size_t ivBufLen = 0;

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
    size_t len = strlen(buffer);
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

/**********************************************************************/
char *readTree(const char *fileName, int ignore_lengths, bool useColoring)
{

    char string[MAXLEN];
    char nameString[16];
    char floatString[16];
    char tabString[100];
    char tabString2[100];
    char *treestart, *curr; //, *lastchar;s
    int strlength, depth, depth2;
    int i, j; //,a;
    float distance, yoffset, y1, y2, ymidpoint, yheight; //, start_yoffset
    float offsetStackL[STACKSIZE], offsetStackR[STACKSIZE];
    float distanceStack[STACKSIZE];
    int offsetLRindex[STACKSIZE];
    int offsetStackTop, distanceStackTop;
    int id = 0;

    float real_colors[MYNUMCOLORS][3] = {
        { 0.8f, 0.5f, 0.5f },
        { 0.63f, 0.67f, 0.5f },
        { 0.47f, 0.83f, 0.5f },
        { 0.3f, 1.0f, 0.5f },
        { 0.3f, 0.83f, 0.67f },
        { 0.3f, 0.67f, 0.83f },
        { 0.3f, 0.5f, 1.0f },
        { 0.47f, 0.5f, 0.83f },
        { 0.63f, 0.5f, 0.67f },

        { 0.8f, 0.2f, 0.2f },
        { 0.53f, 0.47f, 0.2f },
        { 0.27f, 0.73f, 0.2f },
        { 0.0f, 1.0f, 0.2f },
        { 0.0f, 0.73f, 0.47f },
        { 0.0f, 0.47f, 0.73f },
        { 0.0f, 0.2f, 1.0f },
        { 0.27f, 0.2f, 0.73f },
        { 0.53f, 0.2f, 0.47f },
    };

    float colors[MYNUMCOLORS][3] = {
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },

        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
        { .8f, .8f, .8f },
    };
    if (ivFileBuffer == NULL)
    {
        ivFileBuffer = new char[IVBUFLEN];
        ivAllocLen = IVBUFLEN;
    }
    ivBufLen = 0;
    ivFileBuffer[0] = '\0';

    if (useColoring)
    {
        for (i = 0; i < MYNUMCOLORS; i++)
            for (j = 0; j < 3; j++)
                colors[i][j] = real_colors[i][j];
    }

    /*if ((argc < 2) || (argc > 4)) {
     fprintf(stderr, "usage: convert <treefile> [-i] [-c]\n");
     fprintf(stderr, "       -i = ignore branch lengths\n");
     fprintf(stderr, "       -c = use depth coloring\n");
     return NULL;
   }
   ignore_lengths = 0;
   for (a = 2; a < argc; a++) {
     if (!strcasecmp(argv[a], "-i")) {
       ignore_lengths = 1;
       fprintf(stderr, "ignoring branch lengths\n");
   }
   else if (!strcasecmp(argv[a], "-c")) {
   fprintf(stderr, "using depth coloring\n");
   for (i = 0; i < MYNUMCOLORS; i++)
   for (j = 0; j < 3; j++)
   colors[i][j] = real_colors[i][j];
   }
   else {
   fprintf(stderr, "unknown option: [%s]\n", argv[a]);
   }
   }*/

    offsetStackTop = -1;
    distanceStackTop = -1;
    strlength = readString(fileName, string);
    if (strlength < 0)
    {
        fprintf(stderr, "Error: failed to read %s\n", fileName);
        return NULL;
    }

    /*
     fprintf(stderr, "strlength = %d, strlen = %d\n", strlength, strlen(string));
     writeToBuf( "String = \n%s\n", string);
     writeToBuf( "*************************************************\n");
   */

    treestart = findTreeStart(string);
    /*fprintf(stderr, "Data = \n%s...\n", treestart); */

    depth = 0;
    depth2 = 0;
    yoffset = 0;
    writeToBuf("#Inventor V2.1 ascii\n");
    writeToBuf("# %s\n", string);
    /*writeToBuf( "SoFont { name \"Helvetica\"  size %d }\n\n", FONTSIZE);*/
    writeToBuf("SoFont {\n");
    writeToBuf("   size %d  \n", FONTSIZE);
    writeToBuf("   name \"Helvetica-Bold\"\n");
    writeToBuf("}\n\n");

    treeheader2iv(string);
    yoffset += 2 * YSPACING;

    /* parse through data portion of string backwards */
    curr = &string[strlength];

    while (*curr != ']')
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
            ymidpoint = (y1 + y2) / 2.0f;
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
                       colors[(depth + 1) % MYNUMCOLORS][0],
                       colors[(depth + 1) % MYNUMCOLORS][1],
                       colors[(depth + 1) % MYNUMCOLORS][2]);
            writeToBuf("%s\tDEF V_CONNECTOR_XFORM Transform {translation\t %f %f %f }\n",
                       tabString, 0.0, ymidpoint, 0.0);

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
                       colors[depth % MYNUMCOLORS][0],
                       colors[depth % MYNUMCOLORS][1],
                       colors[depth % MYNUMCOLORS][2]);
            writeToBuf("%s\tTransform {\n", tabString, id++);
            writeToBuf("%s\t\trotation\t 0 0 1  1.5707963\n", tabString);
            writeToBuf("%s\t\ttranslation\t %f %f %f\n",
                       tabString, -distance * DISTSCALE / 2.0, ymidpoint, 0.0);
            writeToBuf("%s\t}\n", tabString);
            writeToBuf("%s\tDEF H_CYL Cylinder { \n", tabString);
            writeToBuf("%s\t\tradius\t %f\n", tabString, CYLINDER_RADIUS);
            writeToBuf("%s\t\theight\t %f\n", tabString, distance * DISTSCALE);
            writeToBuf("%s\t}\n", tabString);
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
               colors[depth%MYNUMCOLORS][0],
               colors[depth%MYNUMCOLORS][1],
               colors[depth%MYNUMCOLORS][2]);
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
            tabString[0] = '\0';
            for (i = 0; i < (depth + depth2); i++)
                strcat(tabString, "\t");
            writeToBuf("%sSeparator { #leaf node\n", tabString);
            writeToBuf("%s\tDEF LEAF_XFORM Transform { translation\t %f %f %f }\n",
                       tabString, 0.0, yoffset, 0.0);
            if (offsetLRindex[offsetStackTop] == 0)
                offsetStackL[offsetStackTop] = yoffset;
            else
                offsetStackR[offsetStackTop] = yoffset;
            yoffset += YSPACING;
            /* the sphere */
            writeToBuf("%s\tMaterial {diffuseColor %f %f %f}\n", tabString,
                       colors[depth % MYNUMCOLORS][0],
                       colors[depth % MYNUMCOLORS][1],
                       colors[depth % MYNUMCOLORS][2]);
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
            /* the text */
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
    const char *tabString = "";
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
               tabString, .95, .80, .2);
    writeToBuf("%s\tText3 { string \"%s\" }\n", tabString, header);

    writeToBuf("%s}\n", tabString);
}

/**********************************************************************/

void treeheader2ivFilename(char *string, char *filename)
{
    char header[256];
    const char *tabString = "";
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

void treeheader2iv_multiline(char *string)
{
    char header[10][256];
    const char *tabString = "";
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
