/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CoCTypes.h"
#include "CoCUtil.h"

#include "RUmacro.h"
#include "v2e_macros.h"
#include "v2e_util.h"

const int TYPE_HEXAGON = 7;
const int TYPE_HEXAEDER = 7;
const int TYPE_PRISM = 6;
const int TYPE_PYRAMID = 5;
const int TYPE_TETRAHEDER = 4;
const int TYPE_QUAD = 3;
const int TYPE_TRIANGLE = 2;
const int TYPE_BAR = 1;
const int TYPE_NONE = 0;
const int TYPE_POINT = 10;

static char COVISE_directory[MAXLINE];

/*-------------------------------------------------------------------
  Initialize a CoC_UNSGRD struct (set all entries to zero or NULL)
  -------------------------------------------------------------------*/
void
initUNSGRD(CoC_UNSGRD *grd)
{
    if (grd)
    {
        grd->numElem = 0;
        grd->numCoords = 0;
        grd->numConn = 0;
        grd->elem_l = NULL;
        grd->type_l = NULL;
        grd->conn_l = NULL;
        grd->x_coord = NULL;
        grd->y_coord = NULL;
        grd->z_coord = NULL;
    }
}

void
reorientUNSGRD(CoC_UNSGRD *grd)
{
    int i, j, k;
    int beg, end;
    int idx;
    int cl[21];

    float x[10]; /* we're on the safe side */
    float y[10];
    float z[10];

    float xt[4];
    float yt[4];
    float zt[4];

    if (grd)
    {
        for (i = 0; i < grd->numElem; ++i)
        {
            beg = grd->elem_l[i];
            if (i < grd->numElem - 1)
            {
                end = grd->elem_l[i + 1];
            }
            else
            {
                end = grd->numElem;
            }

            /* copy the connection-list for current element*/
            k = 0;
            for (j = beg; j < end; ++j)
            {
                cl[k] = grd->conn_l[j];
                k++;
            }

            k = 0;
            for (j = beg; j < end; ++j)
            {
                idx = grd->conn_l[j];
                x[k] = grd->x_coord[idx];
                y[k] = grd->y_coord[idx];
                z[k] = grd->z_coord[idx];
                k++;
            }

            switch (grd->type_l[i])
            {

            case 4:
                /* get tetra's right */
                if (!tetraVol(x, y, z))
                {
                    transposeIdx(cl, 4, 1, 2);
                }
                break;

            case 5:
                /* a pyramid is dissected into 2 tetras */
                xt[0] = x[0];
                yt[0] = y[0];
                zt[0] = z[0];
                xt[1] = x[1];
                yt[1] = y[1];
                zt[1] = z[1];
                xt[2] = x[3];
                yt[2] = y[3];
                zt[2] = z[3];
                xt[3] = x[4];
                yt[3] = y[4];
                zt[3] = z[4];
                if (!tetraVol(xt, yt, zt))
                {
                    transposeIdx(cl, 5, 1, 3);
                    transposeFld(x, 5, 1, 3);
                    transposeFld(y, 5, 1, 3);
                    transposeFld(z, 5, 1, 3);
                }

                xt[0] = x[1];
                yt[0] = y[1];
                zt[0] = z[1];
                xt[1] = x[2];
                yt[1] = y[2];
                zt[1] = z[2];
                xt[2] = x[3];
                yt[2] = y[3];
                zt[2] = z[3];
                xt[3] = x[4];
                yt[3] = y[4];
                zt[3] = z[4];
                if (!tetraVol(xt, yt, zt))
                {
                    transposeIdx(cl, 5, 1, 3);
                    transposeFld(x, 5, 1, 3);
                    transposeFld(y, 5, 1, 3);
                    transposeFld(z, 5, 1, 3);
                }
                break;

            case 7:
                /* a hexahedron is dissected into 6 tetras */
                xt[0] = x[4];
                yt[0] = y[4];
                zt[0] = z[4];
                xt[1] = x[7];
                yt[1] = y[7];
                zt[1] = z[7];
                xt[2] = x[5];
                yt[2] = y[5];
                zt[2] = z[5];
                xt[3] = x[0];
                yt[3] = y[0];
                zt[3] = z[0];
                if (!tetraVol(xt, yt, zt))
                {
                    /* 		    fprintf(stderr, " transpose1 %d\n", i); */
                    transposeIdx(cl, 8, 5, 7);
                    transposeFld(x, 8, 5, 7);
                    transposeFld(y, 8, 5, 7);
                    transposeFld(z, 8, 5, 7);
                }

                xt[0] = x[1];
                yt[0] = y[1];
                zt[0] = z[1];
                xt[1] = x[6];
                yt[1] = y[6];
                zt[1] = z[6];
                xt[2] = x[2];
                yt[2] = y[2];
                zt[2] = z[2];
                xt[3] = x[3];
                yt[3] = y[3];
                zt[3] = z[3];
                if (!tetraVol(xt, yt, zt))
                {
                    /* 		    fprintf(stderr, " transpose2 %d\n",i); */
                    transposeIdx(cl, 8, 1, 6);
                    transposeFld(x, 8, 1, 6);
                    transposeFld(y, 8, 1, 6);
                    transposeFld(z, 8, 1, 6);
                }

                xt[0] = x[0];
                yt[0] = y[0];
                zt[0] = z[0];
                xt[1] = x[1];
                yt[1] = y[1];
                zt[1] = z[1];
                xt[2] = x[3];
                yt[2] = y[3];
                zt[2] = z[3];
                xt[3] = x[5];
                yt[3] = y[5];
                zt[3] = z[5];
                if (!tetraVol(xt, yt, zt))
                {
                    /* 		    fprintf(stderr, " transpose3 %d\n",i); */
                    transposeIdx(cl, 8, 1, 3);
                    transposeFld(x, 8, 1, 3);
                    transposeFld(y, 8, 1, 3);
                    transposeFld(z, 8, 1, 3);
                }

                xt[0] = x[0];
                yt[0] = y[0];
                zt[0] = z[0];
                xt[1] = x[3];
                yt[1] = y[3];
                zt[1] = z[3];
                xt[2] = x[7];
                yt[2] = y[7];
                zt[2] = z[7];
                xt[3] = x[6];
                yt[3] = y[6];
                zt[3] = z[6];
                if (!tetraVol(xt, yt, zt))
                {
                    /* 		    fprintf(stderr, " transpose4 %d\n",i); */
                    transposeIdx(cl, 8, 0, 7);
                    transposeFld(x, 8, 0, 7);
                    transposeFld(y, 8, 0, 7);
                    transposeFld(z, 8, 0, 7);
                }

                xt[0] = x[0];
                yt[0] = y[0];
                zt[0] = z[0];
                xt[1] = x[5];
                yt[1] = y[5];
                zt[1] = z[5];
                xt[2] = x[1];
                yt[2] = y[1];
                zt[2] = z[1];
                xt[3] = x[3];
                yt[3] = y[3];
                zt[3] = z[3];
                if (!tetraVol(xt, yt, zt))
                {
                    /* 		    fprintf(stderr, " transpose5 %d\n",i); */
                    transposeIdx(cl, 8, 0, 5);
                    transposeFld(x, 8, 0, 5);
                    transposeFld(y, 8, 0, 5);
                    transposeFld(z, 8, 0, 5);
                }

                xt[0] = x[6];
                yt[0] = y[6];
                zt[0] = z[6];
                xt[1] = x[7];
                yt[1] = y[7];
                zt[1] = z[7];
                xt[2] = x[3];
                yt[2] = y[3];
                zt[2] = z[3];
                xt[3] = x[5];
                yt[3] = y[5];
                zt[3] = z[5];
                if (!tetraVol(xt, yt, zt))
                {
                    /* 		    fprintf(stderr, " transpose6 %d\n",i); */
                    transposeIdx(cl, 8, 3, 6);
                    transposeFld(x, 8, 3, 6);
                    transposeFld(y, 8, 3, 6);
                    transposeFld(z, 8, 3, 6);
                }
                break;
            }
            /* restore the connection list */
            k = 0;
            for (j = beg; j < end; ++j)
            {
                grd->conn_l[j] = cl[k];
                k++;
            }
        }
    }
}

int
transposeIdx(int *tbl, int n, int idxA, int idxB)
{
    int tmp;

    if ((idxA >= n) || (idxB >= n))
    {
        return 0;
    }

    if (tbl)
    {
        tmp = tbl[idxA];
        tbl[idxA] = tbl[idxB];
        tbl[idxB] = tmp;
    }
    else
    {
        return 0;
    }

    return 1;
}

int
transposeFld(float *fld, int n, int idxA, int idxB)
{
    float tmpf;

    if ((idxA >= n) || (idxB >= n))
    {
        return 0;
    }

    if (fld)
    {
        tmpf = fld[idxA];
        fld[idxA] = fld[idxB];
        fld[idxB] = tmpf;
    }
    else
    {
        return 0;
    }

    return 1;
}

/*
  
returns vol(t) > 0

*/
int
tetraVol(float *x, float *y, float *z)
{
    int i;

    float c[3][3];
    float vol;

    /* create tetraeder vectors */
    for (i = 0; i < 3; ++i)
    {
        c[i][0] = x[i + 1] - x[0];
        c[i][1] = y[i + 1] - y[0];
        c[i][2] = z[i + 1] - z[0];
    }
    /* calc volume */
    vol = c[0][0] * (c[1][1] * c[2][2] - c[2][1] * c[1][2]);
    vol -= c[1][0] * (c[0][1] * c[2][2] - c[2][1] * c[0][2]);
    vol += c[2][0] * (c[0][1] * c[1][2] - c[1][1] * c[0][2]);

    return (vol > 0);
}

void
cleanUNSGRD(CoC_UNSGRD *grd)
{
    int i, j, k;
    int re;
    int numDegPts;
    int hDeg, pDeg, tDeg, totDeg;

    hDeg = 0;
    pDeg = 0;
    tDeg = 0;
    totDeg = 0;

    if (grd)
    {
        for (i = 0; i < grd->numElem - 1; i++)
        {
            re = 0;
            numDegPts = 0;
            for (j = grd->elem_l[i]; j < grd->elem_l[i + 1] - 1; j++)
            {
                for (k = j + 1; k < grd->elem_l[i + 1]; k++)
                {
                    if (grd->conn_l[j] == grd->conn_l[k])
                    {
                        totDeg++;
                        numDegPts++;
                        if (re == 0)
                        {

                            fprintf(stderr, "cleanUNSGRD: degenerated cells: celltype %d  actual number of corners %d\n", grd->type_l[i], (grd->elem_l[i + 1] - grd->elem_l[i]));

                            re = 1;
                        }
                        fprintf(stderr, "              %d : ( cl [ %d ] = cl [ %d ] )\n", i, j, k);
                    }
                }
            }
            if ((numDegPts == 28) && (grd->type_l[i] == TYPE_HEXAEDER))
            {
                fprintf(stderr, "cleanUNSGRD: fully degenerated HEXAEDER found\n");
                hDeg++;
            }
            if ((numDegPts == 10) && (grd->type_l[i] == TYPE_PRISM))
            {
                fprintf(stderr, "cleanUNSGRD: fully degenerated PRISM found\n");
                pDeg++;
            }
            if ((numDegPts == 6) && (grd->type_l[i] == TYPE_TETRAHEDER))
            {
                fprintf(stderr, "cleanUNSGRD: fully degenerated TETRAHEDER found\n");
                tDeg++;
            }
        }
    }

    fprintf(stderr, "cleanUNSGRD: %d degenerated cells found\n", totDeg);
    fprintf(stderr, "cleanUNSGRD: %d fully degenerated HEXAHEDRA found\n", hDeg);
    fprintf(stderr, "cleanUNSGRD: %d fully degenerated TETRAHEDRA found\n", tDeg);
    fprintf(stderr, "cleanUNSGRD: %d fully degenerated PRISMS found\n", pDeg);
}

/* returns the number of different corners  */
/* 1     : fully degenerated polyeder       */
/* num   : polyeder topology OK             */
/* -1    : unsuccesful                      */
int
checkPolyeder(int *poly, int num)
{
    int i, j;
    int fail, wrong;

    wrong = 0;

    if (poly)
    {
        for (i = 0; i < num; ++i)
        {
            fail = 0;
            for (j = i + 1; j < num; ++j)
            {
                if (poly[i] == poly[j])
                {
                    fail = 1;
                }
            }
            if (fail)
            {
                wrong++;
            }
        }
        return (num - wrong);
    }
    else
    {
        return -1;
    }
}

void
initIdxPhd(CoC_Polyed_UNSGRD *pIdx)
{
    if (pIdx)
    {
        if (pIdx->hexaIdx)
        {
            FREE(pIdx->hexaIdx);
        }
        pIdx->hexaIdx = NULL;
        pIdx->numHexa = 0;

        if (pIdx->tetraIdx)
        {
            FREE(pIdx->tetraIdx);
        }
        pIdx->tetraIdx = NULL;
        pIdx->numTetra = 0;

        if (pIdx->pyraIdx)
        {
            FREE(pIdx->pyraIdx);
        }
        pIdx->pyraIdx = NULL;
        pIdx->numPyra = 0;
    }
}

void
initIdxPgn(CoC_Polyed_POLYGN *pIdx)
{
    if (pIdx)
    {
        if (pIdx->triIdx)
        {
            FREE(pIdx->triIdx);
        }
        pIdx->triIdx = NULL;
        pIdx->numTri = 0;

        if (pIdx->quadIdx)
        {
            FREE(pIdx->quadIdx);
        }
        pIdx->quadIdx = NULL;
        pIdx->numQuad = 0;
    }
}

/*-------------------------------------------------------------------
  Write out a CoC_UNSGRD 
  -------------------------------------------------------------------*/
void
writeUNSGRD(FILE *fd, CoC_UNSGRD *grd)
{
    char *gtype;
    int ret;

    if (fd)
    {
        if (grd)
        {

            reorientUNSGRD(grd);

            gtype = (char *)RU_allocMem(6 * sizeof(char), "PNAME");
            strcpy(gtype, "UNSGRD");

            fwrite(gtype, sizeof(char), 6, fd);

            fwrite(&(grd->numElem), sizeof(int), 1, fd);
            fwrite(&(grd->numConn), sizeof(int), 1, fd);
            fwrite(&(grd->numCoords), sizeof(int), 1, fd);

            if (grd->elem_l)
            {
                ret = fwrite(grd->elem_l, sizeof(int), grd->numElem, fd);
                fprintf(stderr, "             ........wrote elements  %d\n", ret);
            }

            if (grd->type_l)
            {
                ret = fwrite(grd->type_l, sizeof(int), grd->numElem, fd);
                fprintf(stderr, "             ........wrote types   %d\n", ret);
            }

            if (grd->conn_l)
            {
                ret = fwrite(grd->conn_l, sizeof(int), grd->numConn, fd);
                fprintf(stderr, "             ........wrote connections   %d\n", ret);
            }
            /* write out coords */
            ret = fwrite(grd->x_coord, sizeof(float), grd->numCoords, fd);
            fprintf(stderr, "             ........wrote x-coords   %d\n", ret);
            ret = fwrite(grd->y_coord, sizeof(float), grd->numCoords, fd);
            fprintf(stderr, "             ........wrote y-coords   %d\n", ret);
            ret = fwrite(grd->z_coord, sizeof(float), grd->numCoords, fd);
            fprintf(stderr, "             ........wrote z-coords   %d\n", ret);
        }
    }
}

void
writeUNSGRDasci(FILE *fd, CoC_UNSGRD *grd)
{
    char *gtype;
    int i;

    if (fd)
    {
        if (grd)
        {

            gtype = (char *)RU_allocMem(6 * sizeof(char), "PNAME");
            strcpy(gtype, "UNSGRD");

            fprintf(fd, "%s\n", gtype);
            fprintf(fd, "nE:  %d\n", grd->numElem);
            fprintf(fd, "nC:  %d\n", grd->numConn);
            fprintf(fd, "nCo: %d\n", grd->numCoords);

            if (grd->elem_l)
            {
                for (i = 0; i < grd->numElem; ++i)
                {
                    fprintf(fd, "eL: %d\n", grd->elem_l[i]);
                }
            }

            if (grd->type_l)
            {
                for (i = 0; i < grd->numElem; ++i)
                {
                    fprintf(fd, "tL: %d\n", grd->type_l[i]);
                }
            }

            if (grd->conn_l)
            {

                for (i = 0; i < grd->numConn; ++i)
                {
                    fprintf(fd, "cL: %d\n", grd->conn_l[i]);
                }
            }
            /* write out coords */

            for (i = 0; i < grd->numCoords; ++i)
            {
                fprintf(fd, "xCo: %f\n", grd->x_coord[i]);
            }

            /* 	    ret = fwrite(grd->x_coord, sizeof(float), grd->numCoords, fd); */
            /* 	    fprintf(stderr, "             ........wrote x-coords   %d\n",ret); */
            /* 	    ret = fwrite(grd->y_coord, sizeof(float), grd->numCoords, fd); */
            /* 	    fprintf(stderr, "             ........wrote y-coords   %d\n",ret); */
            /* 	    ret = fwrite(grd->z_coord, sizeof(float), grd->numCoords, fd); */
            /* 	    fprintf(stderr, "             ........wrote z-coords   %d\n",ret); */
            /* ---------- */
        }
    }
}

void
writeCoFileHeader(FILE *fd)
{
    char *header;
    if (fd)
    {
        header = (char *)RU_allocMem(6 * sizeof(char), "PNAME");
        strcpy(header, "COV_LE");
        fwrite(header, sizeof(char), 6, fd);
    }
}

void
writeCoSetHeader(FILE *fd, int numSteps)
{
    char *header;
    if (fd)
    {
        header = (char *)RU_allocMem(6 * sizeof(char), "SNAME");
        strcpy(header, "SETELE");
        fwrite(header, sizeof(char), 6, fd);
        fwrite(&numSteps, sizeof(int), 1, fd);
    }
}

void
writeCoTimeStepsAttr(FILE *fd, int numSteps)
{

    char value[64];
    char timeAttr[16];
    int numattrib = 1;
    int size;

    strcpy(timeAttr, "TIMESTEP");

    sprintf(value, "%d %d", 1, numSteps);

    size = sizeof(int) + strlen(timeAttr) + strlen(value) + 2;

    fwrite(&size, sizeof(int), 1, fd);
    fwrite(&numattrib, sizeof(int), 1, fd);
    fwrite(timeAttr, sizeof(char), strlen(timeAttr) + 1, fd);
    fwrite(value, sizeof(char), strlen(value) + 1, fd);
}

/**********************************************************************
 *        Make an ensight directory                                   *
 **********************************************************************/

void
make_covise_directory(char *basename)
{
    char message[MAXLINE];

    /* construct directory name : ensight.<POST file name> */
    strcpy(COVISE_directory, "COVISE.");
    strcat(COVISE_directory, basename);

/* make the directory */
#ifdef WIN32
    if (_mkdir(COVISE_directory))
    {
#else
    if (mkdir(COVISE_directory, 00774))
    {
#endif
#ifdef DEBUG
        sprintf(message, "make COVISE directory \"%s\" failed (may already exist) \n", COVISE_directory);
        covise_message(ENSIGHT_WARNING, message);
#else
        sprintf(message, "make COVISE directory \"%s\" failed (may already exist) : overwriting \n", COVISE_directory);
        covise_message(ENSIGHT_FATAL_ERROR, message);
#endif /* DEBUG */
    }
    else
    {
        sprintf(message, "made COVISE directory \"%s\" \n", COVISE_directory);
        covise_message(ENSIGHT_INFO, message);
    }
}

/**********************************************************************
 *        Open a file in the ensight directory                        *
 **********************************************************************/

FILE *
open_covise_file(char *fname, char *mode)
{
    FILE *fp;

    fp = open_file(get_covise_pathname(fname), mode);

    return (fp);
}

char *
get_covise_pathname(char *fname)
{
    char *pathname;
    pathname = (char *)RU_allocMem(MAXLINE * sizeof(char), "PNAME");

    /* construct full pathname : ./ensight.<POST file name>/<fname> */
    strcpy(pathname, "./");
    strcat(pathname, COVISE_directory);
    strcat(pathname, "/");
    strcat(pathname, fname);

    return (pathname);
}

void
covise_message(int msgtype, char *s)
{
    char str[MAXLINE];

    /* Add message type to output string */
    switch (msgtype)
    {
    case ENSIGHT_FATAL_ERROR: /* Fatal error message */
        strcpy(str, "    *FATAL*");
        break;
    case ENSIGHT_WARNING: /* Warning message */
        strcpy(str, "    *WARNING*");
        break;
    case ENSIGHT_INFO: /* Information message */
        strcpy(str, "    *INFO*");
        break;
    default: /* Unknown message (fatal) */
        strcpy(str, "Unknown message type (");
        strcat(str, s); /* Add original message onto error message */
        strcat(str, ")");
        covise_message(ENSIGHT_FATAL_ERROR, str);
        break;
    }

    /* Add origin of message (i.e. external duct solver) */
    strcat(str, " COVISE TRANSLATOR : ");

    /* Add message to output string */
    strcat(str, s);

    /* Print output string to stderr */
    fprintf(stderr, "%s", str);

    /* flush stderr */
    fflush(stderr);

    /* exit if fatal error */
    if (msgtype == ENSIGHT_FATAL_ERROR)
        exit(EXIT_FAILURE);
}
