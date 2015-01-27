/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "Diablo.h"

// some fortran definitions

#ifdef _SGI
extern "C" {
void dif1dim_(char[], double[][5500], int *, int *, double *, double *,
              double *, int *, int *, double *,
              double *, double *, double[], int *, int *, int);
void hard_(double[][5500], double[], int *, int *, int *, double *,
           double *, double *, int *, int *, int *,
           double *, int *, double *, int *, double *, int *);
/*     HARD    (H, TSCHM, TX, CCMAX, CCRESM, CCRESZ, DTRESZ,
                   DTMAXZ, EXEMOD, CCDEF, FMESH, ERRFLG)	       */
void prep_(char[], char[], int[], double[], double[], double[], int *, int *,
           double[], int *, int *, double[][5500], int *, int, int);
/*     PREP    (NODEID, XND, YND, ZND, NNODES, NNDMAX,
                   TAU, NTAU, NTMAX, TEMPARR, ERRFLG)             */
void rdabk_(double[], double *, double *, int *);
/*     RDABK   (DT, TSCHM, TX, ERRFLG)                         */
void rdc1dd_(double[], double *, double *, int *, int *);
/*     rdc1dd  (CC, CCMIN, CCMAX, FMESH, ERRFLG)               */
void rdcout_(double[], double *, double *, int *, int *, int *);
/*     RDCOUT  (CC, CCMIN, CCMAX, NND, NNDMAX, ERRFLG)         */
void rddk_(int *);
/*     RDDK    (ERRFLG)                                        */
void rdmart_(char[], double *, int *, double *, double *, int *, int);
/*     RDMART  (MFILE, CCMAX, CCRESM, TSCHM, DTMAX, ERRFLG)    */
void rdzta_(char[], double *, int *, int);
/*     RDZTA   (AFILE, TSCHM, ERRFLG)                          */
void rdzw_(char[], double *, int *, double *, int *, int *, int);
/*     RDZW    (ZFILE, CCMAX, CCRESZ, DTMAX, DTRESZ, ERRFLG)   */
void wrfiin_(double *, double *, int *, int *, double *, double *,
             double *, int *);
/*     WRFIIN  (LKORN, FMIKRO, FMESH, NSTEPS, TAU1, TAU2,
                   TEMP1, ERRFLG)                                 */
void wrtver_(int *, int *, double[], int *, double *, double *,
             int *, double *, double *, double *, double *, double *, int *,
             int *, double[][5500], int *);
/*     WRTVER  (NODENO, NNODES, TAU, NTAU, CCNODE, CCMAX,
                   CCRESM, TAC3, TAUA3X, TAUAC3, TSCHM, TAUMS, NOMAX,
                   NTAUMAX, TEMPARR, ERRFLG)                      */
}
#endif
#ifdef _CRAY
extern "C" {
void DIF1DIM(char[], double[][5500], int *, int *, double *, double *,
             double *, int *, int *, double *,
             double *, double *, double[], int *, int *, int);
void HARD(double[][5500], double[], int *, int *, int *, double *,
          double *, double *, int *, int *, int *,
          double *, int *, double *, int *, double *, int *);
/*     HARD    (H, TSCHM, TX, CCMAX, CCRESM, CCRESZ, DTRESZ,
                   DTMAXZ, EXEMOD, CCDEF, FMESH, ERRFLG)	       */
void PREP(char[], char[], int[], double[], double[], double[], int *, int *,
          double[], int *, int *, double[][5500], int *, int, int);
/*     PREP    (NODEID, XND, YND, ZND, NNODES, NNDMAX,
                   TAU, NTAU, NTMAX, TEMPARR, ERRFLG)             */
void RDABK(double[], double *, double *, int *);
/*     RDABK   (DT, TSCHM, TX, ERRFLG)                         */
void RDC1dd(double[], double *, double *, int *, int *);
/*     rdc1dd  (CC, CCMIN, CCMAX, FMESH, ERRFLG)               */
void RDCOUT(double[], double *, double *, int *, int *, int *);
/*     RDCOUT  (CC, CCMIN, CCMAX, NND, NNDMAX, ERRFLG)         */
void RDDK(int *);
/*     RDDK    (ERRFLG)                                        */
void RDMART(char[], double *, int *, double *, double *, int *, int);
/*     RDMART  (MFILE, CCMAX, CCRESM, TSCHM, DTMAX, ERRFLG)    */
void RDZTA(char[], double *, int *, int);
/*     RDZTA   (AFILE, TSCHM, ERRFLG)                          */
void RDZW(char[], double *, int *, double *, int *, int *, int);
/*     RDZW    (ZFILE, CCMAX, CCRESZ, DTMAX, DTRESZ, ERRFLG)   */
void WRFIIN(double *, double *, int *, int *, double *, double *,
            double *, int *);
/*     WRFIIN  (LKORN, FMIKRO, FMESH, NSTEPS, TAU1, TAU2,
                   TEMP1, ERRFLG)                                 */
void WRTVER(int *, int *, double[], int *, double *, double *,
            int *, double *, double *, double *, double *, double *, int *,
            int *, double[][5500], int *);
/*     WRTVER  (NODENO, NNODES, TAU, NTAU, CCNODE, CCMAX,
                   CCRESM, TAC3, TauA3x, TAUAC3, TSCHM, TAUMS, NOMAX,
                   NTAUMAX, TEMPARR, ERRFLG)                      */
}
#endif

// Hilfsproceduren
void fsearch(FILE *fp, char *str)
{
    char buf[300];

    // Sucht String in einem File
    fgets(buf, 300, fp);
    while (strcmp(buf, str) != 0)
    {
        fgets(buf, 300, fp);
    }
}

int get_meshheader(FILE *fp, int *n_coord, int *n_elem, int *ngroups, int *n_conn)
{
    int i, j, dummy;
    int anzelem, anznode, geometr;
    char buf[300];

    // Header lesen
    for (i = 0; i < 5; i++)
    {
        fgets(buf, 300, fp);
    }
    if (fscanf(fp, "%d%d%d%d%d\n", n_coord, n_elem, ngroups, &dummy, &dummy) == EOF)
    {
        return (-1);
    }

    // Element Gruppen suchen
    fsearch(fp, "ELEMENT GROUPS\n");

    // File nach Anzahl der Connections durchsuchen
    *n_conn = 0;
    for (i = 0; i < *ngroups; i++)
    {
        fscanf(fp, "%s%d%s%d%s%d%s%d%s%d\n", &buf, &dummy, &buf, &anzelem, &buf,
               &anznode, &buf, &geometr, &buf, &dummy);
        fgets(buf, 300, fp);
        for (j = 0; j < anzelem; j++)
        {
            fgets(buf, 300, fp);
            switch (anznode)
            {
            case 6:
                *n_conn += 6;
                break;
            case 8:
                *n_conn += 8;
                break;
            }
        }
    }
    // wieder Anfang der Datei
    rewind(fp);
    for (i = 0; i < 13; i++)
    {
        fgets(buf, 300, fp);
    }
    return (0);
}

int get_geometrie(FILE *fp, int npunkte, int ngroups, float *x, float *y, float *z,
                  int *vlist, int *elist, int *tlist)
{
    char buf[300];
    int i, j, p1, p2, p3, p4, p5, p6, p7, p8, dummy;
    int anzelem, anznode, geometr;
    int vlcount;

    // Knoten einlesen
    for (i = 0; i < npunkte; i++)
    {
        if (fscanf(fp, "%d%f%f%f\n", &dummy, x, y, z) == EOF)
        {
            return (-1);
        }
        x++;
        y++;
        z++;
    }

    // Element Gruppen suchen
    fsearch(fp, "ELEMENT GROUPS\n");

    vlcount = 0;
    for (i = 0; i < ngroups; i++)
    {
        fscanf(fp, "%s%d%s%d%s%d%s%d%s%d\n", &buf, &dummy, &buf, &anzelem, &buf,
               &anznode, &buf, &geometr, &buf, &dummy);
        fgets(buf, 300, fp);
        for (j = 0; j < anzelem; j++)
        {
            *elist = vlcount;
            switch (anznode)
            {
            case 6:
                fscanf(fp, "%d%d%d%d%d%d%d\n", &dummy, &p1, &p2, &p3, &p4,
                       &p5, &p6);
                *vlist = p1 - 1;
                vlist++;
                *vlist = p2 - 1;
                vlist++;
                *vlist = p3 - 1;
                vlist++;
                *vlist = p4 - 1;
                vlist++;
                *vlist = p5 - 1;
                vlist++;
                *vlist = p6 - 1;
                vlist++;
                vlcount += 6;
                break;
            case 8:
                fscanf(fp, "%d%d%d%d%d%d%d%d%d\n", &dummy, &p1, &p2, &p3, &p4,
                       &p5, &p6, &p7, &p8);
                *vlist = p1 - 1;
                vlist++;
                *vlist = p2 - 1;
                vlist++;
                *vlist = p4 - 1;
                vlist++;
                *vlist = p3 - 1;
                vlist++;
                *vlist = p5 - 1;
                vlist++;
                *vlist = p6 - 1;
                vlist++;
                *vlist = p8 - 1;
                vlist++;
                *vlist = p7 - 1;
                vlist++;
                vlcount += 8;
                break;
            }
            switch (geometr)
            {
            case 3:
                *tlist = TYPE_HEXAGON;
                break;
            case 4:
                *tlist = TYPE_PRISM;
                break;
            }
            elist++;
            tlist++;
        }
    }
    fclose(fp);
    return (0);
}
