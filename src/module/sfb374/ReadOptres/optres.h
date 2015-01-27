/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef OPTRES
#define OPTRES

extern "C" {

extern int readoptrisstate_V6_0_C(
    const char *NomFic,
    int FlagGen, const char *Title, const char *Version, const char *Client, const char *Machine,
    int *NumState, int *ProgType, float *TimeState, float *ProgState, int *ProgNode,
    float *CPUCum, int *Icycle, int *Memory, int *Day, int *Month, int *Year, int *NbProc,
    int FlagNdNb, int *NdNb,
    int FlagNdNum, int **NdNum,
    int FlagNdCoord, float **NdX, float **NdY, float **NdZ,
    int FlagNdSpeed, float **NdVx, float **NdVy, float **NdVz,
    int FlagNdNorPress, float **NdNorPress,
    int FlagNdTgPress, float **NdTgPress,
    int FlagNdCrush, float **NdCrush,
    int FlagEl2Nb, int *El2Nb,
    int FlagEl2Num, int **El2Mat, int **El2Num, int **El2N1, int **El2N2,
    int FlagEl2Falg, float **El2Falg,
    int FlagEl2Dl, float **El2Dl,
    int FlagEl2Eint, float **El2Eint,
    int FlagEl2L0, float **El2L0,
    int FlagEl2DT, float **El2DT,
    int FlagEl3Nb, int *El3Nb,
    int FlagEl3Num, int **El3Mat, int **El3Num, int **El3N1, int **El3N2, int **El3N3,
    int FlagEl3Thk0, float **El3Thk0,
    int FlagEl3Thk, float **El3Thk,
    int FlagEl3DThk, float **El3DThk,
    int FlagEl3Eint, float **El3Eint,
    int FlagEl3LocS, float **El3Ux, float **El3Uy, float **El3Uz, float **El3Vx, float **El3Vy, float **El3Vz,
    int FlagEl3Sig, float **El3Sxx1, float **El3Syy1, float **El3Sxy1,
    float **El3Sxx3, float **El3Syy3, float **El3Sxy3,
    float **El3Sxx5, float **El3Syy5, float **El3Sxy5,
    int FlagEl3Eps, float **El3Ep1, float **El3Ep3, float **El3Ep5,
    int FlagEl3Epp, float **El3Epp1, float **El3Epp3, float **El3Epp5,
    int FlagEl3Ep, float **El3Exx1, float **El3Eyy1, float **El3Exy1,
    float **El3Exx3, float **El3Eyy3, float **El3Exy3,
    float **El3Exx5, float **El3Eyy5, float **El3Exy5,
    int FlagEl3DT, float **El3DT,
    int FlagEl3Seq, float **El3Seq1, float **El3Seq3, float **El3Seq5,
    int FlagEl4Nb, int *El4Nb,
    int FlagEl4Num, int **El4Mat, int **El4Num, int **El4N1, int **El4N2, int **El4N3, int **El4N4,
    int FlagEl4Thk0, float **El4Thk0,
    int FlagEl4Thk, float **El4Thk,
    int FlagEl4DThk, float **El4DThk,
    int FlagEl4Eint, float **El4Eint,
    int FlagEl4Ehg, float **El4Ehg,
    int FlagEl4LocS, float **El4Ux, float **El4Uy, float **El4Uz, float **El4Vx, float **El4Vy, float **El4Vz,
    int FlagEl4Sig, float **El4Sxx1, float **El4Syy1, float **El4Sxy1,
    float **El4Sxx3, float **El4Syy3, float **El4Sxy3,
    float **El4Sxx5, float **El4Syy5, float **El4Sxy5,
    int FlagEl4Eps, float **El4Ep1, float **El4Ep3, float **El4Ep5,
    int FlagEl4Epp, float **El4Epp1, float **El4Epp3, float **El4Epp5,
    int FlagEl4Ep, float **El4Exx1, float **El4Eyy1, float **El4Exy1,
    float **El4Exx3, float **El4Eyy3, float **El4Exy3,
    float **El4Exx5, float **El4Eyy5, float **El4Exy5,
    int FlagEl4DT, float **El4DT,
    int FlagEl4Seq, float **El4Seq1, float **El4Seq3, float **El4Seq5,
    int FlagEl8Nb, int *El8Nb,
    int FlagEl8Num, int **El8Mat, int **El8Num, int **El8N1, int **El8N2, int **El8N3, int **El8N4,
    int **El8N5, int **El8N6, int **El8N7, int **El8N8,
    int FlagEl8Eint, float **El8Eint,
    int FlagEl8Ehg, float **El8Ehg,
    int FlagEl8Sig, float **El8Sxx, float **El8Syy, float **El8Szz, float **El8Sxy, float **El8Sxz, float **El8Syz,
    int FlagEl8Eps, float **El8Ep,
    int FlagEl8Epp, float **El8Epp,
    int FlagEl8Ep, float **El8Exx, float **El8Eyy, float **El8Ezz, float **El8Exy, float **El8Exz, float **El8Eyz,
    int FlagEl8DT, float **El8DT,
    int FlagEl8Seq, float **El8Seq,
    int FlagMat, int *MatNb, char **MatNom, int **MatNum, int **MatType);
}

#define readStateFile(filename) readoptrisstate_V6_0_C(filename, flagGen, Title, Version, Client, Machine, &NumState, &ProgType, &TimeState, &ProgState, &ProgNode, &CPUCum, &Icycle, &Memory, &Day, &Month, &Year, &NbProc, flagNdNb, &NdNb, flagNdNum, &NdNum, flagNdCoord, &NdX, &NdY, &NdZ, flagNdSpeed, &NdVx, &NdVy, &NdVz, flagNdNorPress, &NdNorPress, flagNdTgPress, &NdTgPress, flagNdCrush, &NdCrush, flagEl2Nb, &El2Nb, flagEl2Num, &El2Mat, &El2Num, &El2N1, &El2N2, flagEl2Falg, &El2Falg, flagEl2Dl, &El2Dl, flagEl2Eint, &El2Eint, flagEl2L0, &El2L0, flagEl2DT, &El2DT, flagEl3Nb, &El3Nb, flagEl3Num, &El3Mat, &El3Num, &El3N1, &El3N2, &El3N3, flagEl3Thk0, &El3Thk0, flagEl3Thk, &El3Thk, flagEl3DThk, &El3DThk, flagEl3Eint, &El3Eint, flagEl3LocS, &El3Ux, &El3Uy, &El3Uz, &El3Vx, &El3Vy, &El3Vz, flagEl3Sig, &El3Sxx1, &El3Syy1, &El3Sxy1, &El3Sxx3, &El3Syy3, &El3Sxy3, &El3Sxx5, &El3Syy5, &El3Sxy5, flagEl3Eps, &El3Ep1, &El3Ep3, &El3Ep5, flagEl3Epp, &El3Epp1, &El3Epp3, &El3Epp5, flagEl3Ep, &El3Exx1, &El3Eyy1, &El3Exy1, &El3Exx3, &El3Eyy3, &El3Exy3, &El3Exx5, &El3Eyy5, &El3Exy5, flagEl3DT, &El3DT, flagEl3Seq, &El3Seq1, &El3Seq3, &El3Seq5, flagEl4Nb, &El4Nb, flagEl4Num, &El4Mat, &El4Num, &El4N1, &El4N2, &El4N3, &El4N4, flagEl4Thk0, &El4Thk0, flagEl4Thk, &El4Thk, flagEl4DThk, &El4DThk, flagEl4Eint, &El4Eint, flagEl4Ehg, &El4Ehg, flagEl4LocS, &El4Ux, &El4Uy, &El4Uz, &El4Vx, &El4Vy, &El4Vz, flagEl4Sig, &El4Sxx1, &El4Syy1, &El4Sxy1, &El4Sxx3, &El4Syy3, &El4Sxy3, &El4Sxx5, &El4Syy5, &El4Sxy5, flagEl4Eps, &El4Ep1, &El4Ep3, &El4Ep5, flagEl4Epp, &El4Epp1, &El4Epp3, &El4Epp5, flagEl4Ep, &El4Exx1, &El4Eyy1, &El4Exy1, &El4Exx3, &El4Eyy3, &El4Exy3, &El4Exx5, &El4Eyy5, &El4Exy5, flagEl4DT, &El4DT, flagEl4Seq, &El4Seq1, &El4Seq3, &El4Seq5, flagEl8Nb, &El8Nb, flagEl8Num, &El8Mat, &El8Num, &El8N1, &El8N2, &El8N3, &El8N4, &El8N5, &El8N6, &El8N7, &El8N8, flagEl8Eint, &El8Eint, flagEl8Ehg, &El8Ehg, flagEl8Sig, &El8Sxx, &El8Syy, &El8Szz, &El8Sxy, &El8Sxz, &El8Syz, flagEl8Eps, &El8Ep, flagEl8Epp, &El8Epp, flagEl8Ep, &El8Exx, &El8Eyy, &El8Ezz, &El8Exy, &El8Exz, &El8Eyz, flagEl8DT, &El8DT, flagEl8Seq, &El8Seq, flagMat, &MatNb, &MatNom, &MatNum, &MatType)

/* General data */
static int flagGen = 0;
static char Title[256];
static char Version[17];
static char Client[256];
static char Machine[256];
static int NumState;
static int ProgType;
static float TimeState;
static float ProgState;
static int ProgNode;
static float CPUCum;
static int Icycle;
static int Memory;
static int Day;
static int Month;
static int Year;
static int NbProc;

/* Nodes */
static int flagNdNb = 0;
static int NdNb = 0;
static int flagNdNum = 0;
static int *NdNum = NULL;
static int flagNdCoord = 0;
static float *NdX = NULL, *NdY = NULL, *NdZ = NULL;
static int flagNdSpeed = 0;
static float *NdVx = NULL, *NdVy = NULL, *NdVz = NULL;
static int flagNdNorPress = 0;
static float *NdNorPress = NULL;
static int flagNdTgPress = 0;
static float *NdTgPress = NULL;
static int flagNdCrush = 0;
static float *NdCrush = NULL;

/* 2-node elements */
static int flagEl2Nb = 0;
static int El2Nb = 0, El2NbTmp = 0;
static int flagEl2Num = 0;
static int *El2Mat = NULL, *El2Num = NULL, *El2N1 = NULL, *El2N2 = NULL;
static int flagEl2Falg = 0;
static float *El2Falg = NULL;
static int flagEl2Dl = 0;
static float *El2Dl = NULL;
static int flagEl2Eint = 0;
static float *El2Eint = NULL;
static int flagEl2L0 = 0;
static float *El2L0 = NULL;
static int flagEl2DT = 0;
static float *El2DT = NULL;

/* 3-node elements */
static int flagEl3Nb = 0;
static int El3Nb = 0, El3NbTmp = 0;
static int flagEl3Num = 0;
static int *El3Mat = NULL, *El3Num = NULL, *El3N1 = NULL, *El3N2 = NULL, *El3N3 = NULL;
static int flagEl3Thk0 = 0;
static float *El3Thk0 = NULL;
static int flagEl3Thk = 0;
static float *El3Thk = NULL;
static int flagEl3DThk = 0;
static float *El3DThk = NULL;
static int flagEl3Eint = 0;
static float *El3Eint = NULL;
static int flagEl3LocS = 0;
static float *El3Ux = NULL, *El3Uy = NULL, *El3Uz = NULL, *El3Vx = NULL, *El3Vy = NULL, *El3Vz = NULL;
static int flagEl3Sig = 0;
static float *El3Sxx1 = NULL, *El3Syy1 = NULL, *El3Sxy1 = NULL;
static float *El3Sxx2 = NULL, *El3Syy2 = NULL, *El3Sxy2 = NULL;
static float *El3Sxx3 = NULL, *El3Syy3 = NULL, *El3Sxy3 = NULL;
static float *El3Sxx4 = NULL, *El3Syy4 = NULL, *El3Sxy4 = NULL;
static float *El3Sxx5 = NULL, *El3Syy5 = NULL, *El3Sxy5 = NULL;
static float *El3Sxz = NULL, *El3Syz = NULL;
static int flagEl3Eps = 0;
static float *El3Ep1 = NULL, *El3Ep2 = NULL, *El3Ep3 = NULL, *El3Ep4 = NULL, *El3Ep5 = NULL;
static int flagEl3Epp = 0;
static float *El3Epp1 = NULL, *El3Epp2 = NULL, *El3Epp3 = NULL, *El3Epp4 = NULL, *El3Epp5 = NULL;
static int flagEl3Ep = 0;
static float *El3Exx1 = NULL, *El3Eyy1 = NULL, *El3Exy1 = NULL;
static float *El3Exx3 = NULL, *El3Eyy3 = NULL, *El3Exy3 = NULL;
static float *El3Exx5 = NULL, *El3Eyy5 = NULL, *El3Exy5 = NULL;
static int flagEl3DT = 0;
static float *El3DT = NULL;
static int flagEl3Seq = 0;
static float *El3Seq1 = NULL, *El3Seq2 = NULL, *El3Seq3 = NULL, *El3Seq4 = NULL, *El3Seq5 = NULL;

/* 4-node elements */
static int flagEl4Nb = 0;
static int El4Nb = 0, El4NbTmp = 0;
static int flagEl4Num = 0;
static int *El4Mat = NULL, *El4Num = NULL, *El4N1 = NULL, *El4N2 = NULL, *El4N3 = NULL, *El4N4 = NULL;
static int flagEl4Thk0 = 0;
static float *El4Thk0 = NULL;
static int flagEl4Thk = 0;
static float *El4Thk = NULL;
static int flagEl4DThk = 0;
static float *El4DThk = NULL;
static int flagEl4Eint = 0;
static float *El4Eint = NULL;
static int flagEl4Ehg = 0;
static float *El4Ehg = NULL;
static int flagEl4LocS = 0;
static float *El4Ux = NULL, *El4Uy = NULL, *El4Uz = NULL, *El4Vx = NULL, *El4Vy = NULL, *El4Vz = NULL;
static int flagEl4Sig = 0;
static float *El4Sxx1 = NULL, *El4Syy1 = NULL, *El4Sxy1 = NULL;
static float *El4Sxx2 = NULL, *El4Syy2 = NULL, *El4Sxy2 = NULL;
static float *El4Sxx3 = NULL, *El4Syy3 = NULL, *El4Sxy3 = NULL;
static float *El4Sxx4 = NULL, *El4Syy4 = NULL, *El4Sxy4 = NULL;
static float *El4Sxx5 = NULL, *El4Syy5 = NULL, *El4Sxy5 = NULL;
static float *El4Sxz = NULL, *El4Syz = NULL;
static int flagEl4Eps = 0;
static float *El4Ep1 = NULL, *El4Ep2 = NULL, *El4Ep3 = NULL, *El4Ep4 = NULL, *El4Ep5 = NULL;
static int flagEl4Epp = 0;
static float *El4Epp1 = NULL, *El4Epp2 = NULL, *El4Epp3 = NULL, *El4Epp4 = NULL, *El4Epp5 = NULL;
static int flagEl4Ep = 0;
static float *El4Exx1 = NULL, *El4Eyy1 = NULL, *El4Exy1 = NULL;
static float *El4Exx3 = NULL, *El4Eyy3 = NULL, *El4Exy3 = NULL;
static float *El4Exx5 = NULL, *El4Eyy5 = NULL, *El4Exy5 = NULL;
static int flagEl4DT = 0;
static float *El4DT = NULL;
static int flagEl4Seq = 0;
static float *El4Seq1 = NULL, *El4Seq2 = NULL, *El4Seq3 = NULL, *El4Seq4 = NULL, *El4Seq5 = NULL;

/* 8-node elements */
static int flagEl8Nb = 0;
static int El8Nb = 0, El8NbTmp = 0;
static int flagEl8Num = 0;
static int *El8Mat = NULL, *El8Num = NULL, *El8N1 = NULL, *El8N2 = NULL, *El8N3 = NULL, *El8N4 = NULL,
           *El8N5 = NULL, *El8N6 = NULL, *El8N7 = NULL, *El8N8 = NULL;
static int flagEl8Eint = 0;
static float *El8Eint = NULL;
static int flagEl8Ehg = 0;
static float *El8Ehg = NULL;
static int flagEl8Sig = 0;
static float *El8Sxx = NULL, *El8Syy = NULL, *El8Szz = NULL,
             *El8Sxy = NULL, *El8Sxz = NULL, *El8Syz = NULL;
static int flagEl8Eps = 0;
static float *El8Ep = NULL;
static int flagEl8Epp = 0;
static float *El8Epp = NULL;
static int flagEl8Ep = 0;
static float *El8Exx = NULL, *El8Eyy = NULL, *El8Ezz = NULL,
             *El8Exy = NULL, *El8Exz = NULL, *El8Eyz = NULL;
static int flagEl8DT = 0;
static float *El8DT = NULL;
static int flagEl8Seq = 0;
static float *El8Seq = NULL;

/* Materials */
static int flagMat = 0;
static int MatNb = 0;
static char *MatNom = NULL;
static int *MatNum = NULL;
static int *MatType = NULL;

static int flagProgType = 0;
static int flagNbPts = 0;
static int NbPts = 0;
static int flagNbRes = 0;
static int NbRes = 0;
static int flagNomsRes = 0;
static char *NomsRes = NULL;
static int flagRes = 0;
static float *Tps = NULL, *Prog = NULL, *Fx = NULL, *Fy = NULL, *Fz = NULL;
static char NomRes[256];

#define myDelete(x) \
    ;               \
    if (x)          \
    {               \
        free(x);    \
        x = NULL;   \
    }

void freeStateData()
{
    myDelete(NdNum);
    myDelete(NdX);
    myDelete(NdY);
    myDelete(NdZ);
    myDelete(NdVx);
    myDelete(NdVy);
    myDelete(NdVz);
    myDelete(NdNorPress);
    myDelete(NdTgPress);
    myDelete(NdCrush);

    myDelete(El2Mat);
    myDelete(El2Num);
    myDelete(El2N1);
    myDelete(El2N2);
    myDelete(El2Falg);
    myDelete(El2Dl);
    myDelete(El2Eint);
    myDelete(El2L0);
    myDelete(El2DT);

    myDelete(El3Mat);
    myDelete(El3Num);
    myDelete(El3N1);
    myDelete(El3N2);
    myDelete(El3N3);
    myDelete(El3Thk0);
    myDelete(El3Thk);
    myDelete(El3DThk);
    myDelete(El3Eint);
    myDelete(El3Ux);
    myDelete(El3Uy);
    myDelete(El3Uz);
    myDelete(El3Vx);
    myDelete(El3Vy);
    myDelete(El3Vz);
    myDelete(El3Sxx1);
    myDelete(El3Syy1);
    myDelete(El3Sxy1);
    myDelete(El3Sxx3);
    myDelete(El3Syy3);
    myDelete(El3Sxy3);
    myDelete(El3Sxx5);
    myDelete(El3Syy5);
    myDelete(El3Sxy5);
    myDelete(El3Ep1);
    myDelete(El3Ep3);
    myDelete(El3Ep5);
    myDelete(El3Epp1);
    myDelete(El3Epp3);
    myDelete(El3Epp5);
    myDelete(El3Exx1);
    myDelete(El3Eyy1);
    myDelete(El3Exy1);
    myDelete(El3Exx3);
    myDelete(El3Eyy3);
    myDelete(El3Exy3);
    myDelete(El3Exx5);
    myDelete(El3Eyy5);
    myDelete(El3Exy5);
    myDelete(El3DT);
    myDelete(El3Seq1);
    myDelete(El3Seq3);
    myDelete(El3Seq5);

    myDelete(El4Mat);
    myDelete(El4Num);
    myDelete(El4N1);
    myDelete(El4N2);
    myDelete(El4N3);
    myDelete(El4Thk0);
    myDelete(El4Thk);
    myDelete(El4DThk);
    myDelete(El4Eint);
    myDelete(El4Ux);
    myDelete(El4Uy);
    myDelete(El4Uz);
    myDelete(El4Vx);
    myDelete(El4Vy);
    myDelete(El4Vz);
    myDelete(El4Sxx1);
    myDelete(El4Syy1);
    myDelete(El4Sxy1);
    myDelete(El4Sxx3);
    myDelete(El4Syy3);
    myDelete(El4Sxy3);
    myDelete(El4Sxx5);
    myDelete(El4Syy5);
    myDelete(El4Sxy5);
    myDelete(El4Ep1);
    myDelete(El4Ep3);
    myDelete(El4Ep5);
    myDelete(El4Epp1);
    myDelete(El4Epp3);
    myDelete(El4Epp5);
    myDelete(El4Exx1);
    myDelete(El4Eyy1);
    myDelete(El4Exy1);
    myDelete(El4Exx3);
    myDelete(El4Eyy3);
    myDelete(El4Exy3);
    myDelete(El4Exx5);
    myDelete(El4Eyy5);
    myDelete(El4Exy5);
    myDelete(El4DT);
    myDelete(El4Seq1);
    myDelete(El4Seq3);
    myDelete(El4Seq5);

    myDelete(El8Mat);
    myDelete(El8Num);
    myDelete(El8N1);
    myDelete(El8N2);
    myDelete(El8N3);
    myDelete(El8N4);
    myDelete(El8N5);
    myDelete(El8N6);
    myDelete(El8N7);
    myDelete(El8N8);
    myDelete(El8Eint);
    myDelete(El8Ehg);
    myDelete(El8Sxx);
    myDelete(El8Syy);
    myDelete(El8Szz);
    myDelete(El8Sxy);
    myDelete(El8Sxz);
    myDelete(El8Syz);
    myDelete(El8Ep);
    myDelete(El8Epp);
    myDelete(El8Exx);
    myDelete(El8Eyy);
    myDelete(El8Ezz);
    myDelete(El8Exy);
    myDelete(El8Exz);
    myDelete(El8Eyz);
    myDelete(El8DT);
    myDelete(El8Seq);

    myDelete(MatNom);
    myDelete(MatNum);
    myDelete(MatType);
}
#endif
