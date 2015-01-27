/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <api/coModule.h>
using namespace covise;

#define NUM_ALLOC_LIST 100 // number of entries in familyNameList to allocate
// in one step

class ModifyHeadBox : public coModule
{

private:
    virtual int compute();
    virtual void quit();
    virtual void param(const char *name);
    virtual void postInst();

    int MAX_MOVE;

    coFloatSliderParam *p_trans_value1;
    coFloatSliderParam *p_trans_value2;
    coFloatSliderParam *p_trans_value3;
    coFloatSliderParam *p_trans_value4;
    coFloatSliderParam *p_trans_value5;
    coFloatSliderParam *p_trans_value6;
    coFloatSliderParam *p_trans_value7;
    coFloatSliderParam *p_trans_value8;
    coFloatSliderParam *p_trans_value9;
    coFloatSliderParam *p_trans_value10;
    coFloatSliderParam *p_trans_value11;
    coFloatSliderParam *p_trans_value12;

    coBooleanParam *p_recalc;
    coBooleanParam *p_mode;

    coFloatSliderParam *p_red;
    coFloatSliderParam *p_green;
    coFloatSliderParam *p_blue;
    coFloatSliderParam *p_opaque;

    coStringParam *p_user, *p_host, *p_dir, *p_run;

    coOutputPort *p_command;
    coOutputPort *p_geom;

    coOutputPort *p_points;

    char *user, *host, *dir, *run, *hexa_run;
    CoviseConfig *hexaconf;

    void createCube();
    coDoPolygons *surface;

    int flag, runmode;
    float trn1, otrn1, sotrn1;
    float trn2, otrn2, sotrn2;
    float trn3, otrn3, sotrn3;
    float trn4, otrn4, sotrn4;
    float trn5, otrn5, sotrn5;
    float trn6, otrn6, sotrn6;
    float trn7, otrn7, sotrn7;
    float trn8, otrn8, sotrn8;
    float trn9, otrn9, sotrn9;
    float trn10, otrn10, sotrn10;
    float trn11, otrn11, sotrn11;
    float trn12, otrn12, sotrn12;

    float dmin1, dmax1;
    float dmin2, dmax2;
    float dmin3, dmax3;
    float dmin4, dmax4;
    float dmin5, dmax5;
    float dmin6, dmax6;
    float dmin7, dmax7;
    float dmin8, dmax8;
    float dmin9, dmax9;
    float dmin10, dmax10;
    float dmin11, dmax11;
    float dmin12, dmax12;

    int cmode, ocmode;

    int rec, orec;

    void selfExec();
    int parser(char *, char **, int, char *);

    float red, green, blue, opq;
    float dmin, dmax;

public:
    ModifyHeadBox();
};
