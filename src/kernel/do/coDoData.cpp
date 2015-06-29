/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coDoData.h"

/***********************************************************************\ 
 **                                                                     **
 **   Structured Data classes Routines                   Version: 1.0   **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of data on structured     **
 **                  grids in a distributed manner.                     **
 **                                                                     **
 **   Classes      : DO_Scalar_3D_Data, DO_Vector_3D_Data,              **
 **                  DO_Solution                                        **
 **                                                                     **
 **   Copyright (C) 1993     by University of Stuttgart                 **
 **                             Computer Center (RUS)                   **
 **                             Allmandring 30                          **
 **                             7000 Stuttgart 80                       **
 **                                                                     **
 **                                                                     **
 **   Author       : A. Wierse   (RUS)                                  **
 **                                                                     **
 **   History      :                                                    **
 **                  23.06.93  Ver 1.0                                  **
 **                                                                     **
 **                                                                     **
 **                                                                     **
\***********************************************************************/

namespace covise
{

const char USTSDT[] = "USTSDT";
const char INTDT[] = "INTDT ";
const char RGBADT[] = "RGBADT";
const char BYTEDT[] = "BYTEDT";
}
using namespace covise;
///////////////////////////////////////////////////////////////////////////

coDistributedObject *coDoVec2::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;

    ret = new coDoVec2(coObjInfo(), arr);
    return ret;
}

coDoVec2 *coDoVec2::cloneObject(const coObjInfo &newinfo) const
{
    float *d[2];
    getAddresses(&d[0], &d[1]);
    return new coDoVec2(newinfo, getNumPoints(), d[0], d[1]);
}

int coDoVec2::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 3)
    {
        (*il)[0].description = "Number of Points";
        (*il)[1].description = "Data Values: 1. Component";
        (*il)[2].description = "Data Values: 2. Component";
        return 3;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoVec2::coDoVec2(const coObjInfo &info,
                   coShmArray *arr)
    : coDoAbstractData(info, "USTSTD")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoVec2::coDoVec2(const coObjInfo &info,
                   int no, float *s, float *t)
    : coDoAbstractData(info, "USTSTD")
{
    covise_data_list dl[3];

    s_data.set_length(no);
    t_data.set_length(no);
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&s_data;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&t_data;
    new_ok = store_shared_dl(3, dl) != 0;
    if (!new_ok)
        return;
    no_of_points = no;
    ;

    int n = no * sizeof(float);
    float *xtmp, *ytmp;
    getAddresses(&xtmp, &ytmp);
    memcpy(xtmp, s, n);
    memcpy(ytmp, t, n);

    /*	
       for(int n = 0;n < no;n++) {
      s_data[n] = s[n];
      t_data[n] = t[n];
       }
    */
}

coDoVec2::coDoVec2(const coObjInfo &info,
                   int no)
    : coDoAbstractData(info, "USTSTD")
{
    covise_data_list dl[3];

    s_data.set_length(no);
    t_data.set_length(no);
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&s_data;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&t_data;
    new_ok = store_shared_dl(3, dl) != 0;
    if (!new_ok)
        return;
    no_of_points = no;
    ;
}

int coDoVec2::rebuildFromShm()
{
    covise_data_list dl[3];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&s_data;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&t_data;
    return restore_shared_dl(3, dl);
}

int coDoVec2::setSize(int numElem)
{
    if (numElem > no_of_points)
        return -1;

    no_of_points = numElem;
    return 0;
}

///////////////////////////////////////////////////////////////////////////

coDistributedObject *coDoVec3::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;

    ret = new coDoVec3(coObjInfo(), arr);
    return ret;
}

int coDoVec3::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 4)
    {
        (*il)[0].description = "Number of Points";
        (*il)[1].description = "Data Values: 1. Component";
        (*il)[2].description = "Data Values: 2. Component";
        (*il)[3].description = "Data Values: 3. Component";
        return 4;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoVec3::coDoVec3(const coObjInfo &info,
                   int no, float *xc, float *yc, float *zc)
    : coDoAbstractData(info, "USTVDT")
{
    covise_data_list dl[4];

    u.set_length(no);
    v.set_length(no);
    w.set_length(no);
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&u;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&v;
    dl[3].type = FLOATSHMARRAY;
    dl[3].ptr = (void *)&w;
    new_ok = store_shared_dl(4, dl) != 0;
    if (!new_ok)
        return;
    no_of_points = no;
    ;
    int n = no * sizeof(float);
    float *tmpu, *tmpv, *tmpw;

    getAddresses(&tmpu, &tmpv, &tmpw);
    memcpy(tmpu, xc, n);
    memcpy(tmpv, yc, n);
    memcpy(tmpw, zc, n);
    /*
       for(n = 0;n < no;n++)
      u[n] = xc[n];
       for(n = 0;n < no;n++)
      v[n] = yc[n];
       for(n = 0;n < no;n++)
      w[n] = zc[n];
   */
}

coDoVec3::coDoVec3(const coObjInfo &info,
                   int no)
    : coDoAbstractData(info, "USTVDT")
{
    covise_data_list dl[4];

    u.set_length(no);
    v.set_length(no);
    w.set_length(no);
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&u;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&v;
    dl[3].type = FLOATSHMARRAY;
    dl[3].ptr = (void *)&w;
    new_ok = store_shared_dl(4, dl) != 0;
    if (!new_ok)
        return;
    no_of_points = no;
    ;
}

coDoVec3::coDoVec3(const coObjInfo &info,
                   coShmArray *arr)
    : coDoAbstractData(info, "USTVDT")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoVec3 *coDoVec3::cloneObject(const coObjInfo &newinfo) const
{
    float *d[3];
    getAddresses(&d[0], &d[1], &d[2]);
    return new coDoVec3(newinfo, getNumPoints(), d[0], d[1], d[2]);
}

int coDoVec3::rebuildFromShm()
{
    covise_data_list dl[4];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&u;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&v;
    dl[3].type = FLOATSHMARRAY;
    dl[3].ptr = (void *)&w;
    return restore_shared_dl(4, dl);
}

int coDoVec3::setSize(int numElem)
{
    if (numElem > no_of_points)
        return -1;

    no_of_points = numElem;
    return 0;
}

///////////////////////////////////////////////////////////////////////////

coDistributedObject *coDoRGBA::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;

    ret = new coDoRGBA(coObjInfo(), arr);
    return ret;
}

coDoRGBA *coDoRGBA::cloneObject(const coObjInfo &newinfo) const
{
    return new coDoRGBA(newinfo, getNumPoints(), getAddress());
}

// changed from abgr to rgba
// Uwe Woessner

int coDoRGBA::setFloatRGBA(int pos, float r, float g, float b, float a)
{
    if (pos < 0 || pos >= no_of_points)
        return 0;
    unsigned char *chptr = (unsigned char *)&s_data[pos];
#ifdef BYTESWAP
    *chptr = (unsigned char)(a * 255.0);
    chptr++;
    *(chptr) = (unsigned char)(b * 255.0);
    chptr++;
    *(chptr) = (unsigned char)(g * 255.0);
    chptr++;
    *(chptr) = (unsigned char)(r * 255.0);
#else
    *chptr = (unsigned char)(r * 255.0);
    chptr++;
    *(chptr) = (unsigned char)(g * 255.0);
    chptr++;
    *(chptr) = (unsigned char)(b * 255.0);
    chptr++;
    *(chptr) = (unsigned char)(a * 255.0);
#endif
    return 1;
}

int coDoRGBA::setIntRGBA(int pos, int r, int g, int b, int a)
{
    if (pos < 0 || pos >= no_of_points)
        return 0;
    unsigned char *chptr = (unsigned char *)&s_data[pos];
#ifdef BYTESWAP
    *chptr = (unsigned char)(a);
    chptr++;
    *(chptr) = (unsigned char)(b);
    chptr++;
    *(chptr) = (unsigned char)(g);
    chptr++;
    *(chptr) = (unsigned char)(r);
#else
    *chptr = (unsigned char)(r);
    chptr++;
    *(chptr) = (unsigned char)(g);
    chptr++;
    *(chptr) = (unsigned char)(b);
    chptr++;
    *(chptr) = (unsigned char)(a);
#endif
    return 1;
}

int coDoRGBA::getFloatRGBA(int pos, float *r, float *g, float *b, float *a) const
{
    if (pos < 0 || pos >= no_of_points)
        return 0;
    unsigned char *chptr = (unsigned char *)&s_data[pos];
#ifdef BYTESWAP
    *a = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *b = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *g = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *r = ((float)(*chptr)) / (float)255.0;
#else
    *r = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *g = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *b = ((float)(*chptr)) / (float)255.0;
    chptr++;
    *a = ((float)(*chptr)) / (float)255.0;
#endif
    return 1;
}

int coDoRGBA::getIntRGBA(int pos, int *r, int *g, int *b, int *a) const
{
    if (pos < 0 || pos >= no_of_points)
        return 0;
    unsigned char *chptr = (unsigned char *)&s_data[pos];
#ifdef BYTESWAP
    *a = *chptr;
    chptr++;
    *b = *chptr;
    chptr++;
    *g = *chptr;
    chptr++;
    *r = *chptr;
#else
    *r = *chptr;
    chptr++;
    *g = *chptr;
    chptr++;
    *b = *chptr;
    chptr++;
    *a = *chptr;
#endif
    return 1;
}

///////////////////////////////////////////////////////////////////////////

coDistributedObject *coDoMat3::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;

    ret = new coDoMat3(coObjInfo(), arr);
    return ret;
}

int coDoMat3::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 2)
    {
        (*il)[0].description = "Number of References";
        (*il)[1].description = "Reference system";
        return 2;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoMat3::coDoMat3(const coObjInfo &info,
                   int no, float *r)
    : coDoAbstractData(info, "USTREF")
{
    covise_data_list dl[2];

    references.set_length(9 * no);
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&references;
    new_ok = store_shared_dl(2, dl) != 0;
    if (!new_ok)
        return;
    no_of_points = no;
    int n = no * 9 * sizeof(float);
    float *tmpref;

    getAddress(&tmpref);
    memcpy(tmpref, r, n);
}

coDoMat3::coDoMat3(const coObjInfo &info,
                   int no)
    : coDoAbstractData(info, "USTREF")
{
    covise_data_list dl[2];

    references.set_length(9 * no);
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&references;
    new_ok = store_shared_dl(2, dl) != 0;
    if (!new_ok)
        return;
    no_of_points = no;
}

coDoMat3::coDoMat3(const coObjInfo &info,
                   coShmArray *arr)
    : coDoAbstractData(info, "USTREF")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoMat3 *coDoMat3::cloneObject(const coObjInfo &newinfo) const
{
    return new coDoMat3(newinfo, getNumPoints(), getAddress());
}

int coDoMat3::rebuildFromShm()
{
    covise_data_list dl[2];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = FLOATSHMARRAY;
    dl[1].ptr = (void *)&references;
    return restore_shared_dl(2, dl);
}

int coDoMat3::setSize(int numElem)
{
    if (numElem > no_of_points)
        return -1;

    no_of_points = numElem;
    return 0;
}

///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

coDistributedObject *coDoTensor::virtualCtor(coShmArray *arr)
{
    coDistributedObject *ret;

    ret = new coDoTensor(coObjInfo(), arr);
    return ret;
}

// leave UNKNOWN as the last entry in these arrays!!!
const coDoTensor::TensorType
    coDoTensor::listOfTypes_[] = { S2D, F2D, S3D, F3D, UNKNOWN };
const char *coDoTensor::strTypes_[] = { "Sym 2D", "Sym 3D", "Full 2D", "Full 3D", "Unknown" };

coDoTensor::TensorType coDoTensor::getTensorType() const
{
    int i = 0;
    int int_tens_type = TensorType_;
    while (listOfTypes_[i] != UNKNOWN)
    {
        if (int_tens_type == listOfTypes_[i])
            return listOfTypes_[i];
        ++i;
    }
    return UNKNOWN;
}

const char *coDoTensor::getTensorCharType() const
{
    int i = 0;
    int int_tens_type = TensorType_;
    while (listOfTypes_[i] != UNKNOWN)
    {
        if (int_tens_type == listOfTypes_[i])
            return strTypes_[i];
        ++i;
    }
    return strTypes_[i]; // "Unknown"
}

int coDoTensor::getObjInfo(int no, coDoInfo **il) const
{
    if (no == 3)
    {
        (*il)[0].description = "Number of tensors";
        (*il)[1].description = "Number of components";
        (*il)[2].description = "Tensor data";
        return 3;
    }
    else
    {
        print_error(__LINE__, __FILE__, "number wrong for object info");
        return 0;
    }
}

coDoTensor::coDoTensor(const coObjInfo &info,
                       int no, float *r, TensorType ttype)
    : coDoAbstractData(info, "USTTDT")
{
    covise_data_list dl[3];

    t_data.set_length(ttype * no);
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&TensorType_;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&t_data;
    new_ok = store_shared_dl(3, dl) != 0;
    if (!new_ok)
        return;
    no_of_points = no;
    TensorType_ = ttype;
    int n = no * ttype * sizeof(float);
    float *tmpref;

    getAddress(&tmpref);
    memcpy(tmpref, r, n);
}

coDoTensor::coDoTensor(const coObjInfo &info,
                       int no, TensorType ttype)
    : coDoAbstractData(info, "USTTDT")
{
    covise_data_list dl[3];

    t_data.set_length(ttype * no);
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&TensorType_;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&t_data;
    new_ok = store_shared_dl(3, dl) != 0;
    if (!new_ok)
        return;
    no_of_points = no;
    TensorType_ = ttype;
}

coDoTensor::coDoTensor(const coObjInfo &info,
                       coShmArray *arr)
    : coDoAbstractData(info, "USTTDT")
{
    if (createFromShm(arr) == 0)
    {
        print_comment(__LINE__, __FILE__, "createFromShm == 0");
        new_ok = 0;
    }
}

coDoTensor *coDoTensor::cloneObject(const coObjInfo &newinfo) const
{
    return new coDoTensor(newinfo, getNumPoints(), getAddress(), getTensorType());
}

int coDoTensor::rebuildFromShm()
{
    covise_data_list dl[3];

    if (shmarr == NULL)
    {
        cerr << "called rebuildFromShm without shmarray\n";
        print_exit(__LINE__, __FILE__, 1);
    }
    dl[0].type = INTSHM;
    dl[0].ptr = (void *)&no_of_points;
    dl[1].type = INTSHM;
    dl[1].ptr = (void *)&TensorType_;
    dl[2].type = FLOATSHMARRAY;
    dl[2].ptr = (void *)&t_data;
    return restore_shared_dl(3, dl);
}

int coDoTensor::setSize(int numElem)
{
    if (numElem > no_of_points)
        return -1;

    no_of_points = numElem;
    return 0;
}

///////////////////////////////////////////////////////////////////////////
