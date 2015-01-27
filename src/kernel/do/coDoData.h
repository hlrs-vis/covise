/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_DO_DATA_H
#define CO_DO_DATA_H

#include "coDistributedObject.h"

/***********************************************************************\ 
 **                                                                     **
 **   Unstructured data class                            Version: 1.0   **
 **                                                                     **
 **                                                                     **
 **   Description  : Classes for the handling of the data on            **
 **                  a structured grid in a distributed manner.         **
 **                                                                     **
 **   Classes      : DO_Unstructured_S3d_data, DO_Unstructured_V3d_data **
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
\***********************************************************************/

namespace covise
{

inline int coIndex(const int i[3], const int dims[3])
{
    return (i[0] * dims[1] + i[1]) * dims[2] + i[2];
}

inline int coIndex(int i, int j, int k, int im, int jm, int km)
{
    int index[3] = { i, j, k };
    int sizes[3] = { im, jm, km };
    return coIndex(index, sizes);
}

inline int coIndex(int i, int j, int k, const int dims[3])
{
    int index[3] = { i, j, k };
    return coIndex(index, dims);
}

class DOEXPORT coDoAbstractData : public coDistributedObject
{
public:
    coDoAbstractData(const coObjInfo &info, const char *t)
        : coDistributedObject(info, t)
    {
    }
    virtual void cloneValue(int dstIdx, const coDoAbstractData *src, int srcIdx) = 0;
    virtual void setNullValue(int dstIdx) = 0;
    virtual coDoAbstractData *cloneType(const coObjInfo &info, int numElements) const = 0;

    virtual int getNumPoints() const // returns gridsize
    {
        return no_of_points;
    }

protected:
    coIntShm no_of_points; // number of points
};

extern DOEXPORT const char USTSDT[];
extern DOEXPORT const char RGBADT[];
extern DOEXPORT const char INTDT[];
extern DOEXPORT const char BYTEDT[];

template <typename ValueType, int typenum, const char *typetag>
class DOEXPORT coDoScalarData : public coDoAbstractData
{
protected:
    coDataShmArray<ValueType, typenum> s_data; // scalar data

    int rebuildFromShm()
    {
        covise_data_list dl[2];

        if (shmarr == NULL)
        {
            cerr << "called rebuildFromShm without shmarray\n";
            print_exit(__LINE__, __FILE__, 1);
        }
        dl[0].type = INTSHM;
        dl[0].ptr = &no_of_points;
        dl[1].type = typenum;
        dl[1].ptr = &s_data;
        return restore_shared_dl(2, dl);
    }

    int getObjInfo(int no, coDoInfo **il) const
    {
        if (no == 2)
        {
            (*il)[0].description = "Number of Points";
            (*il)[1].description = "Data Values";
            return 2;
        }
        else
        {
            print_error(__LINE__, __FILE__, "number wrong for object info");
            return 0;
        }
    }

    coDoScalarData *cloneObject(const coObjInfo &newinfo) const
    {
        return new coDoScalarData(newinfo, getNumPoints(), getAddress());
    }

public:
    coDoScalarData(const coObjInfo &info)
        : coDoAbstractData(info, typetag)
    {
        if (name)
        {
            if (getShmArray() != 0)
            {
                if (rebuildFromShm() == 0)
                {
                    print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object %s doesn't exist", name);
                new_ok = 0;
            }
        }
    }

    coDoScalarData(const coObjInfo &info, coShmArray *arr)
        : coDoAbstractData(info, typetag)
    {
        if (createFromShm(arr) == 0)
        {
            print_comment(__LINE__, __FILE__, "createFromShm == 0");
            new_ok = 0;
        }
    }

    coDoScalarData(const coObjInfo &info, int no, ValueType *s)
        : coDoAbstractData(info, typetag)
    {
        covise_data_list dl[2];
        s_data.set_length(no);
        dl[0].type = INTSHM;
        dl[0].ptr = &no_of_points;
        dl[1].type = typenum;
        dl[1].ptr = &s_data;
        new_ok = store_shared_dl(2, dl) != 0;
        if (!new_ok)
            return;
        no_of_points = no;

        int n = no * sizeof(ValueType);
        ValueType *tmps;
        getAddress(&tmps);
        memcpy(tmps, s, n);
    }

    coDoScalarData(const coObjInfo &info, int no)
        : coDoAbstractData(info, typetag)
    {
        covise_data_list dl[2];
        s_data.set_length(no);
        dl[0].type = INTSHM;
        dl[0].ptr = &no_of_points;
        dl[1].type = typenum;
        dl[1].ptr = &s_data;

        new_ok = store_shared_dl(2, dl) != 0;
        if (!new_ok)
            return;
        no_of_points = no;
// workaround for windows error (ShowUsg), method not found
// must be used inside the constructor
#ifdef _WIN32
        setPointValue(0, s_data[0]);
#endif
    }

    coDoAbstractData *cloneType(const coObjInfo &info, int no) const
    {
        return new coDoScalarData(info, no);
    }

    void getPointValue(int no, ValueType *s) const
    {
        *s = s_data[no];
    }

    void setPointValue(int no, ValueType s)
    {
        s_data[no] = s;
    }

    void cloneValue(int dstIdx, const coDoAbstractData *src, int srcIdx)
    {
        s_data[dstIdx] = static_cast<const coDoScalarData *>(src)->s_data[srcIdx];
    }

    void setNullValue(int dstIdx)
    {
        s_data[dstIdx] = 0;
    }

    // data in linearized form:
    void getAddress(ValueType **data) const
    {
        *data = static_cast<ValueType *>(s_data.getDataPtr());
    }
    ValueType *getAddress() const
    {
        return static_cast<ValueType *>(s_data.getDataPtr());
    }

    /** set new values for sizes: only DECREASING is allowed
       *  @return   0 if ok, -1 on error
       *  @param    numElem    New size of element list
       *  @param    numConn    New size of connectivity list
       *  @param    numCoord   New size of coordinale list
       */
    int setSize(int numElem)
    {
        if (numElem > no_of_points)
            return -1;

        no_of_points = numElem;
        return 0;
    }

    static coDistributedObject *virtualCtor(coShmArray *arr)
    {
        return new coDoScalarData(coObjInfo(), arr);
    }
};

INST_TEMPLATE3(template class DOEXPORT coDoScalarData<float, FLOATSHMARRAY, USTSDT>)
typedef coDoScalarData<float, FLOATSHMARRAY, USTSDT> coDoFloat;
INST_TEMPLATE3(template class DOEXPORT coDoScalarData<int, INTSHMARRAY, INTDT>)
typedef coDoScalarData<int, INTSHMARRAY, INTDT> coDoInt;
INST_TEMPLATE3(template class DOEXPORT coDoScalarData<char, CHARSHMARRAY, BYTEDT>)
typedef coDoScalarData<unsigned char, CHARSHMARRAY, BYTEDT> coDoByte;

INST_TEMPLATE3(template class DOEXPORT coDoScalarData<int, INTSHMARRAY, RGBADT>)
class DOEXPORT coDoRGBA : public coDoScalarData<int, INTSHMARRAY, RGBADT>
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

public:
    coDoRGBA(const coObjInfo &info)
        : coDoScalarData<int, INTSHMARRAY, RGBADT>(info)
    {
    }
    coDoRGBA(const coObjInfo &info, coShmArray *arr)
        : coDoScalarData<int, INTSHMARRAY, RGBADT>(info, arr)
    {
    }
    coDoRGBA(const coObjInfo &info, int no, int *pc)
        : coDoScalarData<int, INTSHMARRAY, RGBADT>(info, no, pc)
    {
    }
    coDoRGBA(const coObjInfo &info, int no)
        : coDoScalarData<int, INTSHMARRAY, RGBADT>(info, no)
    {
    }

    coDoAbstractData *cloneType(const coObjInfo &info, int no) const
    {
        return new coDoRGBA(info, no);
    }

    int setFloatRGBA(int pos, float r, float g, float b, float a = 1.0);
    int setIntRGBA(int pos, int r, int g, int b, int a = 255);
    int getFloatRGBA(int pos, float *r, float *g, float *b, float *a) const;
    int getIntRGBA(int pos, int *r, int *g, int *b, int *a) const;

protected:
    coDoRGBA *cloneObject(const coObjInfo &newinfo) const;
};

class DOEXPORT coDoVec2 : public coDoAbstractData
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);
    coFloatShmArray s_data; // scalar data
    coFloatShmArray t_data; // scalar data
protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoVec2 *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoVec2(const coObjInfo &info)
        : coDoAbstractData(info, "USTSTD")
    {
        if (name)
        {
            if (getShmArray() != 0)
            {
                if (rebuildFromShm() == 0)
                {
                    print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object %s doesn't exist", name);
                new_ok = 0;
            }
        }
    };
    coDoVec2(const coObjInfo &info, coShmArray *arr);
    coDoVec2(const coObjInfo &info, int no, float *s, float *t);
    coDoVec2(const coObjInfo &info, int no);
    coDoAbstractData *cloneType(const coObjInfo &info, int no) const
    {
        return new coDoVec2(info, no);
    }
    void getPointValue(int no, float *s, float *t) const
    {
        *s = s_data[no];
        *t = t_data[no];
    };
    void cloneValue(int dstIdx, const coDoAbstractData *src, int srcIdx)
    {
        s_data[dstIdx] = static_cast<const coDoVec2 *>(src)->s_data[srcIdx];
        t_data[dstIdx] = static_cast<const coDoVec2 *>(src)->t_data[srcIdx];
    }
    void setNullValue(int dstIdx)
    {
        s_data[dstIdx] = 0;
        t_data[dstIdx] = 0;
    }
    // data in linearized form:
    void getAddresses(float **s_d, float **t_d) const
    {
        *s_d = (float *)(s_data.getDataPtr());
        *t_d = (float *)(t_data.getDataPtr());
    };

    /** set new values for sizes: only DECREASING is allowed
       *  @return   0 if ok, -1 on error
       *  @param    numElem    New size of element list
       *  @param    numConn    New size of connectivity list
       *  @param    numCoord   New size of coordinale list
       */
    int setSize(int numElem);
};

class DOEXPORT coDoVec3 : public coDoAbstractData
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);
    coFloatShmArray u; // data for vector
    coFloatShmArray v; // data for vector
    coFloatShmArray w; // data for vector
protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoVec3 *cloneObject(const coObjInfo &newinfo) const;

public:
    coDoVec3(const coObjInfo &info)
        : coDoAbstractData(info, "USTVDT")
    {
        if (name)
        {
            if (getShmArray() != 0)
            {
                if (rebuildFromShm() == 0)
                {
                    print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object %s doesn't exist", name);
                new_ok = 0;
            }
        }
    };
    coDoVec3(const coObjInfo &info, coShmArray *arr);
    coDoVec3(const coObjInfo &info, int no,
             float *xc, float *yc, float *zc);
    coDoVec3(const coObjInfo &info, int no);
    coDoAbstractData *cloneType(const coObjInfo &info, int no) const
    {
        return new coDoVec3(info, no);
    }
    void getPointValue(int no, float *s) const
    {
        s[0] = u[no];
        s[1] = v[no];
        s[2] = w[no];
    };
    void cloneValue(int dstIdx, const coDoAbstractData *src, int srcIdx)
    {
        u[dstIdx] = static_cast<const coDoVec3 *>(src)->u[srcIdx];
        v[dstIdx] = static_cast<const coDoVec3 *>(src)->v[srcIdx];
        w[dstIdx] = static_cast<const coDoVec3 *>(src)->w[srcIdx];
    }
    void setNullValue(int dstIdx)
    {
        u[dstIdx] = v[dstIdx] = w[dstIdx] = 0;
    }
    void getAddresses(float **u_v, float **v_v, float **w_v) const
    {
        *u_v = (float *)u.getDataPtr();
        *v_v = (float *)v.getDataPtr();
        *w_v = (float *)w.getDataPtr();
    };

    /** set new values for sizes: only DECREASING is allowed
       *  @return   0 if ok, -1 on error
       *  @param    numElem    New size of element list
       *  @param    numConn    New size of connectivity list
       *  @param    numCoord   New size of coordinale list
       */
    int setSize(int numElem);
};

// storage order
// R3D: XX XY XZ YX YY YZ ZX ZY ZZ
class DOEXPORT coDoMat3 : public coDoAbstractData
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);
    // a single coFloatShmArray may make construction more efficient
    // in many cases than having as many coFloatShmArrays as components
    coFloatShmArray references;

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoMat3 *cloneObject(const coObjInfo &newinfo) const;

public:
    enum
    {
        X = 0,
        Y = 1,
        Z = 2
    };
    coDoMat3(const coObjInfo &info)
        : coDoAbstractData(info, "USTREF")
    {
        if (name)
        {
            if (getShmArray() != 0)
            {
                if (rebuildFromShm() == 0)
                {
                    print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object %s doesn't exist", name);
                new_ok = 0;
            }
        }
    };
    coDoMat3(const coObjInfo &info, coShmArray *arr);
    coDoMat3(const coObjInfo &info, int no, float *r_c);
    coDoMat3(const coObjInfo &info, int no);
    coDoAbstractData *cloneType(const coObjInfo &info, int no) const
    {
        return new coDoMat3(info, no);
    }
    void getPointValue(int no, float *m) const
    {
        memcpy(m, &references[9 * no], 9 * sizeof(float));
    };
    void cloneValue(int dstIdx, const coDoAbstractData *src, int srcIdx)
    {
        memcpy(&references[9 * dstIdx], &static_cast<const coDoMat3 *>(src)->references[9 * srcIdx], 9 * sizeof(float));
    }
    void setNullValue(int dstIdx)
    {
        memset(&references[9 * dstIdx], 0, 9 * sizeof(float));
    }
    void getAddress(float **ref) const
    {
        *ref = (float *)(references.getDataPtr());
    };
    float *getAddress() const
    {
        return static_cast<float *>(references.getDataPtr());
    }

    /** set new values for sizes: only DECREASING is allowed
       *  @return   0 if ok, -1 on error
       *  @param    numElem    New size of element list
       *  @param    numConn    New size of connectivity list
       *  @param    numCoord   New size of coordinale list
       */
    int setSize(int numElem);
};

// storage order
// S2D: XX YY XY
// F2D: XX XY YX YY
// S3D: XX YY ZZ XY YZ ZX
// F3D: XX XY XZ YX YY YZ ZX ZY ZZ
class DOEXPORT coDoTensor : public coDoAbstractData
{
    friend class coDoInitializer;
    static coDistributedObject *virtualCtor(coShmArray *arr);

public:
    // new entries in enum TensorType require registration in
    // in intTypes_ and strTypes_ leaving UNKNOWN as the last entry
    // in these arrays
    enum TensorType
    {
        UNKNOWN = 0,
        S2D = 3,
        F2D = 4,
        S3D = 6,
        F3D = 9
    };

private:
    static const TensorType listOfTypes_[];
    static const char *strTypes_[];
    coIntShm TensorType_;
    // a single coFloatShmArray may make construction more efficient
    // in many cases than having as many coFloatShmArrays as components
    coFloatShmArray t_data;

protected:
    int rebuildFromShm();
    int getObjInfo(int, coDoInfo **) const;
    coDoTensor *cloneObject(const coObjInfo &newinfo) const;

public:
    TensorType getTensorType() const; //{return TensorType_;}
    int dimension() const
    {
        return TensorType_;
    }
    const char *getTensorCharType() const;
    coDoTensor(const coObjInfo &info, TensorType ttype)
        : coDoAbstractData(info, "USTTDT")
    {
        TensorType_ = ttype;
        if (name)
        {
            if (getShmArray() != 0)
            {
                if (rebuildFromShm() == 0)
                {
                    print_comment(__LINE__, __FILE__, "rebuildFromShm == 0");
                }
            }
            else
            {
                print_comment(__LINE__, __FILE__, "object %s doesn't exist", name);
                new_ok = 0;
            }
        }
    };
    coDoTensor(const coObjInfo &info, coShmArray *arr);
    coDoTensor(const coObjInfo &info, int no, float *r_c, TensorType ttype);
    coDoTensor(const coObjInfo &info, int no, TensorType ttype);
    coDoAbstractData *cloneType(const coObjInfo &info, int no) const
    {
        return new coDoTensor(info, no, getTensorType());
    }
    void getPointValue(int no, float *m) const
    {
        memcpy(m, &t_data[TensorType_ * no], TensorType_ * sizeof(float));
    };
    void cloneValue(int dstIdx, const coDoAbstractData *src, int srcIdx)
    {
        memcpy(&t_data[TensorType_ * dstIdx], &static_cast<const coDoTensor *>(src)->t_data[TensorType_ * srcIdx], TensorType_ * sizeof(float));
    }
    void setNullValue(int dstIdx)
    {
        memset(&t_data[TensorType_ * dstIdx], 0, TensorType_ * sizeof(float));
    }
    void getAddress(float **ref) const
    {
        *ref = (float *)(t_data.getDataPtr());
    };
    float *getAddress() const
    {
        return static_cast<float *>(t_data.getDataPtr());
    }

    /** set new values for sizes: only DECREASING is allowed
       *  @return   0 if ok, -1 on error
       *  @param    numElem    New size of element list
       *  @param    numConn    New size of connectivity list
       *  @param    numCoord   New size of coordinale list
       */
    int setSize(int numElem);
};
}
#endif
