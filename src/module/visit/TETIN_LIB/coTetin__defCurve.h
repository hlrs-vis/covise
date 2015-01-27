/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__DEFINE_CURVE_H_
#define _CO_TETIN__DEFINE_CURVE_H_

#include "iostream.h"
#include "coTetinCommand.h"
#include "coTetin__utils.h"

// 04.06.99

/**
 * Class coTetin__defCurve implements Tetin file "define_curve" command
 *
 */

class coTetin__curve_record;
class coTetin__mesh_curve;

class coTetin__defCurve : public coTetinCommand
{

private:
    // ===================== the command's data =====================

    coTetin__curve_record *cr;

    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__defCurve(const coTetin__defCurve &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__defCurve &operator=(const coTetin__defCurve &)
    {
        return *this;
    };

    /// Default constructor:
    coTetin__defCurve()
        : cr(0){};

    // utility fct.
    coTetin__mesh_curve *pj_new_mesh_curve(istream &str, ostream &ostr,
                                           int n_nodes, int n_edges);

    coTetin__mesh_curve *pj_new_mesh_curve(float *points_x,
                                           float *points_y, float *points_z, int n_nodes);

public:
    /// wrapper for getLine
    istream &getNextLine(char *line, int length, istream &str);

    /// wrapper for getString
    char *getNextString(char *&chPtr);

    /// read from file
    coTetin__defCurve(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__defCurve(int *&, float *&, char *&);

    /// exchange curve with name crv_name by mesh curve
    coTetin__defCurve(float *points_x, float *points_y,
                      float *points_z, int n_points,
                      char *crv_name);

    /// Destructor
    virtual ~coTetin__defCurve();

    /// whether object is valid
    virtual int isValid() const;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;
    /// print to a stream in Tetin format
    virtual void print(ostream &str) const;

    // ===================== command-specific functions =====================
};

#define INVALID_CURVE -1
#define PARAMETRIC_CURVE 0
#define MESH_CURVE 1
#define UNSTRUCT_CURVE 2

typedef float point[3];
typedef int curve_edge[2];

class coTetin__mesh_curve;
class coTetin__param_curve;
class coTetin__unstruct_curve;

class coTetin__curve_record
{
    friend class coTetin__mesh_curve;
    friend class coTetin__param_curve;
    friend class coTetin__unstruct_curve;

public:
    char *family;
    // max tetrahedra size on curve
    float maxsize;
    // width
    int width;
    // ratio
    float ratio;
    // name
    char *name;

    // user specified min limit
    float minlimit;

    // user specified deviation
    float deviation;

    // height
    float height;
    short curve_type;
    char dormant;

    // names of the end vertices (temporary for contruction)
    char *end_names[2];

    virtual void internal_write(ostream &str,
                                int *&intDat, float *&floatDat, char *&charDat) const;

    coTetin__curve_record(int typ);
    ~coTetin__curve_record();
};

class coTetin__mesh_curve : public coTetin__curve_record
{
public:
    int n_pnts;
    point *pnts;
    int n_edges;
    curve_edge *edges;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;
    virtual void addBSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;
    /// put given set of pointers to my data
    virtual void putBinary(int *&intDat, float *&floatDat, char *&charDat);

    virtual void internal_write(ostream &str,
                                int *&intDat, float *&floatDat, char *&charDat) const;

    coTetin__mesh_curve();
    coTetin__mesh_curve(int npnts, int nedges);

    ~coTetin__mesh_curve();
};

class coTetin__param_curve : public coTetin__curve_record
{
public:
    coTetin__BSpline *spl;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;
    virtual void addBSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;
    /// put given set of pointers to my data
    virtual void putBinary(int *&intDat, float *&floatDat, char *&charDat);

    virtual void internal_write(ostream &str,
                                int *&intDat, float *&floatDat, char *&charDat) const;

    coTetin__param_curve();
    coTetin__param_curve(coTetin__BSpline *bspl);
    ~coTetin__param_curve();
};

class coTetin__unstruct_curve : public coTetin__curve_record
{
public:
    char *path;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;
    virtual void addBSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;
    /// put given set of pointers to my data
    virtual void putBinary(int *&intDat, float *&floatDat, char *&charDat);

    virtual void internal_write(ostream &str,
                                int *&intDat, float *&floatDat, char *&charDat) const;

    coTetin__unstruct_curve();
    coTetin__unstruct_curve(char *path);
    ~coTetin__unstruct_curve();
};
#endif
