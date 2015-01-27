/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CO_TETIN__DEFINE_SURFACE_H_
#define _CO_TETIN__DEFINE_SURFACE_H_

#include "iostream.h"
#include "coTetinCommand.h"
#include "coTetin__utils.h"

// 08.06.99

/**
 * Class coTetin__defSurf implements Tetin file "define_surface" command
 *
 */

class coTetin__surface_record;
class coTetin__face_surface;
class coTetin__mesh_surface;
class coTetin__param_surface;
class coTetin__unstruct_surface;
class coTetin__Loop;
class coTetin__coedge;
class coTetin__pcurve;
class coTetin__BSpline;

class coTetin__defSurf : public coTetinCommand
{
private:
    coTetin__surface_record *sr;

    /// Copy-Constructor: NOT IMPLEMENTED
    coTetin__defSurf(const coTetin__defSurf &){};

    /// Assignment operator: NOT IMPLEMENTED
    coTetin__defSurf &operator=(const coTetin__defSurf &)
    {
        return *this;
    };

    /// Default constructor: NOT IMPLEMENTED
    coTetin__defSurf()
        : sr(0){};

    // ===================== the command's data =====================

    // utility fct.
    coTetin__param_surface *pj_new_param_surface(int linec, char *linev[], char *name,
                                                 istream &str, ostream &ostr);
    coTetin__mesh_surface *pj_new_mesh_surface(istream &str, ostream &ostr,
                                               int n_points, int n_tris);
    coTetin__face_surface *pj_new_face_surface(int linec, char *linev[], char *name,
                                               istream &str, ostream &ostr);
    coTetin__Loop *read_new_loop(int linec, char *linev[],
                                 istream &str, ostream &ostr);
    coTetin__Loop *read_old_loop(int npnts,
                                 istream &str, ostream &ostr);
    coTetin__coedge *read_coedge(istream &str, ostream &ostr);
    coTetin__pcurve *read_polyline(int linec, char *linev[],
                                   istream &str, ostream &ostr);

public:
    /// wrapper for getString
    char *getNextString(char *&chPtr);

    /// read from file
    coTetin__defSurf(istream &str, int binary, ostream &ostr = cerr);

    /// read from memory
    coTetin__defSurf(int *&, float *&, char *&);

    /// Destructor
    virtual ~coTetin__defSurf();

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

typedef float twoD[2];
typedef float point[3];
typedef int Triangle[3];

#define INVALID_SURFACE -1
#define PARAMETRIC_SURFACE 0
#define MESH_SURFACE 1
#define FACE_SURFACE 2
#define UNSTRUCT_SURFACE 3

class coTetin__surface_record
{
    friend class coTetin__face_surface;
    friend class coTetin__mesh_surface;
    friend class coTetin__param_surface;
    friend class coTetin__unstruct_surface;

public:
    /* recommended size at each point */
    char *family;
    unsigned int reverse_normal : 1;
    // for tetra how many layers of tetrahedra near the surface
    float width;
    // maximum length of any tetrahedron intersecting this surface
    float maxsize;
    // ratio of element sizes
    float ratio;
    // minimum size specified on surface
    float minsize;
    // deviation
    float dev;
    // height of first prism element
    // if height is set the tetra will not use surfwid or surfrat
    float height;

    char *name;
    // level number from ddn
    int level;
    // sequence number from ddn
    int number;
    int by_ids;
#ifdef __hpux
#define SIGNED
#else
#define SIGNED signed
#endif
    // surface types are defined in project.h
    SIGNED char new_format;
    SIGNED char surface_type;

    virtual void internal_write(ostream &str,
                                int *&intDat, float *&floatDat, char *&charDat) const;

    coTetin__surface_record(int typ);
    ~coTetin__surface_record();
};

class coTetin__face_surface : public coTetin__surface_record
{
public:
    int n_loops;
    // should the loops be written in the new format
    coTetin__Loop **loops;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;
    virtual void addBSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;
    /// put given set of pointers to my data
    virtual void putBinary(int *&intDat, float *&floatDat, char *&charDat);
    virtual void internal_write(ostream &str,
                                int *&intDat, float *&floatDat, char *&charDat) const;

    coTetin__face_surface();
    ~coTetin__face_surface();
};

class coTetin__pcurve
{
public:
#define STD_PCURVE 0
#define BSPLINE_PCURVE 1
    int type;
    int npnts;
    twoD *pnts;
    virtual void write(ostream &str,
                       int *&intDat, float *&floatDat, char *&charDat) const;
    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;

    coTetin__pcurve();
    ~coTetin__pcurve();
};

// parameter space curve
class coTetin__coedge
{
public:
    // name of 3-D curve, this is temporary while reading a tetin file
    char *curve_name;
    // if the coedge is reversed with respect to the curve
    int rev;
    // pointer to 3-D curve
    coTetin__pcurve *p_curve;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;
    virtual void addBSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;
    /// put given set of pointers to my data
    virtual void putBinary(int *&intDat, float *&floatDat, char *&charDat);
    virtual void write(ostream &str,
                       int *&intDat, float *&floatDat, char *&charDat) const;

    coTetin__coedge();
    ~coTetin__coedge();
};

class coTetin__Loop
{
public:
    int ncoedges;
    coTetin__coedge **coedges;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;
    virtual void addBSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;
    /// put given set of pointers to my data
    virtual void putBinary(int *&intDat, float *&floatDat, char *&charDat);
    virtual void write(ostream &str,
                       int *&intDat, float *&floatDat, char *&charDat) const;

    coTetin__Loop();
    ~coTetin__Loop();
};

class coTetin__mesh_subsurface
{
public:
    int n_tri;
    Triangle *tris;
    ~coTetin__mesh_subsurface();
    coTetin__mesh_subsurface();
};

class coTetin__mesh_surface : public coTetin__surface_record
{
public:
    int npnts;
    point *pnts;
    coTetin__mesh_subsurface subsurf;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;
    virtual void addBSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;
    /// put given set of pointers to my data
    virtual void putBinary(int *&intDat, float *&floatDat, char *&charDat);
    virtual void internal_write(ostream &str,
                                int *&intDat, float *&floatDat, char *&charDat) const;

    coTetin__mesh_surface();
    ~coTetin__mesh_surface();
};

class coTetin__unstruct_surface : public coTetin__surface_record
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

    coTetin__unstruct_surface();
    coTetin__unstruct_surface(char *name);
    ~coTetin__unstruct_surface();
};

class coTetin__BSpline;
class coTetin__param_surface : public coTetin__surface_record
{
public:
    /* surf == 0 indicates an unused surface */
    coTetin__BSpline *surf;
    int n_loops;
    // should the loops be written in the new format
    coTetin__Loop **loops;

    /// count size required in fields
    virtual void addSizes(int &numInt, int &numFloat, int &numChar) const;
    virtual void addBSizes(int &numInt, int &numFloat, int &numChar) const;

    /// put my data to a given set of pointers
    virtual void getBinary(int *&intDat, float *&floatDat, char *&charDat) const;
    /// put given set of pointers to my data
    virtual void putBinary(int *&intDat, float *&floatDat, char *&charDat);
    virtual void internal_write(ostream &str,
                                int *&intDat, float *&floatDat, char *&charDat) const;

    coTetin__param_surface();
    ~coTetin__param_surface();
};
#endif
