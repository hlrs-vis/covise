/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __D_Primitive_H
#define __D_Primitive_H

#ifdef WIN32
#include <windows.h>
#pragma warning(disable : 4786)
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include <osg/Node>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osg/Material>
#include <osg/ref_ptr>
#include "basicelement.h"

const int LINE_WIDTH_DEFAULT = 1;

class parmblock;

////////////////////////////////////////////////////////////////////////

class D_Primitive : public BasicElement
{
public:
    inline D_Primitive();
    D_Primitive(parmblock &block);
    ~D_Primitive(){};

public:
    virtual inline void Define();
    inline void set_bodyid(int id);
    inline void set_forceid(int id);
    inline int bodyid() const;
    inline int forceid() const;

    inline std::string type() const;

    void set_mytriads(GLfloat *AA);
    osg::MatrixTransform *getNode()
    {
        return trans;
    };

    static osg::ref_ptr<osg::Material> globalmtl;

protected:
    void setMaterial();
    void applyTrans();
    osg::MatrixTransform *trans;
    std::string type_val;
    int bodyid_val, forceid_val;
    osg::Vec3 shift;
    osg::Vec3 rotat;
    osg::Vec4 color;
    osg::Vec3Array *vert;
    osg::Vec3Array *norm;
    osg::Geometry *geom;
    osg::Geode *geode;
    // only used within force elements
    GLfloat triadfrom[3];
    GLfloat triadto[3];
};

inline D_Primitive::D_Primitive()
    : bodyid_val()
    , forceid_val()
{
}

inline void D_Primitive::Define()
{
}

inline void D_Primitive::set_bodyid(int id)
{
    bodyid_val = id;
}

inline void D_Primitive::set_forceid(int id)
{
    forceid_val = id;
}

inline int D_Primitive::bodyid() const
{
    return bodyid_val;
}

inline int D_Primitive::forceid() const
{
    return forceid_val;
}

inline std::string D_Primitive::type() const
{
    return type_val;
}

////////////////////////////////////////////////////////////////////////

class World : public D_Primitive
{
public:
    World()
    {
    }
    World(parmblock &block);

public:
    void GenD_Primitive(GLenum surf);
};

////////////////////////////////////////////////////////////////////////

class D_Box : public D_Primitive
{
public:
    D_Box(parmblock &block);

public:
private:
    osg::Vec3 size;
};

////////////////////////////////////////////////////////////////////////

class D_Sphere : public D_Primitive
{
public:
    D_Sphere(parmblock &block);

public:
    void Surface(GLfloat *v, GLfloat *n,
                 const GLfloat rx, const GLfloat ry, const GLfloat rz,
                 const GLfloat phi, const GLfloat theta);

private:
    GLfloat rx, ry, rz;
    int nphi, ntheta;
};

////////////////////////////////////////////////////////////////////////

class D_Cylinder : public D_Primitive
{
public:
    D_Cylinder(parmblock &block);

public:
private:
    double rx_bot, ry_bot, rx_top, ry_top;
    double h;
    int nphi;
};

////////////////////////////////////////////////////////////////////////

class D_Cone : public D_Primitive
{
public:
    D_Cone(parmblock &block);

public:
private:
    double rx, ry, h;
    int nphi;
};

////////////////////////////////////////////////////////////////////////

class D_Axes : public D_Primitive
{
public:
    D_Axes(parmblock &block);

public:
private:
    GLfloat length;
    osg::Vec4 colorX, colorY;
};

////////////////////////////////////////////////////////////////////////

class D_Extrude : public D_Primitive
{
public:
    D_Extrude(parmblock &block);
    virtual ~D_Extrude();

public:
private:
    GLfloat **vtxbuf;
    int nvtx;
    GLfloat *vtx_x;
    GLfloat *vtx_y;
    GLfloat *vtx_z;
    double h;
};

////////////////////////////////////////////////////////////////////////

class D_Tetraeder : public D_Primitive
{
public:
    D_Tetraeder(parmblock &block);

public:
private:
    GLfloat vtx1[3];
    GLfloat vtx2[3];
    GLfloat vtx3[3];
    GLfloat vtx4[3];
};

////////////////////////////////////////////////////////////////////////

class D_Surface : public D_Primitive
{
public:
    D_Surface(parmblock &block);

public:
private:
    std::string sname;
    char *surfacename;
    GLfloat scale;
};

////////////////////////////////////////////////////////////////////////

class D_Muscle : public D_Primitive
{
public:
    D_Muscle(parmblock &block);

public:
private:
    std::string myforcename;
    double linethickness;
};

////////////////////////////////////////////////////////////////////////

class D_IVD : public D_Primitive
{
public:
    D_IVD(parmblock &block);

public:
private:
    std::string myforcename;
    double myradius;
};

#endif // __D_Primitive_H
