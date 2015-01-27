/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Klasse __HlParametricSurface3d__ -------------------------------- //
#ifndef __HlPARAMETRICSURFACE3D__
#define __HlPARAMETRICSURFACE3D__

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "hlvector.h"
#include "HlCAS.h"

/*-----------------------------------------------------------*/

class HlParametricSurface3d
{
private:
    bool _defined;

    string mULexem;
    string mVLexem;

    double *_u;
    double *_v;

    double *_a;
    double *_b;
    double *_c;

    HlExprList *_xuv;
    HlExprList *_yuv;
    HlExprList *_zuv;

    HlExprList *_dxu;
    HlExprList *_dyu;
    HlExprList *_dzu;
    HlExprList *_dxv;
    HlExprList *_dyv;
    HlExprList *_dzv;

    HlExprList *_dxuu;
    HlExprList *_dyuu;
    HlExprList *_dzuu;
    HlExprList *_dxvv;
    HlExprList *_dyvv;
    HlExprList *_dzvv;
    HlExprList *_dxuv;
    HlExprList *_dyuv;
    HlExprList *_dzuv;

    void nullX()
    {
        _xuv = _dxu = _dxv = _dxuu = _dxvv = _dxuv = NULL;
    }
    void nullY()
    {
        _yuv = _dyu = _dyv = _dyuu = _dyvv = _dyuv = NULL;
    }
    void nullZ()
    {
        _zuv = _dzu = _dzv = _dzuu = _dzvv = _dzuv = NULL;
    }
    void nullAll()
    {
        nullX();
        nullY();
        nullZ();
    }

    void deleteX()
    {
        delete _xuv;
        delete _dxu;
        delete _dxv;
        delete _dxuu;
        delete _dxvv;
        delete _dxuv;
        nullX();
    }
    void deleteY()
    {
        delete _yuv;
        delete _dyu;
        delete _dyv;
        delete _dyuu;
        delete _dyvv;
        delete _dyuv;
        nullY();
    }
    void deleteZ()
    {
        delete _zuv;
        delete _dzu;
        delete _dzv;
        delete _dzuu;
        delete _dzvv;
        delete _dzuv;
        nullZ();
    }
    void deleteAll()
    {
        deleteX();
        deleteY();
        deleteZ();
    }

public:
    HlParametricSurface3d();
    ~HlParametricSurface3d();
    HlParametricSurface3d(const string &uname, const string &vname);
    void setUName(const string &uname)
    {
        mULexem = uname;
        _u = HLCAS.getValPtr(mULexem);
    }
    void setVName(const string &vname)
    {
        mVLexem = vname;
        _v = HLCAS.getValPtr(mVLexem);
    }
    void setVarNames(const string &u, const string &v)
    {
        setUName(u), setVName(v);
    }
    const string &getUName()
    {
        return mULexem;
    }
    const string &getVName()
    {
        return mVLexem;
    }

    double x(double u, double v)
    {
        return HLCAS.evalf(_xuv, _u, u, _v, v);
    }
    double y(double u, double v)
    {
        return HLCAS.evalf(_yuv, _u, u, _v, v);
    }
    double z(double u, double v)
    {
        return HLCAS.evalf(_zuv, _u, u, _v, v);
    }
    double dxu(double u, double v)
    {
        return HLCAS.evalf(_dxu, _u, u, _v, v);
    }
    double dyu(double u, double v)
    {
        return HLCAS.evalf(_dyu, _u, u, _v, v);
    }
    double dzu(double u, double v)
    {
        return HLCAS.evalf(_dzu, _u, u, _v, v);
    }
    double dxv(double u, double v)
    {
        return HLCAS.evalf(_dxv, _u, u, _v, v);
    }
    double dyv(double u, double v)
    {
        return HLCAS.evalf(_dyv, _u, u, _v, v);
    }
    double dzv(double u, double v)
    {
        return HLCAS.evalf(_dzv, _u, u, _v, v);
    }
    double dxuu(double u, double v)
    {
        return HLCAS.evalf(_dxuu, _u, u, _v, v);
    }
    double dyuu(double u, double v)
    {
        return HLCAS.evalf(_dyuu, _u, u, _v, v);
    }
    double dzuu(double u, double v)
    {
        return HLCAS.evalf(_dzuu, _u, u, _v, v);
    }
    double dxvv(double u, double v)
    {
        return HLCAS.evalf(_dxvv, _u, u, _v, v);
    }
    double dyvv(double u, double v)
    {
        return HLCAS.evalf(_dyvv, _u, u, _v, v);
    }
    double dzvv(double u, double v)
    {
        return HLCAS.evalf(_dzvv, _u, u, _v, v);
    }
    double dxuv(double u, double v)
    {
        return HLCAS.evalf(_dxuv, _u, u, _v, v);
    }
    double dyuv(double u, double v)
    {
        return HLCAS.evalf(_dyuv, _u, u, _v, v);
    }
    double dzuv(double u, double v)
    {
        return HLCAS.evalf(_dzuv, _u, u, _v, v);
    }

    double E(double u, double v)
    {
        return dfu(u, v).quadrat();
    }
    double F(double u, double v)
    {
        return dfu(u, v) * dfv(u, v);
    }
    double G(double u, double v)
    {
        return dfv(u, v).quadrat();
    }
    double EG_FF(double u, double v)
    {
        double f = F(u, v);
        return E(u, v) * G(u, v) - f * f;
    }
    double L(double u, double v)
    {
        return spat(dfu(u, v), dfv(u, v), dfuu(u, v)) / sqrt(EG_FF(u, v));
    }
    double M(double u, double v)
    {
        return spat(dfu(u, v), dfv(u, v), dfuv(u, v)) / sqrt(EG_FF(u, v));
    }
    double N(double u, double v)
    {
        return spat(dfu(u, v), dfv(u, v), dfvv(u, v)) / sqrt(EG_FF(u, v));
    }

    double K(double u, double v);
    double H(double u, double v);

    HlVector f(double u, double v)
    {
        _defined = true;
        return HlVector(x(u, v), y(u, v), z(u, v));
    }
    HlVector dfu(double u, double v)
    {
        _defined = true;
        return HlVector(dxu(u, v), dyu(u, v), dzu(u, v));
    }
    HlVector dfv(double u, double v)
    {
        _defined = true;
        return HlVector(dxv(u, v), dyv(u, v), dzv(u, v));
    }
    HlVector dfuu(double u, double v)
    {
        _defined = true;
        return HlVector(dxuu(u, v), dyuu(u, v), dzuu(u, v));
    }
    HlVector dfvv(double u, double v)
    {
        _defined = true;
        return HlVector(dxvv(u, v), dyvv(u, v), dzvv(u, v));
    }
    HlVector dfuv(double u, double v)
    {
        _defined = true;
        return HlVector(dxuv(u, v), dyuv(u, v), dzuv(u, v));
    }
    HlVector nvek(double u, double v)
    {
        _defined = true;
        return dfu(u, v) % dfv(u, v);
    }

    bool SetFunktionX(const string &s);
    bool SetFunktionY(const string &s);
    bool SetFunktionZ(const string &s);

    void SetA(double a)
    {
        *_a = a;
    }
    void SetB(double b)
    {
        *_b = b;
    }
    void SetC(double c)
    {
        *_c = c;
    }

    double GetA()
    {
        return *_a;
    }
    double GetB()
    {
        return *_b;
    }
    double GetC()
    {
        return *_c;
    }

    bool isDefined()
    {
        return _defined;
    }
};

inline HlParametricSurface3d::HlParametricSurface3d()
{
    nullAll();
    setVarNames("u", "v");
    HLCAS.evalString("a=1");
    HLCAS.evalString("b=1");
    HLCAS.evalString("c=1");
    _a = HLCAS.getValPtr("a");
    _b = HLCAS.getValPtr("b");
    _c = HLCAS.getValPtr("c");
}

inline HlParametricSurface3d::HlParametricSurface3d(const string &uname, const string &vname)
{
    nullAll();
    setVarNames(uname, vname);
    HLCAS.evalString("a=1");
    HLCAS.evalString("b=1");
    HLCAS.evalString("c=1");
    _a = HLCAS.getValPtr("a");
    _b = HLCAS.getValPtr("b");
    _c = HLCAS.getValPtr("c");
}

inline HlParametricSurface3d::~HlParametricSurface3d()
{
    deleteAll();
}

inline double HlParametricSurface3d::K(double u, double v)
{
    HlVector du = dfu(u, v);
    HlVector dv = dfv(u, v);
    HlVector duu = dfuu(u, v);
    HlVector dvv = dfvv(u, v);
    HlVector duv = dfuv(u, v);
    double e = du * du;
    double f = du * dv;
    double g = dv * dv;
    double egff = e * g - f * f;
    double w = sqrt(egff);
    double l = spat(du, dv, duu) / w;
    double m = spat(du, dv, duv) / w;
    double n = spat(du, dv, dvv) / w;

    return (l * n - m * m) / egff;
}

inline double HlParametricSurface3d::H(double u, double v)
{
    HlVector du = dfu(u, v);
    HlVector dv = dfv(u, v);
    HlVector duu = dfuu(u, v);
    HlVector dvv = dfvv(u, v);
    HlVector duv = dfuv(u, v);
    double e = du * du;
    double f = du * dv;
    double g = dv * dv;
    double egff = e * g - f * f;
    double w = sqrt(egff);
    double l = spat(du, dv, duu) / w;
    double m = spat(du, dv, duv) / w;
    double n = spat(du, dv, dvv) / w;

    return 0.5 * (e * n + l * g - 2 * f * m) / egff;
}

inline bool HlParametricSurface3d::SetFunktionX(const string &s)
{
    deleteX();

    _xuv = HLCAS.parseString(s);
    _dxu = HLCAS.diffTo(_xuv, mULexem);
    _dxv = HLCAS.diffTo(_xuv, mVLexem);
    _dxuu = HLCAS.diffTo(_dxu, mULexem);
    _dxvv = HLCAS.diffTo(_dxv, mVLexem);
    _dxuv = HLCAS.diffTo(_dxu, mVLexem);
    _u = HLCAS.getValPtr(mULexem);
    _v = HLCAS.getValPtr(mVLexem);

    return _xuv->ok();
}

inline bool HlParametricSurface3d::SetFunktionY(const string &s)
{
    deleteY();

    _yuv = HLCAS.parseString(s);
    _dyu = HLCAS.diffTo(_yuv, mULexem);
    _dyv = HLCAS.diffTo(_yuv, mVLexem);
    _dyuu = HLCAS.diffTo(_dyu, mULexem);
    _dyvv = HLCAS.diffTo(_dyv, mVLexem);
    _dyuv = HLCAS.diffTo(_dyu, mVLexem);

    _u = HLCAS.getValPtr(mULexem);
    _v = HLCAS.getValPtr(mVLexem);

    return _yuv->ok();
}

inline bool HlParametricSurface3d::SetFunktionZ(const string &s)
{
    deleteZ();

    _zuv = HLCAS.parseString(s);
    _dzu = HLCAS.diffTo(_zuv, mULexem);
    _dzv = HLCAS.diffTo(_zuv, mVLexem);
    _dzuu = HLCAS.diffTo(_dzu, mULexem);
    _dzvv = HLCAS.diffTo(_dzv, mVLexem);
    _dzuv = HLCAS.diffTo(_dzu, mVLexem);

    _u = HLCAS.getValPtr(mULexem);
    _v = HLCAS.getValPtr(mVLexem);

    return _zuv->ok();
}

#endif // __HlPARAMETRICSURFACE3D__
