/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// Klasse HlImplizitFunktion -------------------------------- //
#ifndef __HlPARAMETRICFUNCTION3D__
#define __HlPARAMETRICFUNCTION3D__

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "hlvector.h"
#include "hlcas.h"

typedef vector<HlExprList *> ExprListVector;

class HlParametricFunction3d
{

private:
    int _maxDiffOrder;
    int _aktualDiffOrder;
    double *_t;

    ExprListVector _dx;
    ExprListVector _dy;
    ExprListVector _dz;
    HlExprList *_kappa;
    HlExprList *_dkappat;

    double _taylorT;
    HlVector _taylorKoeff[30];

    string mTLexem;

    void nullKappa()
    {
        _kappa = _dkappat = NULL;
    }
    void nullAll()
    {
        nullKappa();
    }

    void deleteX()
    {
        deleteDX();
    }
    void deleteY()
    {
        deleteDY();
    }
    void deleteZ()
    {
        deleteDZ();
    }
    void deleteDX()
    {
        for (unsigned int i = 0; i < _dx.size(); i++)
            delete _dx[i];
        _dx.clear();
    }
    void deleteDY()
    {
        for (unsigned int i = 0; i < _dy.size(); i++)
            delete _dy[i];
        _dy.clear();
    }
    void deleteDZ()
    {
        for (unsigned int i = 0; i < _dz.size(); i++)
            delete _dz[i];
        _dz.clear();
    }
    void deleteKappa()
    {
        delete _kappa, _dkappat;
        nullKappa();
    }
    void deleteAll()
    {
        deleteX();
        deleteY();
        deleteZ();
        deleteKappa();
    }

public:
    HlParametricFunction3d();
    ~HlParametricFunction3d();
    HlParametricFunction3d(const string &tname);
    void setTName(const string &tname)
    {
        mTLexem = tname;
        _t = HLCAS.getValPtr(mTLexem);
    }

    double dx(int n, double t)
    {
        return HLCAS.evalf(_dx[n], _t, t);
    }
    double dy(int n, double t)
    {
        return HLCAS.evalf(_dy[n], _t, t);
    }
    double dz(int n, double t)
    {
        return HLCAS.evalf(_dz[n], _t, t);
    }
    double x(double t)
    {
        return HLCAS.evalf(_dx[0], _t, t);
    }
    double y(double t)
    {
        return HLCAS.evalf(_dy[0], _t, t);
    }
    double z(double t)
    {
        return HLCAS.evalf(_dz[0], _t, t);
    }
    double dxt(double t)
    {
        return HLCAS.evalf(_dx[1], _t, t);
    }
    double dyt(double t)
    {
        return HLCAS.evalf(_dy[1], _t, t);
    }
    double dzt(double t)
    {
        return HLCAS.evalf(_dz[1], _t, t);
    }
    double dxtt(double t)
    {
        return HLCAS.evalf(_dx[2], _t, t);
    }
    double dytt(double t)
    {
        return HLCAS.evalf(_dy[2], _t, t);
    }
    double dztt(double t)
    {
        return HLCAS.evalf(_dz[2], _t, t);
    }
    double dxttt(double t)
    {
        return HLCAS.evalf(_dx[3], _t, t);
    }
    double dyttt(double t)
    {
        return HLCAS.evalf(_dy[3], _t, t);
    }
    double dzttt(double t)
    {
        return HLCAS.evalf(_dz[3], _t, t);
    }

    HlVector df(int n, double t)
    {
        return HlVector(dx(n, t), dy(n, t), dz(n, t));
    }
    HlVector f(double t)
    {
        return HlVector(df(0, t));
    }
    HlVector dft(double t)
    {
        return HlVector(df(1, t));
    }
    HlVector dftt(double t)
    {
        return HlVector(df(2, t));
    }
    HlVector dfttt(double t)
    {
        return HlVector(df(3, t));
    }

    HlVector T(double t)
    {
        return dft(t).normiert();
    }
    HlVector N(double t)
    {
        return (B(t) % T(t));
    }
    HlVector B(double t)
    {
        return (dft(t) % dftt(t)).normiert();
    }

    double nKappa(double t);
    double dKappat(double t);
    double ndKappat(double t);
    double Tau(double t);

    double Bogenlaenge(double t1, double t2);

    bool SetFunktionX(const string &s);
    bool SetFunktionY(const string &s);
    bool SetFunktionZ(const string &s);

    const string &getXString()
    {
        return _dx[0]->fktstr();
    }
    const string &getYString()
    {
        return _dy[0]->fktstr();
    }
    const string &getZString()
    {
        return _dz[0]->fktstr();
    }

    const string &getDXTString()
    {
        return _dx[1]->fktstr();
    }
    const string &getDYTString()
    {
        return _dy[1]->fktstr();
    }
    const string &getDZTString()
    {
        return _dz[1]->fktstr();
    }

    const string &getDXTTString()
    {
        return _dx[2]->fktstr();
    }
    const string &getDYTTString()
    {
        return _dy[2]->fktstr();
    }
    const string &getDZTTString()
    {
        return _dz[2]->fktstr();
    }

    const string &getDXTTTString()
    {
        return _dx[3]->fktstr();
    }
    const string &getDYTTTString()
    {
        return _dy[3]->fktstr();
    }
    const string &getDZTTTString()
    {
        return _dz[3]->fktstr();
    }

    bool isDepend(const string &fkt, const string &var);

    void setTaylorT(double t);
    double getTaylorT()
    {
        return _taylorT;
    };
    HlVector Taylor(double t, int n);
};

inline HlParametricFunction3d::HlParametricFunction3d()
{
    _maxDiffOrder = 7;
    _aktualDiffOrder = 3;
    nullAll();
    setTName("t");
}

inline HlParametricFunction3d::HlParametricFunction3d(const string &tname)
{
    nullAll();
    setTName(tname);
}

inline HlParametricFunction3d::~HlParametricFunction3d()
{
    deleteAll();
}

inline bool HlParametricFunction3d::SetFunktionX(const string &s)
{
    deleteX();
    deleteKappa();

    _dx.push_back(HLCAS.parseString(s));
    for (int i = 0; i < _aktualDiffOrder; i++)
    {
        _dx.push_back(HLCAS.diffTo(_dx[i], mTLexem));
    }

    _t = HLCAS.getValPtr(mTLexem);

    return HLCAS.mError.noError();
}

inline bool HlParametricFunction3d::SetFunktionY(const string &s)
{
    deleteY();
    deleteKappa();

    _dy.push_back(HLCAS.parseString(s));
    for (int i = 0; i < _aktualDiffOrder; i++)
    {
        _dy.push_back(HLCAS.diffTo(_dy[i], mTLexem));
    }

    _t = HLCAS.getValPtr(mTLexem);

    return _dy[0]->ok();
}

inline bool HlParametricFunction3d::SetFunktionZ(const string &s)
{
    deleteZ();
    deleteKappa();

    _dz.push_back(HLCAS.parseString(s));
    for (int i = 0; i < _aktualDiffOrder; i++)
    {
        _dz.push_back(HLCAS.diffTo(_dz[i], mTLexem));
    }

    _t = HLCAS.getValPtr(mTLexem);

    return _dz[0]->ok();
}

inline double HlParametricFunction3d::nKappa(double t)
{
    HlVector ft = dft(t);
    HlVector ftt = dftt(t);
    double lft = ft.betrag();
    return (ft % ftt).betrag() / (lft * lft * lft);
}

inline double HlParametricFunction3d::dKappat(double t)
{
    if (_dkappat == NULL)
    {
        if (_kappa != NULL)
        {
            _dkappat = HLCAS.diffTo(_kappa, mTLexem);
        }
    }
    return HLCAS.evalf(_dkappat, _t, t);
}

inline double HlParametricFunction3d::ndKappat(double t)
{
    double xt = dxt(t);
    double yt = dyt(t);
    double zt = dzt(t);
    double xtt = dxtt(t);
    double ytt = dytt(t);
    double ztt = dztt(t);
    double xttt = dxttt(t);
    double yttt = dyttt(t);
    double zttt = dzttt(t);

    double a = yt * ztt - ytt * zt;
    double b = xtt * zt - xt * ztt;
    double c = xt * ytt - xtt * yt;

    double h1 = a * (yt * zttt - yttt * zt);
    double h2 = b * (xttt * zt - xt * zttt);
    double h3 = c * (xt * yttt - xttt * yt);

    double h4 = a * a + b * b + c * c;
    double h7 = xt * xt + yt * yt + zt * zt;

    double h5 = pow(h7, 1.5);

    double h6 = 2 * (h1 + h2 + h3) / (sqrt(h4) * h5);

    double h8 = 2 * (xt * xtt + yt * ytt + zt * ztt);
    double h9 = h5 * h7;
    double h10 = (sqrt(h4) * h8) / h9;

    double h11 = 0.5 * h6 - 1.5 * h10;

    return h11;
}

inline double HlParametricFunction3d::Tau(double t)
{
    HlVector dt = dft(t);
    HlVector dtt = dftt(t);
    HlVector dttt = dfttt(t);

    return spat(dt, dtt, dttt) / (dt % dtt).quadrat();
}

inline double HlParametricFunction3d::Bogenlaenge(double t1, double t2)
{
    const int N = 10;
    double H = (t2 - t1) / N;
    double h = H / 2.0;

    double v;
    int j;

    double s1 = 0;
    double s2 = 0;

    for (j = 0; j < N; j++)
    {
        s1 += dft(t1 + (2 * j + 1) * h).betrag();
    }

    for (j = 1; j < N; j++)
    {
        s2 += dft(t1 + (2 * j) * h).betrag();
    }

    v = (h / 3) * (dft(t1).betrag() + 4 * s1 + 2 * s2 + dft(t2).betrag());

    return v;
}

inline void HlParametricFunction3d::setTaylorT(double t)
{
    _taylorT = t;
    for (int n = 0; n <= _aktualDiffOrder; n++)
    {
        _taylorKoeff[n] = df(n, _taylorT);
    }
}

inline HlVector HlParametricFunction3d::Taylor(double t, int n)
{
    if (n > _aktualDiffOrder && n <= _maxDiffOrder)
    {
        for (int i = _aktualDiffOrder; i <= n; i++)
        {
            _dx.push_back(HLCAS.diffTo(_dx[i], mTLexem));
            _dy.push_back(HLCAS.diffTo(_dy[i], mTLexem));
            _dz.push_back(HLCAS.diffTo(_dz[i], mTLexem));
            _taylorKoeff[i] = df(i, _taylorT);
        }
        _aktualDiffOrder = n;
    }

    if (n > _maxDiffOrder || n < 0)
        return HlVector(0, 0);

    double tt = 1;
    HlVector v = _taylorKoeff[0];

    for (int i = 1; i <= n; i++)
    {
        tt *= (t - _taylorT) / double(i);
        v += _taylorKoeff[i] * tt;
    }

    return v;
}

inline bool HlParametricFunction3d::isDepend(const string &fkt, const string &var)
{
    if (fkt == "X")
        return HLCAS.depend(_dx[0], var);
    if (fkt == "Y")
        return HLCAS.depend(_dy[0], var);
    if (fkt == "Z")
        return HLCAS.depend(_dz[0], var);
    return false;
}

#endif // __HlParametricFunktion3d__
