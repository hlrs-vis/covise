/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NR_H_
#define _NR_H_
#include <fstream>
#include <complex>
#include "nrutil.h"
#include "nrtypes.h"
using namespace std;

namespace NR
{

void addint(Mat_O_DP &uf, Mat_I_DP &uc, Mat_O_DP &res);
void airy(const DP x, DP &ai, DP &bi, DP &aip, DP &bip);
void amebsa(Mat_IO_DP &p, Vec_IO_DP &y, Vec_O_DP &pb, DP &yb, const DP ftol,
            DP funk(Vec_I_DP &), int &iter, const DP temptr);
void amoeba(Mat_IO_DP &p, Vec_IO_DP &y, const DP ftol, DP funk(Vec_I_DP &),
            int &nfunk);
DP amotry(Mat_IO_DP &p, Vec_O_DP &y, Vec_IO_DP &psum, DP funk(Vec_I_DP &),
          const int ihi, const DP fac);
DP amotsa(Mat_IO_DP &p, Vec_O_DP &y, Vec_IO_DP &psum, Vec_O_DP &pb, DP &yb,
          DP funk(Vec_I_DP &), const int ihi, DP &yhi, const DP fac);
void anneal(Vec_I_DP &x, Vec_I_DP &y, Vec_IO_INT &iorder);
DP anorm2(Mat_I_DP &a);
void arcmak(Vec_I_ULNG &nfreq, unsigned long nchh, unsigned long nradd,
            arithcode &acode);
void arcode(unsigned long &ich, string &code, unsigned long &lcd,
            const int isign, arithcode &acode);
void arcsum(Vec_I_ULNG &iin, Vec_O_ULNG &iout, unsigned long ja,
            const int nwk, const unsigned long nrad, const unsigned long nc);
void asolve(Vec_I_DP &b, Vec_O_DP &x, const int itrnsp);
void atimes(Vec_I_DP &x, Vec_O_DP &r, const int itrnsp);
void avevar(Vec_I_DP &data, DP &ave, DP &var);
void balanc(Mat_IO_DP &a);
void banbks(Mat_I_DP &a, const int m1, const int m2, Mat_I_DP &al,
            Vec_I_INT &indx, Vec_IO_DP &b);
void bandec(Mat_IO_DP &a, const int m1, const int m2, Mat_O_DP &al,
            Vec_O_INT &indx, DP &d);
void banmul(Mat_I_DP &a, const int m1, const int m2, Vec_I_DP &x,
            Vec_O_DP &b);
void bcucof(Vec_I_DP &y, Vec_I_DP &y1, Vec_I_DP &y2, Vec_I_DP &y12,
            const DP d1, const DP d2, Mat_O_DP &c);
void bcuint(Vec_I_DP &y, Vec_I_DP &y1, Vec_I_DP &y2, Vec_I_DP &y12,
            const DP x1l, const DP x1u, const DP x2l, const DP x2u,
            const DP x1, const DP x2, DP &ansy, DP &ansy1, DP &ansy2);
void beschb(const DP x, DP &gam1, DP &gam2, DP &gampl, DP &gammi);
DP bessi(const int n, const DP x);
DP bessi0(const DP x);
DP bessi1(const DP x);
void bessik(const DP x, const DP xnu, DP &ri, DP &rk, DP &rip, DP &rkp);
DP bessj(const int n, const DP x);
DP bessj0(const DP x);
DP bessj1(const DP x);
void bessjy(const DP x, const DP xnu, DP &rj, DP &ry, DP &rjp, DP &ryp);
DP bessk(const int n, const DP x);
DP bessk0(const DP x);
DP bessk1(const DP x);
DP bessy(const int n, const DP x);
DP bessy0(const DP x);
DP bessy1(const DP x);
DP beta(const DP z, const DP w);
DP betacf(const DP a, const DP b, const DP x);
DP betai(const DP a, const DP b, const DP x);
DP bico(const int n, const int k);
void bksub(const int ne, const int nb, const int jf, const int k1,
           const int k2, Mat3D_IO_DP &c);
DP bnldev(const DP pp, const int n, int &idum);
DP brent(const DP ax, const DP bx, const DP cx, DP f(const DP),
         const DP tol, DP &xmin);
void broydn(Vec_IO_DP &x, bool &check, void vecfunc(Vec_I_DP &, Vec_O_DP &));
void bsstep(Vec_IO_DP &y, Vec_IO_DP &dydx, DP &xx, const DP htry,
            const DP eps, Vec_I_DP &yscal, DP &hdid, DP &hnext,
            void derivs(const DP, Vec_I_DP &, Vec_O_DP &));
void caldat(const int julian, int &mm, int &id, int &iyyy);
void chder(const DP a, const DP b, Vec_I_DP &c, Vec_O_DP &cder, const int n);
DP chebev(const DP a, const DP b, Vec_I_DP &c, const int m, const DP x);
void chebft(const DP a, const DP b, Vec_O_DP &c, DP func(const DP));
void chebpc(Vec_I_DP &c, Vec_O_DP &d);
void chint(const DP a, const DP b, Vec_I_DP &c, Vec_O_DP &cint, const int n);
DP chixy(const DP bang);
void choldc(Mat_IO_DP &a, Vec_O_DP &p);
void cholsl(Mat_I_DP &a, Vec_I_DP &p, Vec_I_DP &b, Vec_O_DP &x);
void chsone(Vec_I_DP &bins, Vec_I_DP &ebins, const int knstrn, DP &df,
            DP &chsq, DP &prob);
void chstwo(Vec_I_DP &bins1, Vec_I_DP &bins2, const int knstrn, DP &df,
            DP &chsq, DP &prob);
void cisi(const DP x, complex<DP> &cs);
void cntab1(Mat_I_INT &nn, DP &chisq, DP &df, DP &prob, DP &cramrv, DP &ccc);
void cntab2(Mat_I_INT &nn, DP &h, DP &hx, DP &hy, DP &hygx, DP &hxgy,
            DP &uygx, DP &uxgy, DP &uxy);
void convlv(Vec_I_DP &data, Vec_I_DP &respns, const int isign,
            Vec_O_DP &ans);
void copy(Mat_O_DP &aout, Mat_I_DP &ain);
void correl(Vec_I_DP &data1, Vec_I_DP &data2, Vec_O_DP &ans);
void cosft1(Vec_IO_DP &y);
void cosft2(Vec_IO_DP &y, const int isign);
void covsrt(Mat_IO_DP &covar, Vec_I_BOOL &ia, const int mfit);
void crank(Vec_IO_DP &w, DP &s);
void cyclic(Vec_I_DP &a, Vec_I_DP &b, Vec_I_DP &c, const DP alpha,
            const DP beta, Vec_I_DP &r, Vec_O_DP &x);
void daub4(Vec_IO_DP &a, const int n, const int isign);
DP dawson(const DP x);
DP dbrent(const DP ax, const DP bx, const DP cx, DP f(const DP),
          DP df(const DP), const DP tol, DP &xmin);
void ddpoly(Vec_I_DP &c, const DP x, Vec_O_DP &pd);
bool decchk(string str, char &ch);
void derivs_s(const DP x, Vec_I_DP &y, Vec_O_DP &dydx);
DP df1dim(const DP x);
void dfpmin(Vec_IO_DP &p, const DP gtol, int &iter, DP &fret,
            DP func(Vec_I_DP &), void dfunc(Vec_I_DP &, Vec_O_DP &));
DP dfridr(DP func(const DP), const DP x, const DP h, DP &err);
void dftcor(const DP w, const DP delta, const DP a, const DP b,
            Vec_I_DP &endpts, DP &corre, DP &corim, DP &corfac);
void dftint(DP func(const DP), const DP a, const DP b, const DP w,
            DP &cosint, DP &sinint);
void difeq(const int k, const int k1, const int k2, const int jsf,
           const int is1, const int isf, Vec_I_INT &indexv, Mat_O_DP &s,
           Mat_I_DP &y);
void dlinmin(Vec_IO_DP &p, Vec_IO_DP &xi, DP &fret, DP func(Vec_I_DP &),
             void dfunc(Vec_I_DP &, Vec_O_DP &));
void eclass(Vec_O_INT &nf, Vec_I_INT &lista, Vec_I_INT &listb);
void eclazz(Vec_O_INT &nf, bool equiv(const int, const int));
DP ei(const DP x);
void eigsrt(Vec_IO_DP &d, Mat_IO_DP &v);
DP elle(const DP phi, const DP ak);
DP ellf(const DP phi, const DP ak);
DP ellpi(const DP phi, const DP en, const DP ak);
void elmhes(Mat_IO_DP &a);
DP erfcc(const DP x);
DP erff(const DP x);
DP erffc(const DP x);
void eulsum(DP &sum, const DP term, const int jterm, Vec_IO_DP &wksp);
DP evlmem(const DP fdt, Vec_I_DP &d, const DP xms);
DP expdev(int &idum);
DP expint(const int n, const DP x);
DP f1dim(const DP x);
DP factln(const int n);
DP factrl(const int n);
void fasper(Vec_I_DP &x, Vec_I_DP &y, const DP ofac, const DP hifac,
            Vec_O_DP &wk1, Vec_O_DP &wk2, int &nout, int &jmax, DP &prob);
void fdjac(Vec_IO_DP &x, Vec_I_DP &fvec, Mat_O_DP &df,
           void vecfunc(Vec_I_DP &, Vec_O_DP &));
void fgauss(const DP x, Vec_I_DP &a, DP &y, Vec_O_DP &dyda);
void fit(Vec_I_DP &x, Vec_I_DP &y, Vec_I_DP &sig, const bool mwt, DP &a,
         DP &b, DP &siga, DP &sigb, DP &chi2, DP &q);
void fitexy(Vec_I_DP &x, Vec_I_DP &y, Vec_I_DP &sigx, Vec_I_DP &sigy,
            DP &a, DP &b, DP &siga, DP &sigb, DP &chi2, DP &q);
void fixrts(Vec_IO_DP &d);
void fleg(const DP x, Vec_O_DP &pl);
void flmoon(const int n, const int nph, int &jd, DP &frac);
DP fmin(Vec_I_DP &x);
void four1(Vec_IO_DP &data, const int isign);
void fourew(Vec_FSTREAM_p &file, int &na, int &nb, int &nc, int &nd);
void fourfs(Vec_FSTREAM_p &file, Vec_I_INT &nn, const int isign);
void fourn(Vec_IO_DP &data, Vec_I_INT &nn, const int isign);
void fpoly(const DP x, Vec_O_DP &p);
void fred2(const DP a, const DP b, Vec_O_DP &t, Vec_O_DP &f, Vec_O_DP &w,
           DP g(const DP), DP ak(const DP, const DP));
DP fredin(const DP x, const DP a, const DP b, Vec_I_DP &t, Vec_I_DP &f,
          Vec_I_DP &w, DP g(const DP), DP ak(const DP, const DP));
void frenel(const DP x, complex<DP> &cs);
void frprmn(Vec_IO_DP &p, const DP ftol, int &iter, DP &fret,
            DP func(Vec_I_DP &), void dfunc(Vec_I_DP &, Vec_O_DP &));
void ftest(Vec_I_DP &data1, Vec_I_DP &data2, DP &f, DP &prob);
DP gamdev(const int ia, int &idum);
DP gammln(const DP xx);
DP gammp(const DP a, const DP x);
DP gammq(const DP a, const DP x);
DP gasdev(int &idum);
void gaucof(Vec_IO_DP &a, Vec_IO_DP &b, const DP amu0, Vec_O_DP &x,
            Vec_O_DP &w);
void gauher(Vec_O_DP &x, Vec_O_DP &w);
void gaujac(Vec_O_DP &x, Vec_O_DP &w, const DP alf, const DP bet);
void gaulag(Vec_O_DP &x, Vec_O_DP &w, const DP alf);
void gauleg(const DP x1, const DP x2, Vec_O_DP &x, Vec_O_DP &w);
void gaussj(Mat_IO_DP &a, Mat_IO_DP &b);
void gcf(DP &gammcf, const DP a, const DP x, DP &gln);
DP golden(const DP ax, const DP bx, const DP cx, DP f(const DP),
          const DP tol, DP &xmin);
void gser(DP &gamser, const DP a, const DP x, DP &gln);
void hpsel(Vec_I_DP &arr, Vec_O_DP &heap);
void hpsort(Vec_IO_DP &ra);
void hqr(Mat_IO_DP &a, Vec_O_CPLX_DP &wri);
void hufapp(Vec_IO_ULNG &index, Vec_I_ULNG &nprob, const unsigned long n,
            const unsigned long m);
void hufdec(unsigned long &ich, string &code, const unsigned long lcode,
            unsigned long &nb, huffcode &hcode);
void hufenc(const unsigned long ich, string &code, unsigned long &nb,
            huffcode &hcode);
void hufmak(Vec_I_ULNG &nfreq, const unsigned long nchin,
            unsigned long &ilong, unsigned long &nlong, huffcode &hcode);
void hunt(Vec_I_DP &xx, const DP x, int &jlo);
void hypdrv(const DP s, Vec_I_DP &yy, Vec_O_DP &dyyds);
complex<DP> hypgeo(const complex<DP> &a, const complex<DP> &b,
                   const complex<DP> &c, const complex<DP> &z);
void hypser(const complex<DP> &a, const complex<DP> &b,
            const complex<DP> &c, const complex<DP> &z,
            complex<DP> &series, complex<DP> &deriv);
unsigned short icrc(const unsigned short crc, const string &bufptr,
                    const short jinit, const int jrev);
unsigned short icrc1(const unsigned short crc, const unsigned char onech);
unsigned long igray(const unsigned long n, const int is);
void indexx(Vec_I_DP &arr, Vec_O_INT &indx);
void indexx(Vec_I_INT &arr, Vec_O_INT &indx);
void interp(Mat_O_DP &uf, Mat_I_DP &uc);
int irbit1(unsigned long &iseed);
int irbit2(unsigned long &iseed);
void jacobi(Mat_IO_DP &a, Vec_O_DP &d, Mat_O_DP &v, int &nrot);
void jacobn_s(const DP x, Vec_I_DP &y, Vec_O_DP &dfdx, Mat_O_DP &dfdy);
int julday(const int mm, const int id, const int iyyy);
void kendl1(Vec_I_DP &data1, Vec_I_DP &data2, DP &tau, DP &z, DP &prob);
void kendl2(Mat_I_DP &tab, DP &tau, DP &z, DP &prob);
void kermom(Vec_O_DP &w, const DP y);
void ks2d1s(Vec_I_DP &x1, Vec_I_DP &y1, void quadvl(const DP, const DP,
                                                    DP &, DP &, DP &, DP &),
            DP &d1, DP &prob);
void ks2d2s(Vec_I_DP &x1, Vec_I_DP &y1, Vec_I_DP &x2, Vec_I_DP &y2, DP &d,
            DP &prob);
void ksone(Vec_IO_DP &data, DP func(const DP), DP &d, DP &prob);
void kstwo(Vec_IO_DP &data1, Vec_IO_DP &data2, DP &d, DP &prob);
void laguer(Vec_I_CPLX_DP &a, complex<DP> &x, int &its);
void lfit(Vec_I_DP &x, Vec_I_DP &y, Vec_I_DP &sig, Vec_IO_DP &a,
          Vec_I_BOOL &ia, Mat_O_DP &covar, DP &chisq,
          void funcs(const DP, Vec_O_DP &));
void linbcg(Vec_I_DP &b, Vec_IO_DP &x, const int itol, const DP tol,
            const int itmax, int &iter, DP &err);
void linmin(Vec_IO_DP &p, Vec_IO_DP &xi, DP &fret, DP func(Vec_I_DP &));
void lnsrch(Vec_I_DP &xold, const DP fold, Vec_I_DP &g, Vec_IO_DP &p,
            Vec_O_DP &x, DP &f, const DP stpmax, bool &check, DP func(Vec_I_DP &));
void locate(Vec_I_DP &xx, const DP x, int &j);
void lop(Mat_O_DP &out, Mat_I_DP &u);
void lubksb(Mat_I_DP &a, Vec_I_INT &indx, Vec_IO_DP &b);
void ludcmp(Mat_IO_DP &a, Vec_O_INT &indx, DP &d);
void machar(int &ibeta, int &it, int &irnd, int &ngrd, int &machep,
            int &negep, int &iexp, int &minexp, int &maxexp, DP &eps, DP &epsneg,
            DP &xmin, DP &xmax);
void matadd(Mat_I_DP &a, Mat_I_DP &b, Mat_O_DP &c);
void matsub(Mat_I_DP &a, Mat_I_DP &b, Mat_O_DP &c);
void medfit(Vec_I_DP &x, Vec_I_DP &y, DP &a, DP &b, DP &abdev);
void memcof(Vec_I_DP &data, DP &xms, Vec_O_DP &d);
bool metrop(const DP de, const DP t);
void mgfas(Mat_IO_DP &u, const int maxcyc);
void mglin(Mat_IO_DP &u, const int ncycle);
DP midexp(DP funk(const DP), const DP aa, const DP bb, const int n);
DP midinf(DP funk(const DP), const DP aa, const DP bb, const int n);
DP midpnt(DP func(const DP), const DP a, const DP b, const int n);
DP midsql(DP funk(const DP), const DP aa, const DP bb, const int n);
DP midsqu(DP funk(const DP), const DP aa, const DP bb, const int n);
void miser(DP func(Vec_I_DP &), Vec_I_DP &regn, const int npts,
           const DP dith, DP &ave, DP &var);
void mmid(Vec_I_DP &y, Vec_I_DP &dydx, const DP xs, const DP htot,
          const int nstep, Vec_O_DP &yout,
          void derivs(const DP, Vec_I_DP &, Vec_O_DP &));
void mnbrak(DP &ax, DP &bx, DP &cx, DP &fa, DP &fb, DP &fc,
            DP func(const DP));
void mnewt(const int ntrial, Vec_IO_DP &x, const DP tolx, const DP tolf);
void moment(Vec_I_DP &data, DP &ave, DP &adev, DP &sdev, DP &var, DP &skew,
            DP &curt);
void mp2dfr(Vec_IO_UCHR &a, string &s);
void mpadd(Vec_O_UCHR &w, Vec_I_UCHR &u, Vec_I_UCHR &v);
void mpdiv(Vec_O_UCHR &q, Vec_O_UCHR &r, Vec_I_UCHR &u, Vec_I_UCHR &v);
void mpinv(Vec_O_UCHR &u, Vec_I_UCHR &v);
void mplsh(Vec_IO_UCHR &u);
void mpmov(Vec_O_UCHR &u, Vec_I_UCHR &v);
void mpmul(Vec_O_UCHR &w, Vec_I_UCHR &u, Vec_I_UCHR &v);
void mpneg(Vec_IO_UCHR &u);
void mppi(const int np);
void mprove(Mat_I_DP &a, Mat_I_DP &alud, Vec_I_INT &indx, Vec_I_DP &b,
            Vec_IO_DP &x);
void mpsad(Vec_O_UCHR &w, Vec_I_UCHR &u, const int iv);
void mpsdv(Vec_O_UCHR &w, Vec_I_UCHR &u, const int iv, int &ir);
void mpsmu(Vec_O_UCHR &w, Vec_I_UCHR &u, const int iv);
void mpsqrt(Vec_O_UCHR &w, Vec_O_UCHR &u, Vec_I_UCHR &v);
void mpsub(int &is, Vec_O_UCHR &w, Vec_I_UCHR &u, Vec_I_UCHR &v);
void mrqcof(Vec_I_DP &x, Vec_I_DP &y, Vec_I_DP &sig, Vec_I_DP &a,
            Vec_I_BOOL &ia, Mat_O_DP &alpha, Vec_O_DP &beta, DP &chisq,
            void funcs(const DP, Vec_I_DP &, DP &, Vec_O_DP &));
void mrqmin(Vec_I_DP &x, Vec_I_DP &y, Vec_I_DP &sig, Vec_IO_DP &a,
            Vec_I_BOOL &ia, Mat_O_DP &covar, Mat_O_DP &alpha, DP &chisq,
            void funcs(const DP, Vec_I_DP &, DP &, Vec_O_DP &), DP &alamda);
void newt(Vec_IO_DP &x, bool &check, void vecfunc(Vec_I_DP &, Vec_O_DP &));
void odeint(Vec_IO_DP &ystart, const DP x1, const DP x2, const DP eps,
            const DP h1, const DP hmin, int &nok, int &nbad,
            void derivs(const DP, Vec_I_DP &, Vec_O_DP &),
            void rkqs(Vec_IO_DP &, Vec_IO_DP &, DP &, const DP, const DP,
                      Vec_I_DP &, DP &, DP &, void (*)(const DP, Vec_I_DP &, Vec_O_DP &)));
void orthog(Vec_I_DP &anu, Vec_I_DP &alpha, Vec_I_DP &beta, Vec_O_DP &a,
            Vec_O_DP &b);
void pade(Vec_IO_DP &cof, DP &resid);
void pccheb(Vec_I_DP &d, Vec_O_DP &c);
void pcshft(const DP a, const DP b, Vec_IO_DP &d);
void pearsn(Vec_I_DP &x, Vec_I_DP &y, DP &r, DP &prob, DP &z);
void period(Vec_I_DP &x, Vec_I_DP &y, const DP ofac, const DP hifac,
            Vec_O_DP &px, Vec_O_DP &py, int &nout, int &jmax, DP &prob);
void piksr2(Vec_IO_DP &arr, Vec_IO_DP &brr);
void piksrt(Vec_IO_DP &arr);
void pinvs(const int ie1, const int ie2, const int je1, const int jsf,
           const int jc1, const int k, Mat3D_O_DP &c, Mat_IO_DP &s);
DP plgndr(const int l, const int m, const DP x);
DP poidev(const DP xm, int &idum);
void polcoe(Vec_I_DP &x, Vec_I_DP &y, Vec_O_DP &cof);
void polcof(Vec_I_DP &xa, Vec_I_DP &ya, Vec_O_DP &cof);
void poldiv(Vec_I_DP &u, Vec_I_DP &v, Vec_O_DP &q, Vec_O_DP &r);
void polin2(Vec_I_DP &x1a, Vec_I_DP &x2a, Mat_I_DP &ya, const DP x1,
            const DP x2, DP &y, DP &dy);
void polint(Vec_I_DP &xa, Vec_I_DP &ya, const DP x, DP &y, DP &dy);
void powell(Vec_IO_DP &p, Mat_IO_DP &xi, const DP ftol, int &iter,
            DP &fret, DP func(Vec_I_DP &));
void predic(Vec_I_DP &data, Vec_I_DP &d, Vec_O_DP &future);
DP probks(const DP alam);
void psdes(unsigned long &lword, unsigned long &irword);
void pwt(Vec_IO_DP &a, const int n, const int isign);
void pwtset(const int n);
DP pythag(const DP a, const DP b);
void pzextr(const int iest, const DP xest, Vec_I_DP &yest, Vec_O_DP &yz,
            Vec_O_DP &dy);
DP qgaus(DP func(const DP), const DP a, const DP b);
void qrdcmp(Mat_IO_DP &a, Vec_O_DP &c, Vec_O_DP &d, bool &sing);
DP qromb(DP func(const DP), DP a, DP b);
DP qromo(DP func(const DP), const DP a, const DP b,
         DP choose(DP (*)(const DP), const DP, const DP, const int));
void qroot(Vec_I_DP &p, DP &b, DP &c, const DP eps);
void qrsolv(Mat_I_DP &a, Vec_I_DP &c, Vec_I_DP &d, Vec_IO_DP &b);
void qrupdt(Mat_IO_DP &r, Mat_IO_DP &qt, Vec_IO_DP &u, Vec_I_DP &v);
DP qsimp(DP func(const DP), const DP a, const DP b);
DP qtrap(DP func(const DP), const DP a, const DP b);
DP quad3d(DP func(const DP, const DP, const DP), const DP x1, const DP x2);
void quadct(const DP x, const DP y, Vec_I_DP &xx, Vec_I_DP &yy, DP &fa,
            DP &fb, DP &fc, DP &fd);
void quadmx(Mat_O_DP &a);
void quadvl(const DP x, const DP y, DP &fa, DP &fb, DP &fc, DP &fd);
DP ran0(int &idum);
DP ran1(int &idum);
DP ran2(int &idum);
DP ran3(int &idum);
DP ran4(int &idum);
void rank(Vec_I_INT &indx, Vec_O_INT &irank);
void ranpt(Vec_O_DP &pt, Vec_I_DP &regn);
void ratint(Vec_I_DP &xa, Vec_I_DP &ya, const DP x, DP &y, DP &dy);
void ratlsq(DP fn(const DP), const DP a, const DP b, const int mm,
            const int kk, Vec_O_DP &cof, DP &dev);
DP ratval(const DP x, Vec_I_DP &cof, const int mm, const int kk);
DP rc(const DP x, const DP y);
DP rd(const DP x, const DP y, const DP z);
void realft(Vec_IO_DP &data, const int isign);
void rebin(const DP rc, const int nd, Vec_I_DP &r, Vec_O_DP &xin,
           Mat_IO_DP &xi, const int j);
void red(const int iz1, const int iz2, const int jz1, const int jz2,
         const int jm1, const int jm2, const int jmf, const int ic1,
         const int jc1, const int jcf, const int kc, Mat3D_I_DP &c,
         Mat_IO_DP &s);
void relax(Mat_IO_DP &u, Mat_I_DP &rhs);
void relax2(Mat_IO_DP &u, Mat_I_DP &rhs);
void resid(Mat_O_DP &res, Mat_I_DP &u, Mat_I_DP &rhs);
DP revcst(Vec_I_DP &x, Vec_I_DP &y, Vec_I_INT &iorder, Vec_IO_INT &n);
void reverse(Vec_IO_INT &iorder, Vec_I_INT &n);
DP rf(const DP x, const DP y, const DP z);
DP rj(const DP x, const DP y, const DP z, const DP p);
void rk4(Vec_I_DP &y, Vec_I_DP &dydx, const DP x, const DP h,
         Vec_O_DP &yout, void derivs(const DP, Vec_I_DP &, Vec_O_DP &));
void rkck(Vec_I_DP &y, Vec_I_DP &dydx, const DP x,
          const DP h, Vec_O_DP &yout, Vec_O_DP &yerr,
          void derivs(const DP, Vec_I_DP &, Vec_O_DP &));
void rkdumb(Vec_I_DP &vstart, const DP x1, const DP x2,
            void derivs(const DP, Vec_I_DP &, Vec_O_DP &));
void rkqs(Vec_IO_DP &y, Vec_IO_DP &dydx, DP &x, const DP htry,
          const DP eps, Vec_I_DP &yscal, DP &hdid, DP &hnext,
          void derivs(const DP, Vec_I_DP &, Vec_O_DP &));
void rlft3(Mat3D_IO_DP &data, Mat_IO_DP &speq, const int isign);
DP rofunc(const DP b);
void rotate(Mat_IO_DP &r, Mat_IO_DP &qt, const int i, const DP a,
            const DP b);
void rsolv(Mat_I_DP &a, Vec_I_DP &d, Vec_IO_DP &b);
void rstrct(Mat_O_DP &uc, Mat_I_DP &uf);
DP rtbis(DP func(const DP), const DP x1, const DP x2, const DP xacc);
DP rtflsp(DP func(const DP), const DP x1, const DP x2, const DP xacc);
DP rtnewt(void funcd(const DP, DP &, DP &), const DP x1, const DP x2,
          const DP xacc);
DP rtsafe(void funcd(const DP, DP &, DP &), const DP x1, const DP x2,
          const DP xacc);
DP rtsec(DP func(const DP), const DP x1, const DP x2, const DP xacc);
void rzextr(const int iest, const DP xest, Vec_I_DP &yest, Vec_O_DP &yz,
            Vec_O_DP &dy);
void savgol(Vec_O_DP &c, const int np, const int nl, const int nr,
            const int ld, const int m);
void scrsho(DP fx(const DP));
DP select(const int k, Vec_IO_DP &arr);
DP selip(const int k, Vec_I_DP &arr);
void shell(const int n, Vec_IO_DP &a);
void shoot(Vec_I_DP &v, Vec_O_DP &f);
void shootf(Vec_I_DP &v, Vec_O_DP &f);
void simp1(Mat_I_DP &a, const int mm, Vec_I_INT &ll, const int nll,
           const int iabf, int &kp, DP &bmax);
void simp2(Mat_I_DP &a, const int m, const int n, int &ip, const int kp);
void simp3(Mat_IO_DP &a, const int i1, const int k1, const int ip,
           const int kp);
void simplx(Mat_IO_DP &a, const int m1, const int m2, const int m3,
            int &icase, Vec_O_INT &izrov, Vec_O_INT &iposv);
void simpr(Vec_I_DP &y, Vec_I_DP &dydx, Vec_I_DP &dfdx, Mat_I_DP &dfdy,
           const DP xs, const DP htot, const int nstep, Vec_O_DP &yout,
           void derivs(const DP, Vec_I_DP &, Vec_O_DP &));
void sinft(Vec_IO_DP &y);
void slvsm2(Mat_O_DP &u, Mat_I_DP &rhs);
void slvsml(Mat_O_DP &u, Mat_I_DP &rhs);
void sncndn(const DP uu, const DP emmc, DP &sn, DP &cn, DP &dn);
DP snrm(Vec_I_DP &sx, const int itol);
void sobseq(const int n, Vec_O_DP &x);
void solvde(const int itmax, const DP conv, const DP slowc,
            Vec_I_DP &scalv, Vec_I_INT &indexv, const int nb, Mat_IO_DP &y);
void sor(Mat_I_DP &a, Mat_I_DP &b, Mat_I_DP &c, Mat_I_DP &d, Mat_I_DP &e,
         Mat_I_DP &f, Mat_IO_DP &u, const DP rjac);
void sort(Vec_IO_DP &arr);
void sort2(Vec_IO_DP &arr, Vec_IO_DP &brr);
void sort3(Vec_IO_DP &ra, Vec_IO_DP &rb, Vec_IO_DP &rc);
void spctrm(ifstream &fp, Vec_O_DP &p, const int k, const bool ovrlap);
void spear(Vec_I_DP &data1, Vec_I_DP &data2, DP &d, DP &zd, DP &probd,
           DP &rs, DP &probrs);
void sphbes(const int n, const DP x, DP &sj, DP &sy, DP &sjp, DP &syp);
void splie2(Vec_I_DP &x1a, Vec_I_DP &x2a, Mat_I_DP &ya, Mat_O_DP &y2a);
void splin2(Vec_I_DP &x1a, Vec_I_DP &x2a, Mat_I_DP &ya, Mat_I_DP &y2a,
            const DP x1, const DP x2, DP &y);
void spline(Vec_I_DP &x, Vec_I_DP &y, const DP yp1, const DP ypn,
            Vec_O_DP &y2);
void splint(Vec_I_DP &xa, Vec_I_DP &ya, Vec_I_DP &y2a, const DP x, DP &y);
void spread(const DP y, Vec_IO_DP &yy, const DP x, const int m);
void sprsax(Vec_I_DP &sa, Vec_I_INT &ija, Vec_I_DP &x, Vec_O_DP &b);
void sprsin(Mat_I_DP &a, const DP thresh, Vec_O_DP &sa, Vec_O_INT &ija);
void sprspm(Vec_I_DP &sa, Vec_I_INT &ija, Vec_I_DP &sb, Vec_I_INT &ijb,
            Vec_O_DP &sc, Vec_I_INT &ijc);
void sprstm(Vec_I_DP &sa, Vec_I_INT &ija, Vec_I_DP &sb, Vec_I_INT &ijb,
            const DP thresh, Vec_O_DP &sc, Vec_O_INT &ijc);
void sprstp(Vec_I_DP &sa, Vec_I_INT &ija, Vec_O_DP &sb, Vec_O_INT &ijb);
void sprstx(Vec_I_DP &sa, Vec_I_INT &ija, Vec_I_DP &x, Vec_O_DP &b);
void stifbs(Vec_IO_DP &y, Vec_IO_DP &dydx, DP &xx, const DP htry,
            const DP eps, Vec_I_DP &yscal, DP &hdid, DP &hnext,
            void derivs(const DP, Vec_I_DP &, Vec_O_DP &));
void stiff(Vec_IO_DP &y, Vec_IO_DP &dydx, DP &x, const DP htry,
           const DP eps, Vec_I_DP &yscal, DP &hdid, DP &hnext,
           void derivs(const DP, Vec_I_DP &, Vec_O_DP &));
void stoerm(Vec_I_DP &y, Vec_I_DP &d2y, const DP xs,
            const DP htot, const int nstep, Vec_O_DP &yout,
            void derivs(const DP, Vec_I_DP &, Vec_O_DP &));
void svbksb(Mat_I_DP &u, Vec_I_DP &w, Mat_I_DP &v, Vec_I_DP &b, Vec_O_DP &x);
void svdcmp(Mat_IO_DP &a, Vec_O_DP &w, Mat_O_DP &v);
void svdfit(Vec_I_DP &x, Vec_I_DP &y, Vec_I_DP &sig, Vec_O_DP &a,
            Mat_O_DP &u, Mat_O_DP &v, Vec_O_DP &w, DP &chisq,
            void funcs(const DP, Vec_O_DP &));
void svdvar(Mat_I_DP &v, Vec_I_DP &w, Mat_O_DP &cvm);
void toeplz(Vec_I_DP &r, Vec_O_DP &x, Vec_I_DP &y);
void tptest(Vec_I_DP &data1, Vec_I_DP &data2, DP &t, DP &prob);
void tqli(Vec_IO_DP &d, Vec_IO_DP &e, Mat_IO_DP &z);
DP trapzd(DP func(const DP), const DP a, const DP b, const int n);
void tred2(Mat_IO_DP &a, Vec_O_DP &d, Vec_O_DP &e);
void tridag(Vec_I_DP &a, Vec_I_DP &b, Vec_I_DP &c, Vec_I_DP &r, Vec_O_DP &u);
DP trncst(Vec_I_DP &x, Vec_I_DP &y, Vec_I_INT &iorder, Vec_IO_INT &n);
void trnspt(Vec_IO_INT &iorder, Vec_I_INT &n);
void ttest(Vec_I_DP &data1, Vec_I_DP &data2, DP &t, DP &prob);
void tutest(Vec_I_DP &data1, Vec_I_DP &data2, DP &t, DP &prob);
void twofft(Vec_I_DP &data1, Vec_I_DP &data2, Vec_O_DP &fft1,
            Vec_O_DP &fft2);
void vander(Vec_I_DP &x, Vec_O_DP &w, Vec_I_DP &q);
void vegas(Vec_I_DP &regn, DP fxn(Vec_I_DP &, const DP), const int init,
           const int ncall, const int itmx, const int nprn, DP &tgral, DP &sd,
           DP &chi2a);
void voltra(const DP t0, const DP h, Vec_O_DP &t, Mat_O_DP &f,
            DP g(const int, const DP),
            DP ak(const int, const int, const DP, const DP));
void wt1(Vec_IO_DP &a, const int isign,
         void wtstep(Vec_IO_DP &, const int, const int));
void wtn(Vec_IO_DP &a, Vec_I_INT &nn, const int isign,
         void wtstep(Vec_IO_DP &, const int, const int));
void wwghts(Vec_O_DP &wghts, const DP h,
            void kermom(Vec_O_DP &w, const DP y));
bool zbrac(DP func(const DP), DP &x1, DP &x2);
void zbrak(DP fx(const DP), const DP x1, const DP x2, const int n,
           Vec_O_DP &xb1, Vec_O_DP &xb2, int &nroot);
DP zbrent(DP func(const DP), const DP x1, const DP x2, const DP tol);
void zrhqr(Vec_I_DP &a, Vec_O_CPLX_DP &rt);
DP zriddr(DP func(const DP), const DP x1, const DP x2, const DP xacc);
void zroots(Vec_I_CPLX_DP &a, Vec_O_CPLX_DP &roots, const bool &polish);
}
#endif /* _NR_H_ */
