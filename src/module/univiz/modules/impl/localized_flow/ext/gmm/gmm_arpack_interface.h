/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef GMM_ARPACK_INTERFACE_H
#define GMM_ARPACK_INTERFACE_H

#include <ext/gmm/gmm_interface.h>

// --------------------------------------------------------------------------

namespace
{

struct arpack_debug
{
    int logfil, ndigit, mgetv0;
    int msaupd, msaup2, msaitr, mseigt, msapps, msgets, mseupd;
    int mnaupd, mnaup2, mnaitr, mneigt, mnapps, mngets, mneupd;
    int mcaupd, mcaup2, mcaitr, mceigt, mcapps, mcgets, mceupd;
};

extern "C" arpack_debug debug_;

template <typename T>
T *vptr(std::vector<T> &v)
{
    return &*v.begin();
}
}

// --------------------------------------------------------------------------

#define wrap_arpack(type, prefix)                                                                                                                                                                                                                                                       \
    extern "C" void prefix##saupd_(...);                                                                                                                                                                                                                                                \
    extern "C" void prefix##seupd_(...);                                                                                                                                                                                                                                                \
                                                                                                                                                                                                                                                                                        \
    inline void saupd(int &ido, const char *bmat, int n, const char *which, int nev, type tol, type *resid, int ncv, type *V, int ldv, void *param, void *ptr, type *workd, type *workl, int lworkl, int &info)                                                                         \
    {                                                                                                                                                                                                                                                                                   \
        prefix##saupd_(&ido, bmat, &n, which, &nev, &tol, resid, &ncv, V, &ldv, param, ptr, workd, workl, &lworkl, &info);                                                                                                                                                              \
    }                                                                                                                                                                                                                                                                                   \
                                                                                                                                                                                                                                                                                        \
    inline void seupd(int rvec, const char *howmny, int *select, type *D, type *Z, int ldz, type sigma, const char *bmat, int n, const char *which, int nev, type tol, type *resid, int ncv, type *V, int ldv, void *param, void *ptr, type *workd, type *workl, int lworkl, int &info) \
    {                                                                                                                                                                                                                                                                                   \
        prefix##seupd_(&rvec, howmny, select, D, Z, &ldz, &sigma, bmat, &n, which, &nev, &tol, resid, &ncv, V, &ldv, param, ptr, workd, workl, &lworkl, &info);                                                                                                                         \
    }

wrap_arpack(float, s)
    wrap_arpack(double, d)

#undef wrap_arpack

        // --------------------------------------------------------------------------

    namespace gmm
{

    template <typename T>
    struct arpack_workspace
    {
        typedef T value_type;

        size_type _n;
        size_type _nev;

        std::vector<value_type> workl;
        std::vector<value_type> workd;
        std::vector<value_type> resid;
        std::vector<value_type> workv;
        std::vector<value_type> eval;

        struct
        {
            int exact_shifts;
            int _unused1;
            int max_iter;
            int blocksize;
            int n_converged;
            int _unused2;
            int mode;
            int np;
            int num_op;
            int num_opb;
            int num_reo;
        } param;

        struct
        {
            int op_x;
            int op_y;
            int _unused[9];
        } ptr;

        void resize(int n, int nev, int nlv)
        {
            resid.resize(n);
            workd.resize(3 * n);
            workl.resize(nlv * (nlv + 9));
            workv.resize(n * nlv);

            eval.resize(nev);
        }

        void arpack_throw(const std::string &func, int info)
        {
            switch (info)
            {
            case 0:
            case 1:
                return;
            case -8:
                throw std::runtime_error("ARPACK (" + func + "):  LAPACK error");
            case -9:
                throw std::runtime_error("ARPACK (" + func + "):  starting resid. is zero");
            case -9999:
                throw std::runtime_error("ARPACK (" + func + "):  could not build Arnoldi basis");
            case 3:
                throw std::runtime_error("ARPACK (" + func + "):  no shifts could be applied (maybe increase nlv?)");
            default:
                std::cout << "ARPACK: info = " << info << '\n';
                throw std::runtime_error("ARPACK (" + func + "): unknown error");
            }
        }

    public:
        arpack_workspace()
            : _n(0)
            , _nev(0)
        {
        }

        void set_trace_level(unsigned int level)
        {
            debug_.msaupd = level > 1 ? 1 : 0;
            debug_.msaup2 = level > 2 ? 2 : 0;
            debug_.msaitr = level > 3 ? 1 : 0;
        }

        const array1D_reference<value_type *> eigenvalues()
        {
            return array1D_reference<value_type *>(&*eval.begin(), _nev);
        }

        const array1D_reference<value_type *> residuals()
        {
            return array1D_reference<value_type *>(&*resid.begin(), _nev);
        }

        const array2D_col_reference<value_type *> eigenvectors()
        {
            return array2D_col_reference<value_type *>(&*workv.begin(), _n, _nev);
        }

        const array1D_reference<value_type *> eigenvector(size_type i)
        {
            return array1D_reference<value_type *>(&*workv.begin() + i * _n, _n);
        }

        template <typename Mat>
        void iterate(const Mat &m, size_type nev, const char *which,
                     size_type max_iter = 0, double tol = 0.0, size_type nlv = 0)
        {
            // only standard problems for now
            const char *bmat = "I";
            const char *howmny = "A";

            size_type n = mat_nrows(m);

            // determine number of Lanczos vectors if not given (nev < nlv <= n )
            // heuristically, nlv = 5*nev seems a reasonable choice
            if (nlv <= 0)
                nlv = std::max(nev + 1, std::min(n, 5 * nev));

            // determine max. number of iterations if not given
            if (max_iter <= 0)
                max_iter = 10 * n;

            // resize workspace to problem size
            resize(n, nev, nlv);

            // set iteration parameters
            param.exact_shifts = true;
            param.max_iter = max_iter;
            param.blocksize = 1;
            param.mode = 1;

            int info = 0;
            int ido = 0;

            while (true)
            {
                saupd(ido, bmat, n, which, nev, tol, vptr(resid), nlv,
                      vptr(workv), n, &param, &ptr, vptr(workd), vptr(workl),
                      workl.size(), info);

                arpack_throw("saupd", info);

                if (ido == 99)
                {
                    // reverse communication: loop done
                    _n = n;
                    _nev = nev;

                    break;
                }
                else if (ido == -1 || ido == 1)
                {
                    // reverse communication: matrix vector multiplication requested
                    array1D_reference<value_type *> x(&*workd.begin() + ptr.op_x - 1, n);
                    array1D_reference<value_type *> y(&*workd.begin() + ptr.op_y - 1, n);

                    mult(m, x, y);
                }
                else
                    throw std::logic_error("unknown ARPACK reverse communication flag");
            }

            // post-processing of eigenvalues/eigenvectors
            int rvec = 1;

            std::vector<int> select(nlv);

            value_type sigma;
            std::vector<T> Z(n * nev);

            seupd(rvec, howmny, vptr(select), vptr(eval), vptr(Z), nev, sigma,
                  bmat, n, which, nev, tol, vptr(resid), nlv, vptr(workv), n,
                  &param, &ptr, vptr(workd), vptr(workl), workl.size(), info);

            arpack_throw("seupd", info);
        }
    };

} // namespace gmm

#endif // GMM_ARPACK_INTERFACE_H
