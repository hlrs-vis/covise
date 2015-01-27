/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __gmm_util_hpp
#define __gmm_util_hpp

#include <ext/gmm/gmm.h>
#include <netcdfcpp.h>
//#include <teem/nrrd.h>
#include <boost/bind.hpp>
#include <fstream>

namespace gmm
{
//
// --- save/load a CSR matrix in a NetCDF file ------------------------------
//

void NetCDF_save(const std::string &filename, const gmm::csr_matrix<double> &M)
{
    NcFile ncfile(filename.c_str(), NcFile::Replace);

    long nnz = gmm::nnz(M);
    long nr = M.nr + 1;
    long nc = M.nc;

    NcDim *nnz_dim = ncfile.add_dim("nnz", nnz);
    NcDim *nr_dim = ncfile.add_dim("nr", nr);
    NcDim *nc_dim;
    nc_dim = ncfile.add_dim("nc", nc);

    NcVar *pr_var = ncfile.add_var("v", ncDouble, nnz_dim);
    pr_var->put((const double *)M.pr, &nnz);

    NcVar *ir_var = ncfile.add_var("ind", ncInt, nnz_dim);
    ir_var->put((const int *)M.ir, &nnz);

    NcVar *jc_var = ncfile.add_var("ptr", ncInt, nr_dim);
    jc_var->put((const int *)M.jc, &nr);

    ncfile.close();
}

void NetCDF_load(const std::string &filename, gmm::csr_matrix<double> &M)
{
    NcFile ncfile(filename.c_str(), NcFile::ReadOnly);

    NcDim *nnz_dim = ncfile.get_dim("nnz");
    NcDim *nr_dim = ncfile.get_dim("nr");
    NcDim *nc_dim;
    nc_dim = ncfile.get_dim("nc");

    long nnz = nnz_dim->size();
    long nr = nr_dim->size();
    long nc = nc_dim->size();

    if (M.pr)
    {
        delete[] M.pr;
        delete[] M.ir;
        delete[] M.jc;
    }

    M.pr = 0;
    M.jc = new unsigned int[nr];
    M.ir = new unsigned int[nnz];
    M.pr = new double[nnz];

    NcVar *pr_var = ncfile.get_var("v");
    pr_var->get(M.pr, &nnz);

    NcVar *ir_var = ncfile.get_var("ind");
    ir_var->get((int *)M.ir, &nnz);

    NcVar *jc_var = ncfile.get_var("ptr");
    jc_var->get((int *)M.jc, &nr);

    M.nc = nc;
    M.nr = nr - 1;

    ncfile.close();
}

//
// --- save/load a CSC matrix in a NetCDF file ------------------------------
//

void NetCDF_save(const std::string &filename, const gmm::csc_matrix<double> &M)
{
    NcFile ncfile(filename.c_str(), NcFile::Replace);

    long nnz = gmm::nnz(M);
    long nr = M.nr + 1;
    long nc = M.nc;

    NcDim *nnz_dim = ncfile.add_dim("nnz", nnz);
    NcDim *nr_dim = ncfile.add_dim("nr", nr);
    NcDim *nc_dim;
    nc_dim = ncfile.add_dim("nc", nc);

    NcVar *pr_var = ncfile.add_var("v", ncDouble, nnz_dim);
    pr_var->put((const double *)M.pr, &nnz);

    NcVar *ir_var = ncfile.add_var("ind", ncInt, nnz_dim);
    ir_var->put((const int *)M.ir, &nnz);

    NcVar *jc_var = ncfile.add_var("ptr", ncInt, nr_dim);
    jc_var->put((const int *)M.jc, &nr);

    ncfile.close();
}

void NetCDF_load(const std::string &filename, gmm::csc_matrix<double> &M)
{
    NcFile ncfile(filename.c_str(), NcFile::ReadOnly);

    NcDim *nnz_dim = ncfile.get_dim("nnz");
    NcDim *nr_dim = ncfile.get_dim("nr");
    NcDim *nc_dim;
    nc_dim = ncfile.get_dim("nc");

    long nnz = nnz_dim->size();
    long nr = nr_dim->size();
    long nc = nc_dim->size();

    if (M.pr)
    {
        delete[] M.pr;
        delete[] M.ir;
        delete[] M.jc;
    }

    M.pr = 0;
    M.jc = new unsigned int[nr];
    M.ir = new unsigned int[nnz];
    M.pr = new double[nnz];

    NcVar *pr_var = ncfile.get_var("v");
    pr_var->get(M.pr, &nnz);

    NcVar *ir_var = ncfile.get_var("ind");
    ir_var->get((int *)M.ir, &nnz);

    NcVar *jc_var = ncfile.get_var("ptr");
    jc_var->get((int *)M.jc, &nr);

    M.nc = nc;
    M.nr = nr - 1;

    ncfile.close();
}

//
// --- save matrix/vector as ASCII file (row major) -------------------------
//

template <typename L>
void TEXT_save(const std::string &filename, const L &l)
{
    std::ofstream out(filename.c_str());
    TEXT_save(out, l, typename linalg_traits<L>::linalg_type());
}

template <typename V>
void TEXT_save(std::ostream &out, const V &v, abstract_vector)
{
    size_type i;

    for (i = 0; i < vect_size(v) - 1; ++i)
        out << v[i] << ' ';

    out << v[i] << '\n';
}

template <typename M>
void TEXT_save(std::ostream &out, const M &m, abstract_matrix)
{
    size_type r, c;

    for (r = 0; r < mat_nrows(m); ++r)
    {
        for (c = 0; c < mat_ncols(m) - 1; ++c)
            out << m(r, c) << ' ';
        out << m(r, c) << '\n';
    }
}

// --- load/save dense matrix/vector as NRRD (row major) --------------------

// int nrrd_type( double ) { return nrrdTypeDouble; }
// int nrrd_type( float )  { return nrrdTypeFloat;  }
// int nrrd_type( int )    { return nrrdTypeInt;    }

// void nrrd_check( int res )
// {
// 	if( res )
// 	{
// 		char *err = biffGetDone( NRRD );
// 		std::string tmp( err );
// 		free( err );

// 		throw std::runtime_error( tmp );
// 	}
// }

// template<typename T>
// Nrrd* NRRD_save( const std::string& filename, const std::vector<T>& v )
// {
// 	Nrrd* nrrd = nrrdNew();

// 	nrrd_check( nrrdWrap_va( nrrd, &v.front(), nrrd_type( T() ), 1, v.size() ) );
// 	nrrd_check( nrrdSave( filename.c_str(), nrrd, 0 ) );

// 	nrrdNix( nrrd );
// }

// template<typename T>
// Nrrd* NRRD_load( const std::string& filename, const std::vector<T>& v )
// {
// 	Nrrd* nrrd = nrrdNew();

// 	NrrdIoState* nio = nrrdIoStateNew();
// 	nrrdIoStateSet( nio, nrrdIoStateSkipData, AIR_TRUE );

// 	nrrd_check( nrrdLoad( nrrd, filename.c_str(), nio ) );
// 	nrrdIoStateNix( nio );

// 	if( nrrd->dim > 1 )
// 		throw std::logic_error( "could not read vector from" + filename + ": data dimension != 1" );

// 	if( nrrd->type != nrrd_type( T() ) )
// 		throw std::logic_error( "could not read matrix from" + filename + ": data type mismatch" );

// 	v.resize( nrrd->axis[0].size );
// 	nrrd->data = &v.front();

// 	nrrd_check( nrrdLoad( nrrd, filename.c_str(), NULL ) );
// 	nrrdNix( nrrd );
// }

// template<typename T>
// Nrrd* NRRD_save( const std::string& filename, const gmm::dense_matrix<T>& m )
// {
// 	Nrrd* nrrd = nrrdNew();

// 	nrrd_check( nrrdWrap_va( nrrd, &m.front(), nrrd_type( T() ), 1, mat_nrows(m), mat_ncols(m) ) );
// 	nrrd_check( nrrdSave( filename.c_str(), nrrd, 0 ) );

// 	nrrdNix( nrrd );
// }

// template<typename T>
// void NRRD_load( const std::string& filename, gmm::dense_matrix<T>& m )
// {
// 	Nrrd* nrrd = nrrdNew();

// 	NrrdIoState* nio = nrrdIoStateNew();
// 	nrrdIoStateSet( nio, nrrdIoStateSkipData, AIR_TRUE );

// 	nrrd_check( nrrdLoad( nrrd, filename.c_str(), nio ) );
// 	nrrdIoStateNix( nio );

// 	if( nrrd->dim > 2 )
// 		throw std::logic_error( "could not read matrix from" + filename + ": data dimension != 2" );

// 	if( nrrd->type != nrrd_type( T() ) )
// 		throw std::logic_error( "could not read matrix from" + filename + ": data type mismatch" );

// 	if( nrrd->dim == 1 )
// 		m.resize( 1, nrrd->axis[0].size );
// 	else
// 		m.resize( nrrd->axis[0].size, nrrd->axis[1].size );

// 	nrrd->data = &m.front();

//   	nrrd_check( nrrdLoad( nrrd, filename.c_str(), NULL ) );
// 	nrrdNix( nrrd );
// }

//
// --- compute matrix sparsity histogram ------------------------------------
//

template <class M, class L>
void mat_hist(M &m, const L &l)
{
    return mat_hist(m, l, typename linalg_traits<L>::linalg_type());
}

template <class M, class L>
void mat_hist(M &m, const L &l, abstract_matrix)
{
    typedef typename linalg_traits<L>::sub_orientation so_type;
    typedef typename principal_orientation_type<so_type>::potype po_type;

    gmm::clear(m);

    return mat_hist(m, l, po_type());
}

template <class M, class L>
void mat_hist(M &m, const L &l, row_major)
{
    typedef typename linalg_traits<L>::const_sub_row_type Row;
    typedef typename linalg_traits<Row>::const_iterator RowIter;

    for (size_type r = 0; r < mat_nrows(l); ++r)
    {
        Row row = mat_const_row(l, r);

        size_type ir = (size_type)rint((float)r / (float)mat_nrows(l) * (mat_nrows(m) - 1));

        for (RowIter ri = vect_const_begin(row); ri != vect_const_end(row); ++ri)
        {
            size_type ic = (size_type)rint((float)ri.index() / (float)mat_ncols(l) * (mat_ncols(m) - 1));
            m(ir, ic) += fabs(*ri);
        }
    }
}

template <class L>
void mat_hist(gmm::dense_matrix<unsigned int> &m, const L &l, col_major)
{
    std::cerr << "mat_hist: col_major\n";
    mat_hist(gmm::transposed(m), gmm::transposed(l));
}

//
// --- apply (analog to STL for_each and transform, in place) ---------------
//
template <class L, class Function>
Function apply(const L &l, Function f)
{
    return apply(l, f, typename linalg_traits<L>::linalg_type());
}

template <class L, class Function>
Function apply(const L &l, Function f, abstract_vector)
{
    typename linalg_traits<L>::iterator it = vect_const_begin(l), ite = vect_const_end(l);

    for (; it != ite; ++it)
        f(it.index(), *it);
}

template <class L, class Function>
Function apply(const L &l, Function f, abstract_matrix)
{
    return apply(l, f, typename principal_orientation_type<typename linalg_traits<L>::sub_orientation>::potype());
}

template <class L, class Function>
Function apply(const L &l, Function f, row_major)
{
    for (size_type i = 0; i < mat_nrows(l); ++i)
        apply_by_row(i, mat_const_row(l, i), f);

    return f;
}

template <class L, class Function>
Function apply(const L &l, Function f, col_major)
{
    for (size_type i = 0; i < mat_ncols(l); ++i)
        apply_by_col(i, mat_const_col(l, i), f);

    return f;
}

template <class L, class Function>
Function apply_by_row(size_type row, const L &l, Function f)
{
    typename linalg_traits<L>::const_iterator i;

    for (i = vect_const_begin(l); i != vect_const_end(l); ++i)
        f(row, i.index(), const_cast<typename linalg_traits<L>::value_type &>(*i));
}

template <class L, class Function>
Function apply_by_col(size_type col, const L &l, Function f)
{
    typename linalg_traits<L>::const_iterator i;

    for (i = vect_const_begin(l); i != vect_const_end(l); ++i)
        f(i.index(), col, const_cast<typename linalg_traits<L>::value_type &>(*i));
}

} // namespace gmm

#endif // __gmm_util_hpp
