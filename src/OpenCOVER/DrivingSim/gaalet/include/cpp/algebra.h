/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_ALGEBRA_H
#define __GAALET_ALGEBRA_H

#include "multivector.h"
#include "utility.h"

namespace gaalet
{

//metric of multivector
// --- only definable by signature yet
// --- interface may change in order to allow for a general metric tensor
template <unsigned int P, unsigned Q>
struct signature
{
    static const unsigned int p = P;
    static const unsigned int q = Q;
    static const unsigned int dimension = P + Q;

    static const conf_t signature_bitmap = (Power<2, Q>::value - 1) << P;
};
template <typename ML, typename MR>
struct metric_combination_traits;

//combination of signature metrics -> negative signature dominant
template <unsigned int PL, unsigned QL, unsigned int PR, unsigned QR>
struct metric_combination_traits<signature<PL, QL>, signature<PR, QR> >
{
    //C++0x only: static_assert(PL==PR || (PL<PR && QL==0) || (PR<PL && QR==0),
    //              "Combination of different metrics: different number of positive signatures");
    static const unsigned int P = (QL == 0) ? PR : PL;
    static const unsigned int Q = (QL > QR) ? QL : QR;

    typedef signature<P, Q> metric;
    //typedef ::gaalet::metric<SL | SR> metric;
};

template <typename M, typename T = default_element_t>
struct algebra
{
    typedef M metric;

    typedef T element_t;

    //no cpp0x template aliases supported by gcc yet
    /*template<conf_t head, conf_t... tail>
   using mv = multivector<typename insert_element<head, typename mv<tail...>::clist>::clist>;

   template<>
   using mv = multivector<cl_null>;*/

    //multivector configuration elements unpacking
    /*C++0x only: template<conf_t... elements>
   struct mv;
   template<conf_t head, conf_t... tail>
   struct mv<head, tail...>
   {
      typedef multivector<typename insert_element<head, typename mv<tail...>::type::clist>::clist, metric> type;
   };
   template<conf_t head>
   struct mv<head>
   {
      typedef multivector<configuration_list<head, cl_null>, metric> type;
   };*/

    template <conf_t e1 = 0xffffffff, conf_t e2 = 0xffffffff, conf_t e3 = 0xffffffff, conf_t e4 = 0xffffffff, conf_t e5 = 0xffffffff, conf_t e6 = 0xffffffff, conf_t e7 = 0xffffffff, conf_t e8 = 0xffffffff, conf_t e9 = 0xffffffff, conf_t e10 = 0xffffffff, conf_t e11 = 0xffffffff, conf_t e12 = 0xffffffff>
    struct mv
    {
        typedef multivector<typename insert_element<e1,
                                                    typename insert_element<e2,
                                                                            typename insert_element<e3,
                                                                                                    typename insert_element<e4,
                                                                                                                            typename insert_element<e5,
                                                                                                                                                    typename insert_element<e6,
                                                                                                                                                                            typename insert_element<e7,
                                                                                                                                                                                                    typename insert_element<e8,
                                                                                                                                                                                                                            typename insert_element<e9,
                                                                                                                                                                                                                                                    typename insert_element<e10,
                                                                                                                                                                                                                                                                            typename insert_element<e11,
                                                                                                                                                                                                                                                                                                    typename insert_element<e12,
                                                                                                                                                                                                                                                                                                                            cl_null>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist,
                            metric, element_t> type;
    };
    template <conf_t e1, conf_t e2, conf_t e3, conf_t e4, conf_t e5, conf_t e6, conf_t e7, conf_t e8, conf_t e9, conf_t e10, conf_t e11>
    struct mv<e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, 0xffffffff>
    {
        typedef multivector<typename insert_element<e1,
                                                    typename insert_element<e2,
                                                                            typename insert_element<e3,
                                                                                                    typename insert_element<e4,
                                                                                                                            typename insert_element<e5,
                                                                                                                                                    typename insert_element<e6,
                                                                                                                                                                            typename insert_element<e7,
                                                                                                                                                                                                    typename insert_element<e8,
                                                                                                                                                                                                                            typename insert_element<e9,
                                                                                                                                                                                                                                                    typename insert_element<e10,
                                                                                                                                                                                                                                                                            typename insert_element<e11,
                                                                                                                                                                                                                                                                                                    cl_null>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist,
                            metric, element_t> type;
    };
    template <conf_t e1, conf_t e2, conf_t e3, conf_t e4, conf_t e5, conf_t e6, conf_t e7, conf_t e8, conf_t e9, conf_t e10>
    struct mv<e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, 0xffffffff, 0xffffffff>
    {
        typedef multivector<typename insert_element<e1,
                                                    typename insert_element<e2,
                                                                            typename insert_element<e3,
                                                                                                    typename insert_element<e4,
                                                                                                                            typename insert_element<e5,
                                                                                                                                                    typename insert_element<e6,
                                                                                                                                                                            typename insert_element<e7,
                                                                                                                                                                                                    typename insert_element<e8,
                                                                                                                                                                                                                            typename insert_element<e9,
                                                                                                                                                                                                                                                    typename insert_element<e10,
                                                                                                                                                                                                                                                                            cl_null>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist,
                            metric, element_t> type;
    };
    template <conf_t e1, conf_t e2, conf_t e3, conf_t e4, conf_t e5, conf_t e6, conf_t e7, conf_t e8, conf_t e9>
    struct mv<e1, e2, e3, e4, e5, e6, e7, e8, e9, 0xffffffff, 0xffffffff, 0xffffffff>
    {
        typedef multivector<typename insert_element<e1,
                                                    typename insert_element<e2,
                                                                            typename insert_element<e3,
                                                                                                    typename insert_element<e4,
                                                                                                                            typename insert_element<e5,
                                                                                                                                                    typename insert_element<e6,
                                                                                                                                                                            typename insert_element<e7,
                                                                                                                                                                                                    typename insert_element<e8,
                                                                                                                                                                                                                            typename insert_element<e9,
                                                                                                                                                                                                                                                    cl_null>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist,
                            metric, element_t> type;
    };
    template <conf_t e1, conf_t e2, conf_t e3, conf_t e4, conf_t e5, conf_t e6, conf_t e7, conf_t e8>
    struct mv<e1, e2, e3, e4, e5, e6, e7, e8, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff>
    {
        typedef multivector<typename insert_element<e1,
                                                    typename insert_element<e2,
                                                                            typename insert_element<e3,
                                                                                                    typename insert_element<e4,
                                                                                                                            typename insert_element<e5,
                                                                                                                                                    typename insert_element<e6,
                                                                                                                                                                            typename insert_element<e7,
                                                                                                                                                                                                    typename insert_element<e8,
                                                                                                                                                                                                                            cl_null>::clist>::clist>::clist>::clist>::clist>::clist>::clist>::clist,
                            metric, element_t> type;
    };
    template <conf_t e1, conf_t e2, conf_t e3, conf_t e4, conf_t e5, conf_t e6, conf_t e7>
    struct mv<e1, e2, e3, e4, e5, e6, e7, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff>
    {
        typedef multivector<typename insert_element<e1,
                                                    typename insert_element<e2,
                                                                            typename insert_element<e3,
                                                                                                    typename insert_element<e4,
                                                                                                                            typename insert_element<e5,
                                                                                                                                                    typename insert_element<e6,
                                                                                                                                                                            typename insert_element<e7,
                                                                                                                                                                                                    cl_null>::clist>::clist>::clist>::clist>::clist>::clist>::clist,
                            metric, element_t> type;
    };
    template <conf_t e1, conf_t e2, conf_t e3, conf_t e4, conf_t e5, conf_t e6>
    struct mv<e1, e2, e3, e4, e5, e6, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff>
    {
        typedef multivector<typename insert_element<e1,
                                                    typename insert_element<e2,
                                                                            typename insert_element<e3,
                                                                                                    typename insert_element<e4,
                                                                                                                            typename insert_element<e5,
                                                                                                                                                    typename insert_element<e6,
                                                                                                                                                                            cl_null>::clist>::clist>::clist>::clist>::clist>::clist,
                            metric, element_t> type;
    };
    template <conf_t e1, conf_t e2, conf_t e3, conf_t e4, conf_t e5>
    struct mv<e1, e2, e3, e4, e5, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff>
    {
        typedef multivector<typename insert_element<e1,
                                                    typename insert_element<e2,
                                                                            typename insert_element<e3,
                                                                                                    typename insert_element<e4,
                                                                                                                            typename insert_element<e5,
                                                                                                                                                    cl_null>::clist>::clist>::clist>::clist>::clist,
                            metric, element_t> type;
    };
    template <conf_t e1, conf_t e2, conf_t e3, conf_t e4>
    struct mv<e1, e2, e3, e4, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff>
    {
        typedef multivector<typename insert_element<e1,
                                                    typename insert_element<e2,
                                                                            typename insert_element<e3,
                                                                                                    typename insert_element<e4,
                                                                                                                            cl_null>::clist>::clist>::clist>::clist,
                            metric, element_t> type;
    };
    template <conf_t e1, conf_t e2, conf_t e3>
    struct mv<e1, e2, e3, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff>
    {
        typedef multivector<typename insert_element<e1,
                                                    typename insert_element<e2,
                                                                            typename insert_element<e3,
                                                                                                    cl_null>::clist>::clist>::clist,
                            metric, element_t> type;
    };
    template <conf_t e1, conf_t e2>
    struct mv<e1, e2, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff>
    {
        typedef multivector<typename insert_element<e1,
                                                    typename insert_element<e2,
                                                                            cl_null>::clist>::clist,
                            metric, element_t> type;
    };
    template <conf_t e1>
    struct mv<e1, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff>
    {
        typedef multivector<typename insert_element<e1,
                                                    cl_null>::clist,
                            metric, element_t> type;
    };
};

//default multivector: euclidean space (in signature bitmap, P part won't matter if Q part equals zero)
/*C++0x only:template<conf_t... elements>
struct mv
{
   typedef typename algebra<signature<0,0>>::mv<elements...>::type type;
};*/

//Algebra definitions
//typedef algebra<signature<4,1>> cm;
/*C++0x only:struct cm {
   typedef ::gaalet::algebra<signature<4,1>> algebra;

   template<conf_t... elements>
   struct mv
   {
      typedef typename algebra::mv<elements...>::type type;
   };
};*/
typedef algebra<signature<4, 1> > cm;

} //end namespace gaalet

#endif
