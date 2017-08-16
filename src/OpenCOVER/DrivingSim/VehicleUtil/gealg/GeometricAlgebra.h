/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

///----------------------------------
///Author: Florian Seybold, 2009
///www.hlrs.de
///----------------------------------
///
///TODO:
/// - Multivector configuration: bitmapset -> Replace intrinsic type bitmapset (uint64/uint128) by list of bitmaps (uint8), use variadic templates
/// - cpp0x support: static asserts -> identify feasable checks, implement
/// - Inverse operation: uncertain definition -> check implementation,
///                      only applicable to versors yet -> check input multivectors/implement iterative inversion algorithm
/// - Multivector construction: no initialization of multiple elements -> add reasonable initialization constructor
/// - Multiplication multivector with intrinsic type double: scalar multivector not initialized with double -> initialization with double
/// - Type of scalar multiplicator of basis-nth-vector: fixed to double -> add multivector template argument for type

#ifndef __GeometricAlgebra_h
#define __GeometricAlgebra_h

/*#ifdef __GXX_EXPERIMENTAL_CXX0X__
#define __CPP0X__
#endif
#ifdef __GXX_EXPERIMENTAL_CPP0X__ 
#define __CPP0X__
#define __GXX_EXPERIMENTAL_CXX0X__ 
#endif*/

#include <cstring>
#include <cmath>
#if defined(_MSC_VER)
//#include "stdint.h"
#else
#include <stdint.h>
#endif
#include <limits>
#include <iostream>
#include <sstream>
#include <iomanip>
#if !defined(__CUDACC__) && !defined(__INTEL_COMPILER)
#include <tuple>
using std::tuple;
using std::tuple_element;
using std::tuple_size;
using std::get;
#elif defined(__CUDACC__)
#define GEALG_CUDA __host__ __device__
#include <thrust/tuple.h>
using thrust::tuple;
using thrust::tuple_element;
using thrust::tuple_size;
using thrust::get;
#elif defined(SX)
#define FUSION_MAX_TUPLE_SIZE 50
#include <boost/fusion/tuple.hpp>
using boost::fusion::tuple;
using boost::fusion::tuple_element;
using boost::fusion::tuple_size;
using boost::fusion::get;
#else
#define FUSION_MAX_TUPLE_SIZE 50
#include <boost/spirit/fusion/sequence/tuple.hpp>
#include <boost/spirit/fusion/sequence/tuple_size.hpp>
#include <boost/spirit/fusion/sequence/tuple_element.hpp>
#include <boost/spirit/fusion/sequence/get.hpp>
using boost::fusion::tuple;
using boost::fusion::tuple_element;
using boost::fusion::tuple_size;
using boost::fusion::get;
#endif

#include "CanonicalReorderingSign.h"
#include "BitCount.h"
#include "Power.h"
#include "Factorial.h"

namespace gealg
{

#if (__GNUC__ >= 4 && __GNUC_MINOR__ >= 3 && defined(__x86_64__) && !defined(__CUDACC__)) && !defined(__INTEL_COMPILER)
///128-bit integer type, works with gcc 64-bit
typedef unsigned int uint128_t __attribute__((mode(TI))); //128-bit
///Type of multivector configuration bitmap
typedef uint128_t mv_confmap_t;
///Type of multivector configuration bitmap
//typedef uint64_t mv_confmap_t;
#else
typedef uint64_t mv_confmap_t;
#endif
//Intel?
//typedef __m128i mv_confmap_t;

///Searches element index in multivector with respect to its configuration
template <class MT, uint8_t EB, uint8_t I = MT::num>
struct MultivectorElementBitmapSearch
{
    static const uint8_t bitmap = (uint8_t)(MT::bitmap >> (8 * (MT::num - I)));

    static const uint8_t index = (bitmap == EB) ? (MT::num - I) : MultivectorElementBitmapSearch<MT, EB, I - 1>::index;
};
template <class MT, uint8_t EB>
struct MultivectorElementBitmapSearch<MT, EB, 0x0>
{
    static const uint8_t index = MT::num;
};

///Compressed multivector
/** \param N number of elements
 * \param B sorted bitmap representing multivector configuration: one element is described by 8-bits.
 *          Examples: e0 -> 0x00, e1 -> 0x01 (b001), e1^e3 -> 0x05 (b101), e0+e1^e3 -> 0x0500 (b101<<8 | b00)
 * \param S bitmap indicating signature (only non-degenerating) of applied algebra. Set bit indicates negative signature of basis vector.
 *          Example: G(3, 1) -> (+,+,+,-) -> 0x08 (b1000)
 */
template <uint8_t N = 8, mv_confmap_t B = 0x0706050403020100LL, uint8_t S = 0x0>
class Multivector
{
public:
    typedef Multivector<N, B, S> MultivectorType;

    ///Constructor
    Multivector()
    {
        memset(e, 0, N * sizeof(double));
    }

    ///Constructor, first element initialization
    Multivector(const double &e0)
    {
        memset(e, 0, N * sizeof(double));
        e[0] = e0;
    }

    ///Element by index
    double &operator[](const uint8_t &i)
    {
        return e[i];
    }
    ///Element by index
    const double &operator[](const uint8_t &i) const
    {
        return e[i];
    }

    ///Element by configuration
    template <uint8_t EB, class T>
    double &element(const T &)
    {
        static const uint8_t I = MultivectorElementBitmapSearch<MultivectorType, EB>::index;
        return e[I];
    }
    ///Element by configuration
    template <uint8_t EB, class T>
    double element(const T &) const
    {
        static const uint8_t I = MultivectorElementBitmapSearch<MultivectorType, EB>::index;
        return (I == MultivectorType::num) ? 0.0 : e[I];
    }

    ///Evaluate arguments
    template <class T>
    void evaluate(const T &)
    {
    }

    static const uint8_t num = N;
    static const mv_confmap_t bitmap = B;
    static const uint8_t signature = S;

protected:
    double e[N];
};

template <mv_confmap_t B, uint8_t S>
class Multivector<0, B, S>
{
public:
    typedef Multivector<0, B, S> MultivectorType;

    static const uint8_t num = 0;
    static const mv_confmap_t bitmap = B;
    static const uint8_t signature = S;
};

///Evaluation by element
template <class ET, class T = tuple<>, uint8_t I = ET::MultivectorType::num>
struct EvaluateExpression
{
    typedef typename ET::MultivectorType MultivectorType;
    static const uint8_t EB = (uint8_t)(MultivectorType::bitmap >> (8 * (MultivectorType::num - I)));

    static inline void operate(MultivectorType &mv, const ET &e, const T &vars = T())
    {
        evaluateElement(mv, e, vars);
    }
    static inline void evaluateElement(MultivectorType &mv, const ET &e, const T &vars)
    {
        mv[MultivectorType::num - I] = e.template element<EB>(vars);
        EvaluateExpression<ET, T, I - 1>::operate(mv, e, vars);
    }
};
template <class ET, class T>
struct EvaluateExpression<ET, T, 0>
{
    typedef typename ET::MultivectorType MultivectorType;
    static inline void operate(MultivectorType &, const ET &, const T & = T())
    {
    }
    static inline void evaluateElement(MultivectorType &, const ET &, const T &)
    {
    }
};

///Abstract expression, necessary for automatic argument type deduction of helper template functions
/**   \param E type of nested expression/multivector
 *
 *    Generally member functions pass arguments through to nested member.
 */
template <class E>
struct Expression
{
    typedef typename E::MultivectorType MultivectorType;
    typedef Expression<MultivectorType> result_type;

    ///Construction
    Expression()
        : e_(E())
    {
    }

    Expression(const double &e0)
        : e_(e0)
    {
    }

#ifdef __CPP0X__
    ///Evaluation of expression on construction (this class is a multivector)
    template <class ET>
    Expression(Expression<ET> &&e)
    {
        e.e_.evaluate(tuple<>());

        EvaluateExpression<Expression<ET> >::operate(e_, std::forward<Expression<ET> >(e));
    }
#else
    ///Evaluation of expression on construction (this class is a multivector)
    /*template<class ET>
   Expression(Expression<ET>& e) {
      e.e_.evaluate(*((tuple<>*)NULL));

      EvaluateExpression<Expression<ET> >::operate(e_, e);
   }*/
    template <class ET>
    Expression(Expression<ET> e)
    {
        e.e_.evaluate(tuple<>());

        EvaluateExpression<Expression<ET> >::operate(e_, e);
    }
#endif

/*#ifdef __CPP0X__
   ///Construction with move semantics
   Expression(Expression&& e)
      :  e_ (std::move(e.e_))
   { }

   ///Construction with perfect forwarding argument of nested element type
   Expression(E&& e) 
      :  e_ (std::forward<E>(e))
   { }
#else
   ///Construction with copy semantics
	Expression(const E& e) 
      :  e_ (e)
   { }
#endif*/

#ifdef __CPP0X__
    ///Construction
    Expression(Expression &&e)
        : e_(std::move(e.e_))
    {
    }

    ///Construction
    Expression(E &&e)
        : e_(std::move(e))
    {
    }
#endif
    ///Construction
    Expression(const Expression &e)
        : e_(e.e_)
    {
    }

    ///Construction
    Expression(const E &e)
        : e_(e)
    {
    }

#ifdef __CPP0X__
    ///Evaluation with respect to variable tuple (this class is an expression)
    template <class T>
    result_type &&operator()(const T &vars)
    {
        e_.evaluate(vars);

        result_type r;
        EvaluateExpression<Expression<E>, T>::operate(r.e_, e_, vars);
        return std::move(r);
    }

    ///Evaluation (this class is an expression)
    result_type &&operator()()
    {
        e_.evaluate(tuple<>());

        result_type r;
        EvaluateExpression<Expression<E>, tuple<> >::operate(r.e_, e_, tuple<>());
        return std::move(r);
    }
#else
    ///Evaluation with respect to variable tuple (this class is an expression)
    template <class T>
    result_type operator()(const T &vars)
    {
        e_.evaluate(vars);

        result_type r;
        EvaluateExpression<Expression<E>, T>::operate(r.e_, e_, vars);
        return r;
    }

    ///Evaluation (this class is an expression)
    result_type operator()()
    {
        e_.evaluate(tuple<>());

        result_type r;
        EvaluateExpression<Expression<E>, tuple<> >::operate(r.e_, e_, tuple<>());
        return r;
    }
#endif

#ifdef __CPP0X__
    ///Evaluation of expression (this class is a multivector)
    template <class ET>
    void operator=(Expression<ET> &&e)
    {
        e.e_.evaluate(tuple<>());

        EvaluateExpression<Expression<ET> >::operate(e_, e);
    }
#else
    ///Evaluation of expression (this class is a multivector)
    /*template<class ET>
   void operator=(Expression<ET>& e) {
      e.e_.evaluate(*((tuple<>*)NULL));

      EvaluateExpression<Expression<ET> >::operate(e_, e);
   }*/
    template <class ET>
    void operator=(Expression<ET> e)
    {
        e.e_.evaluate(tuple<>());

        EvaluateExpression<Expression<ET> >::operate(e_, e);
    }
#endif

    /*///Access of expression variable (this class is a multivector)
   template<class T>
   E& operator()(T& vars) {
      return E(vars);
   }*/

    ///Element by index
    double &operator[](const uint8_t &i)
    {
        return e_[i];
    }

    ///Element by index
    const double &operator[](const uint8_t &i) const
    {
        return e_[i];
    }

    ///Element by configuration
    template <uint8_t EB>
    double element()
    {
        return e_.template element<EB>(tuple<>());
    }

    ///Element by configuration
    template <uint8_t EB, class T>
    double element(const T &vars) const
    {
        return e_.template element<EB>(vars);
    }

    ///Evaluate arguments
    template <class T>
    void evaluate(const T &vars)
    {
        e_.evaluate(vars);
    }

    ///Nested expression/multivector element
    E e_;
};

template <int I, class T>
class ExpressionVariable
{
public:
    typedef typename tuple_element<I, T>::type ExpressionType;
    typedef typename ExpressionType::MultivectorType MultivectorType;
    typedef Expression<MultivectorType> result_type;

    ExpressionType &operator()(T &vars)
    {
        return get<I>(vars);
    }

    ///Access elements, evaluate on the fly
    template <uint8_t EB>
    double element(const T &vars) const
    {
        return get<I>(vars).template element<EB>(vars);
    }

    ///Evaluate arguments
    void evaluate(const T & /*vars*/)
    {
        //get<I>(vars).evaluate(vars);
    }

    static const uint8_t num = MultivectorType::num;
    static const mv_confmap_t bitmap = MultivectorType::bitmap;
};

//Helper typedefs
template <uint8_t N = 8, mv_confmap_t B = 0x0706050403020100LL, uint8_t S = 0x0>
struct mv
{
    typedef Expression<Multivector<N, B, S> > type;
};

template <int I, class T>
struct var
{
    typedef Expression<ExpressionVariable<I, T> > type;
};

//Unary Expressions
template <class M, class OP, bool EXPR_EAGER = false>
struct UnaryExpression
{
    typedef typename OP::MTR MultivectorType;

#ifdef __CPP0X__
    UnaryExpression(UnaryExpression &&ue)
        : m_(std::move(ue.m_))
    {
    }

    UnaryExpression(M &&m)
        : m_(std::forward<M>(m))
    {
    }
#else
    UnaryExpression(const M &m)
        : m_(m)
    {
    }
#endif

    ///Access elements
    template <uint8_t EB, class T>
    double element(const T &vars) const
    {
        return OP::template evaluateElement<EB>(m_, vars);
    }

    ///Evaluate arguments
    template <class T>
    void evaluate(const T &vars)
    {
        m_.evaluate(vars);
    }

    M m_;
};

//Unary Expressions, eager evaluation of expression
template <class M, class OP>
struct UnaryExpression<M, OP, true>
{
    typedef typename OP::MTR MultivectorType;

#ifdef __CPP0X__
    UnaryExpression(UnaryExpression &&ue)
        : m_(std::move(ue.m_))
    {
    }

    UnaryExpression(M &&m)
        : m_(std::forward<M>(m))
    {
    }
#else
    UnaryExpression(const M &m)
        : m_(m)
    {
    }
#endif

    ///Access elements
    template <uint8_t EB, class T>
    double element(const T &vars) const
    {
        return mve_.element<EB>(vars);
    }

    ///Evaluate arguments
    template <class T>
    void evaluate(const T &vars)
    {
        m_.evaluate(vars);

        mve_ = OP::evaluate(m_, vars);
    }

    M m_;

    MultivectorType mve_;
};

template <class E>
struct Reverse
{
    typedef typename E::MultivectorType MT;
    typedef MT MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const E &e, const T &tuple)
    {
        return Power<-1, BitCount<EB>::value *(BitCount<EB>::value - 1) / 2>::value * e.template element<EB>(tuple);
    }

    template <uint8_t I, uint8_t dummy> //dummy: C++ doesn't allow for full specialization in non-namespace scope, but for partial!
    struct ReverseElement
    {
        static inline void operate(MTR &result, const MT &m)
        {
            static const uint8_t bitmapP = (uint8_t)(MT::bitmap >> (8 * (MT::num - I)));
            result[MT::num - I] = Power<-1, BitCount<bitmapP>::value *(BitCount<bitmapP>::value - 1) / 2>::value * m[MT::num - I];
            ReverseElement<I - 1, 0>::operate(result, m);
        }
    };

    template <uint8_t dummy>
    struct ReverseElement<0, dummy>
    {
        static inline void operate(MTR &, const MT &)
        {
        }
    };
};

//Grade part of Multivector
template <class MV, uint8_t G, class MVR = Multivector<0, 0> >
struct GradeResultMultivectorConfig
{
    static const mv_confmap_t bitmapP = (uint8_t)MV::bitmap;

    typedef GradeResultMultivectorConfig < Multivector < MV::num - 1, (MV::bitmap >> 8), ((MV::num - 1) == 0) ? 0 : MV::signature >, G,
        Multivector < BitCount<bitmapP>::value == G ? MVR::num + 1 : MVR::num,
        BitCount<bitmapP>::value == G ? (MVR::bitmap | (bitmapP << ((uint8_t)MVR::num * 8))) : MVR::bitmap,
        MV::signature
        >> NextConfig;

    typedef typename NextConfig::Type Type;
};

template <uint8_t G, class MVR>
struct GradeResultMultivectorConfig<Multivector<0, 0>, G, MVR>
{
    typedef MVR Type;
};

template <class E, uint8_t G>
struct Grade
{
    typedef typename E::MultivectorType MT;
    typedef typename GradeResultMultivectorConfig<MT, G>::Type MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const E &e, const T &tuple)
    {
        return e.template element<EB>(tuple);
    }

    template <uint8_t I = MTR::num, uint8_t dummy = 0>
    struct GradeElement
    {
        static inline void operate(MTR &result, const MT &mv)
        {
            static const uint8_t bitmapP = (uint8_t)(MTR::bitmap >> 8 * (MTR::num - I));
            static const uint8_t IMT = MultivectorElementBitmapSearch<MT, bitmapP>::index;
            result[MTR::num - I] = mv[IMT];

            GradeElement<I - 1>::operate(result, mv);
        }
    };
    template <uint8_t dummy>
    struct GradeElement<0, dummy>
    {
        static inline void operate(MTR &, const MT &)
        {
        }
    };
};

template <class E, class MTResult>
struct Part
{
    typedef typename E::MultivectorType MT;
    typedef MTResult MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const E &e, const T &tuple)
    {
        static const uint8_t IMT = MultivectorElementBitmapSearch<MT, EB>::index;
        return (IMT == MT::num) ? 0.0 : e.template element<EB>(tuple);
    }

    template <uint8_t I = MTR::num, uint8_t dummy = 0>
    struct PartElement
    {
        static inline void operate(MTR &result, const MT &mv)
        {
            static const uint8_t bitmapP = (uint8_t)(MTR::bitmap >> 8 * (MTR::num - I));
            static const uint8_t IMT = MultivectorElementBitmapSearch<MT, bitmapP>::index;
            result[MTR::num - I] = (IMT == MT::num) ? 0.0 : mv[IMT];

            PartElement<I - 1>::operate(result, mv);
        }
    };
    template <uint8_t dummy>
    struct PartElement<0, dummy>
    {
        static inline void operate(MTR &, const MT &)
        {
        }
    };
};

template <class E, uint8_t BP>
struct Element
{
    typedef typename E::MultivectorType MT;
    typedef Multivector<1, 0x00, MT::signature> MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const E &e, const T &tuple)
    {
        return (EB != 0) ? std::numeric_limits<double>::signaling_NaN() : e.template element<BP>(tuple);
    }
};

//Binary Expressions
template <class L, class H, class OP, bool EAGER = false>
struct BinaryExpression
{
    typedef typename OP::MTR MultivectorType;

#ifdef __CPP0X__
    BinaryExpression(BinaryExpression &&be)
        : l_(std::move(be.l_))
        , h_(std::move(be.h_))
    {
    }

    BinaryExpression(L &&l, H &&h)
        : l_(std::forward<L>(l))
        , h_(std::forward<H>(h))
    {
    }
#else
    BinaryExpression(const L &l, const H &h)
        : l_(l)
        , h_(h)
    {
    }
#endif

    template <uint8_t EB, class T>
    double element(const T &vars) const
    {
        return OP::template evaluateElement<EB>(l_, h_, vars);
    }

    ///Evaluate arguments
    template <class T>
    void evaluate(const T &vars)
    {
        l_.evaluate(vars);
        h_.evaluate(vars);
    }

    L l_;
    H h_;
};

///Specialization: Eager evaluation of children expressions
template <class L, class H, class OP>
struct BinaryExpression<L, H, OP, true>
{
    typedef typename OP::MTR MultivectorType;

#ifdef __CPP0X__
    BinaryExpression(BinaryExpression &&be)
        : l_(std::move(be.l_))
        , h_(std::move(be.h_))
    {
    }

    BinaryExpression(L &&l, H &&h)
        : l_(std::forward<L>(l))
        , h_(std::forward<H>(h))
    {
    }
#else
    BinaryExpression(const L &l, const H &h)
        : l_(l)
        , h_(h)
    {
    }
#endif

    ///Access element
    template <uint8_t EB, class T>
    double element(const T &vars) const
    {
        return OP::template evaluateElement<EB>(mvl_, mvh_, vars);
    }

    ///Evaluate arguments
    template <class T>
    void evaluate(const T &vars)
    {
        l_.evaluate(vars);
        EvaluateExpression<L, T>::operate(mvl_, l_, vars);

        h_.evaluate(vars);
        EvaluateExpression<H, T>::operate(mvh_, h_, vars);
    }

    L l_;
    H h_;

    typename L::MultivectorType mvl_;
    typename H::MultivectorType mvh_;
};

// Adding, Subtracting
template <class MVL, class MVH, class MVR = Multivector<0, 0, MVL::signature | MVH::signature> >
struct AddResultMultivectorConfig
{
    static const uint8_t BLP = (uint8_t)MVL::bitmap;
    static const uint8_t BHP = (uint8_t)MVH::bitmap;
    static const uint8_t numLP = (uint8_t)(BLP <= BHP);
    static const uint8_t numHP = (uint8_t)(BLP >= BHP);
    static const mv_confmap_t bitmapP = (numLP > numHP) ? BLP : BHP;

    typedef AddResultMultivectorConfig<Multivector<(uint8_t)MVL::num - numLP, (MVL::bitmap >> (8 * numLP))>,
                                       Multivector<(uint8_t)MVH::num - numHP, (MVH::bitmap >> (8 * numHP))>,
                                       Multivector<MVR::num + 1, MVR::bitmap | (bitmapP << ((uint8_t)MVR::num * 8)), MVR::signature> > NextConfig;

    typedef typename NextConfig::Type Type;
};

template <class MVH, class MVR>
struct AddResultMultivectorConfig<Multivector<0, 0>, MVH, MVR>
{
    static const mv_confmap_t bitmapP = (uint8_t)MVH::bitmap;

    typedef AddResultMultivectorConfig<Multivector<0, 0>,
                                       Multivector<MVH::num - 1, (MVH::bitmap >> 8)>,
                                       Multivector<MVR::num + 1, MVR::bitmap | (bitmapP << ((uint8_t)MVR::num * 8)), MVR::signature> > NextConfig;
    typedef typename NextConfig::Type Type;
};

template <class MVL, class MVR>
struct AddResultMultivectorConfig<MVL, Multivector<0, 0>, MVR>
{
    static const mv_confmap_t bitmapP = (uint8_t)MVL::bitmap;

    typedef AddResultMultivectorConfig<Multivector<MVL::num - 1, (MVL::bitmap >> 8)>,
                                       Multivector<0, 0>,
                                       Multivector<MVR::num + 1, MVR::bitmap | (bitmapP << ((uint8_t)MVR::num * 8)), MVR::signature> > NextConfig;
    typedef typename NextConfig::Type Type;
};

template <class MVR>
struct AddResultMultivectorConfig<Multivector<0, 0>, Multivector<0, 0>, MVR>
{
    typedef MVR Type;
};

//template<class MTL, class MTH>
template <class EL, class EH>
struct Add
{
    typedef typename EL::MultivectorType MTL;
    typedef typename EH::MultivectorType MTH;
    typedef typename AddResultMultivectorConfig<MTL, MTH>::Type MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const EL &el, const EH &eh, const T &tuple)
    {
        static const uint8_t IL = MultivectorElementBitmapSearch<MTL, EB>::index;
        static const uint8_t IH = MultivectorElementBitmapSearch<MTH, EB>::index;

        return ((IL == MTL::num) ? 0.0 : el.template element<EB>(tuple)) + ((IH == MTH::num) ? 0.0 : eh.template element<EB>(tuple));
    }

    template <uint8_t IR, uint8_t IL, uint8_t IH> //Inverted indices: IR, IL, IH
    struct AddElement
    {
        static const uint8_t IRB = ((MTR::bitmap >> ((MTR::num - IR) * 8)) & 0xff);
        static const uint8_t ILB = ((MTL::bitmap >> ((MTL::num - IL) * 8)) & 0xff);
        static const uint8_t IHB = ((MTH::bitmap >> ((MTH::num - IH) * 8)) & 0xff);
        typedef AddElement<IR - 1,
                           ((IL != 0) ? ((IRB == ILB) ? (IL - 1) : IL) : 0),
                           ((IH != 0) ? ((IRB == IHB) ? (IH - 1) : IH) : 0)> NextAdd;

        static inline void operate(MTR &result, const MTL &l, const MTH &h)
        {
            result[MTR::num - IR] = ((IRB == ILB) ? (l[MTL::num - IL]) : 0.0) + ((IRB == IHB) ? (h[MTH::num - IH]) : 0.0);
            NextAdd::operate(result, l, h);
        }
    };

    template <uint8_t IL, uint8_t IH>
    struct AddElement<0, IL, IH>
    {
        static inline void operate(MTR &, const MTL &, const MTH &)
        {
        }
    };
};

//template<class MTL, class MTH>
template <class EL, class EH>
struct Subtract
{
    typedef typename EL::MultivectorType MTL;
    typedef typename EH::MultivectorType MTH;
    typedef typename AddResultMultivectorConfig<MTL, MTH>::Type MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const EL &el, const EH &eh, const T &tuple)
    {
        static const uint8_t IL = MultivectorElementBitmapSearch<MTL, EB>::index;
        static const uint8_t IH = MultivectorElementBitmapSearch<MTH, EB>::index;

        return ((IL == MTL::num) ? 0.0 : el.template element<EB>(tuple)) - ((IH == MTH::num) ? 0.0 : eh.template element<EB>(tuple));
    }

    template <uint8_t IR, uint8_t IL, uint8_t IH> //Inverted indices: IR, IL, IH
    struct SubtractElement
    {
        static const uint8_t IRB = ((MTR::bitmap >> ((MTR::num - IR) * 8)) & 0xff);
        static const uint8_t ILB = ((MTL::bitmap >> ((MTL::num - IL) * 8)) & 0xff);
        static const uint8_t IHB = ((MTH::bitmap >> ((MTH::num - IH) * 8)) & 0xff);
        typedef SubtractElement<IR - 1,
                                ((IL != 0) ? ((IRB == ILB) ? (IL - 1) : IL) : 0),
                                ((IH != 0) ? ((IRB == IHB) ? (IH - 1) : IH) : 0)> NextSubtract;

        static inline void operate(MTR &result, const MTL &l, const MTH &h)
        {
            result[MTR::num - IR] = ((IRB == ILB) ? (l[MTL::num - IL]) : 0.0) - ((IRB == IHB) ? (h[MTH::num - IH]) : 0.0);
            NextSubtract::operate(result, l, h);
        }
    };

    template <uint8_t IL, uint8_t IH>
    struct SubtractElement<0, IL, IH>
    {
        static inline void operate(MTR &, const MTL &, const MTH &)
        {
        }
    };
};

// Multiplying
template <class MVI, class MVR>
struct InsertSortMultivector
{
    static const uint8_t bitmapP = (uint8_t)MVI::bitmap;

    template <uint8_t IB = MVR::num, uint8_t dummy = 0x0> //IB: Inverted bitmap index, dummy because explicit specialization here not allowed
    struct InBitmap
    {
        static const uint8_t index = (bitmapP <= ((uint8_t)(MVR::bitmap >> ((MVR::num - IB) * 8)))) ? (MVR::num - IB) : InBitmap<IB - 1>::index;
    };
    template <uint8_t dummy>
    struct InBitmap<0, dummy>
    {
        static const uint8_t index = MVR::num;
    };

    template <uint8_t num, mv_confmap_t bitmask = 0x0>
    struct BitMaskFill
    {
        static const mv_confmap_t mask = BitMaskFill<num - 1, bitmask | ((mv_confmap_t)0xff << ((num - 1) * 8))>::mask;
    };
    template <mv_confmap_t bitmask>
    struct BitMaskFill<0, bitmask>
    {
        static const mv_confmap_t mask = bitmask;
    };

    static const uint8_t indexP = InBitmap<>::index;
    static const bool bitmapExistent = (((uint8_t)(MVR::bitmap >> (indexP * 8))) == bitmapP && MVR::num != 0) ? true : false;
    static const mv_confmap_t newBitmap = bitmapExistent
                                              ? MVR::bitmap
                                              : (((MVR::bitmap & (~BitMaskFill<indexP>::mask)) << 8) | ((mv_confmap_t)bitmapP << (8 * indexP)) | (MVR::bitmap & BitMaskFill<indexP>::mask));
    static const uint8_t newNum = bitmapExistent ? MVR::num : (MVR::num + 1);

    typedef typename InsertSortMultivector<Multivector<MVI::num - 1, (MVI::bitmap >> 8)>,
                                           Multivector<newNum, newBitmap, MVR::signature> >::Type Type;
};
template <class MVR>
struct InsertSortMultivector<Multivector<0, 0>, MVR>
{
    typedef MVR Type;
};

template <class MVL, class MVH, class MVR = Multivector<0, 0, MVL::signature | MVH::signature> >
struct MultiplyResultMultivectorConfig
{
    static const uint8_t bitmapP = (uint8_t)MVL::bitmap;

    template <uint8_t IB = MVH::num, uint8_t dummy = 0x0> //Inversed Index
    struct ElementResultConfig
    {
        static const mv_confmap_t bitmap = ElementResultConfig<IB - 1>::bitmap ^ ((mv_confmap_t)bitmapP << (8 * (MVH::num - IB)));
    };
    template <uint8_t dummy>
    struct ElementResultConfig<0, dummy>
    {
        static const mv_confmap_t bitmap = MVH::bitmap;
    };

    typedef MultiplyResultMultivectorConfig<Multivector<MVL::num - 1, (MVL::bitmap >> 8)>,
                                            MVH,
                                            typename InsertSortMultivector<Multivector<MVH::num, ElementResultConfig<>::bitmap>, MVR>::Type> NextConfig;
    typedef typename NextConfig::Type Type;
};
template <class MVH, class MVR>
struct MultiplyResultMultivectorConfig<Multivector<0, 0>, MVH, MVR>
{
    typedef MVR Type;
};

template <class EL, class EH>
struct Multiply
{
    typedef typename EL::MultivectorType MTL;
    typedef typename EH::MultivectorType MTH;
    typedef typename MultiplyResultMultivectorConfig<MTL, MTH>::Type MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const EL &el, const EH &eh, const T &tuple)
    {
        double result = 0.0;

        ResultBitmap<EB>::operate(result, el, eh, tuple);

        return result;
    }

    template <uint8_t IBR = MTR::num, uint8_t dummyR = 0x0> //inversed index
    struct ResultIndex
    {
        static const uint8_t BR = (uint8_t)(MTR::bitmap >> (8 * (MTR::num - IBR)));

        template <uint8_t IBL = MTL::num, uint8_t dummyL = 0x0> //inversed index
        struct lowElementTimesHighMultivector
        {
            static const uint8_t BL = (uint8_t)(MTL::bitmap >> (8 * (MTL::num - IBL)));

            template <uint8_t IBH = MTH::num, uint8_t dummyH = 0x0>
            struct lowElementTimesHighElement
            {
                static const uint8_t BH = (uint8_t)(MTH::bitmap >> (8 * (MTH::num - IBH)));

                /*template<int EQ, int dummyAdd=0>
            struct Add {
               static inline void operate(MTR& result, const MTL& l, const MTH& h) {
               result[MTR::num-IBR] += (l[MTL::num-IBL] * h[MTH::num-IBH] * CanonicalReorderingSign<BL,BH>::value * MetricTensorSign<BL, BH, MTR::signature>::value);
               }
            };
            template<int dummyAdd>
            struct Add<0, dummyAdd> {
               static inline void operate(MTR&, const MTL&, const MTH&) { }
            };*/

                static inline void operate(MTR &result, const MTL &l, const MTH &h)
                {
                    result[MTR::num - IBR] += (BR == (BL ^ BH)) ? (l[MTL::num - IBL] * h[MTH::num - IBH] * CanonicalReorderingSign<BL, BH>::value * MetricTensorSign<BL, BH, MTR::signature>::value) : 0.0;
                    //Add<BR-(BL^BH)>::operate(result, l, h);
                    lowElementTimesHighElement<IBH - 1>::operate(result, l, h);
                }
            };
            template <uint8_t dummyH>
            struct lowElementTimesHighElement<0, dummyH>
            {
                static inline void operate(MTR &, const MTL &, const MTH &)
                {
                }
            };

            static inline void operate(MTR &result, const MTL &l, const MTH &h)
            {
                lowElementTimesHighElement<>::operate(result, l, h);
                lowElementTimesHighMultivector<IBL - 1>::operate(result, l, h);
            }
        };
        template <uint8_t dummyL>
        struct lowElementTimesHighMultivector<0, dummyL>
        {
            static inline void operate(MTR &, const MTL &, const MTH &)
            {
            }
        };

        static inline void operate(MTR &result, const MTL &l, const MTH &h)
        {
            lowElementTimesHighMultivector<>::operate(result, l, h);
            ResultIndex<IBR - 1>::operate(result, l, h);
        }
    };
    template <uint8_t dummyR> //inversed index
    struct ResultIndex<0, dummyR>
    {
        static inline void operate(MTR &, const MTL &, const MTH &)
        {
        }
    };

    template <uint8_t EB> //inversed index
    struct ResultBitmap
    {
        static const uint8_t BR = EB;

        template <uint8_t IBL = MTL::num, uint8_t dummyL = 0x0> //inversed index
        struct lowElementTimesHighMultivector
        {
            static const uint8_t BL = (uint8_t)(MTL::bitmap >> (8 * (MTL::num - IBL)));

            template <uint8_t IBH = MTH::num, uint8_t dummyH = 0x0>
            struct lowElementTimesHighElement
            {
                static const uint8_t BH = (uint8_t)(MTH::bitmap >> (8 * (MTH::num - IBH)));

                template <class T>
                static inline void operate(double &result, const EL &el, const EH &eh, const T &vars)
                {
                    Evaluate<(uint8_t)(BR - (BL ^ BH))>::operate(result, el, eh, vars);
                    lowElementTimesHighElement<IBH - 1>::operate(result, el, eh, vars);
                }

                template <uint8_t R, int dummy = 0>
                struct Evaluate
                {
                    template <class T>
                    static inline void operate(double &, const EL &, const EH &, const T &)
                    {
                    }
                };
                template <int dummy>
                struct Evaluate<0, dummy>
                {
                    template <class T>
                    static inline void operate(double &result, const EL &el, const EH &eh, const T &vars)
                    {
                        result += el.template element<BL>(vars) * eh.template element<BH>(vars) * CanonicalReorderingSign<BL, BH>::value * MetricTensorSign<BL, BH, MTR::signature>::value;
                    }
                };
            };
            template <uint8_t dummyH>
            struct lowElementTimesHighElement<0, dummyH>
            {
                static inline void operate(double &, const EL &, const EH &)
                {
                }
                template <class T>
                static inline void operate(double &, const EL &, const EH &, const T &)
                {
                }
            };

            static inline void operate(double &result, const EL &el, const EH &eh)
            {
                lowElementTimesHighElement<>::operate(result, el, eh);
                lowElementTimesHighMultivector<IBL - 1>::operate(result, el, eh);
            }
            template <class T>
            static inline void operate(double &result, const EL &el, const EH &eh, const T &tuple)
            {
                lowElementTimesHighElement<>::operate(result, el, eh, tuple);
                lowElementTimesHighMultivector<IBL - 1>::operate(result, el, eh, tuple);
            }
        };
        template <uint8_t dummyL>
        struct lowElementTimesHighMultivector<0, dummyL>
        {
            static inline void operate(double &, const EL &, const EH &)
            {
            }
            template <class T>
            static inline void operate(double &, const EL &, const EH &, const T &)
            {
            }
        };

        static inline void operate(double &result, const EL &el, const EH &eh)
        {
            lowElementTimesHighMultivector<>::operate(result, el, eh);
        }
        template <class T>
        static inline void operate(double &result, const EL &el, const EH &eh, const T &tuple)
        {
            lowElementTimesHighMultivector<>::operate(result, el, eh, tuple);
        }
    };
};

//InnerProduct
template <class MVL, class MVH, class MVR = Multivector<0, 0, MVL::signature | MVH::signature> >
struct InnerProductResultMultivectorConfig
{
    static const uint8_t BPL = (uint8_t)MVL::bitmap;

    template <class MVPH = MVH, class MVPR = Multivector<0, 0> > //Inversed Index
    struct ElementResultConfig
    {
        static const uint8_t BPH = (uint8_t)(MVPH::bitmap);

        static const bool InPro = (BitCount<BPL ^ BPH>::value == ((((int)BitCount<BPL>::value - (int)BitCount<BPH>::value) < 0)
                                                                      ? (BitCount<BPH>::value - BitCount<BPL>::value)
                                                                      : (BitCount<BPL>::value - BitCount<BPH>::value))) && BPL != 0x0 && BPH != 0x0;

        typedef ElementResultConfig < Multivector<MVPH::num - 1, (MVPH::bitmap >> 8)>,
            Multivector < InPro ? MVPR::num + 1 : MVPR::num,
            InPro ? (((mv_confmap_t)(BPL ^ BPH) << (8 * MVPR::num)) ^ MVPR::bitmap) : MVPR::bitmap >> NextConfig;
        typedef typename NextConfig::Type Type;
    };
    template <class MVPR>
    struct ElementResultConfig<Multivector<0, 0>, MVPR>
    {
        typedef MVPR Type;
    };

    typedef InnerProductResultMultivectorConfig<Multivector<MVL::num - 1, (MVL::bitmap >> 8)>,
                                                MVH,
                                                typename InsertSortMultivector<typename ElementResultConfig<>::Type, MVR>::Type> NextConfig;
    typedef typename NextConfig::Type Type;
};
template <class MVH, class MVR>
struct InnerProductResultMultivectorConfig<Multivector<0, 0>, MVH, MVR>
{
    typedef MVR Type;
};

template <class EL, class EH>
struct InnerProduct
{
    typedef typename EL::MultivectorType MTL;
    typedef typename EH::MultivectorType MTH;
    typedef typename InnerProductResultMultivectorConfig<MTL, MTH>::Type MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const EL &el, const EH &eh, const T &tuple)
    {
        double result = 0.0;

        ResultBitmap<EB>::operate(result, el, eh, tuple);

        return result;
    }

    template <uint8_t IBR = MTR::num, uint8_t dummyR = 0x0> //inversed index
    struct ResultIndex
    {
        static const uint8_t BR = (uint8_t)(MTR::bitmap >> (8 * (MTR::num - IBR)));

        template <uint8_t IBL = MTL::num, uint8_t dummyL = 0x0> //inversed index
        struct lowElementTimesHighMultivector
        {
            static const uint8_t BL = (uint8_t)(MTL::bitmap >> (8 * (MTL::num - IBL)));

            template <uint8_t IBH = MTH::num, uint8_t dummyH = 0x0>
            struct lowElementTimesHighElement
            {
                static const uint8_t BH = (uint8_t)(MTH::bitmap >> (8 * (MTH::num - IBH)));
                static const bool InPro = (BitCount<BL ^ BH>::value == ((((int)BitCount<BL>::value - (int)BitCount<BH>::value) < 0)
                                                                            ? (BitCount<BH>::value - BitCount<BL>::value)
                                                                            : (BitCount<BL>::value - BitCount<BH>::value))) && BL != 0x0 && BH != 0x0;

                static inline void operate(MTR &result, const MTL &l, const MTH &h)
                {
                    result[MTR::num - IBR] += (BR == (BL ^ BH) && InPro) ? (l[MTL::num - IBL] * h[MTH::num - IBH] * CanonicalReorderingSign<BL, BH>::value * MetricTensorSign<BL, BH, MTR::signature>::value) : 0.0;
                    lowElementTimesHighElement<IBH - 1>::operate(result, l, h);
                }
            };
            template <uint8_t dummyH>
            struct lowElementTimesHighElement<0, dummyH>
            {
                static inline void operate(MTR &, const MTL &, const MTH &)
                {
                }
            };

            static inline void operate(MTR &result, const MTL &l, const MTH &h)
            {
                lowElementTimesHighElement<>::operate(result, l, h);
                lowElementTimesHighMultivector<IBL - 1>::operate(result, l, h);
            }
        };
        template <uint8_t dummyL>
        struct lowElementTimesHighMultivector<0, dummyL>
        {
            static inline void operate(MTR &, const MTL &, const MTH &)
            {
            }
        };

        static inline void operate(MTR &result, const MTL &l, const MTH &h)
        {
            lowElementTimesHighMultivector<>::operate(result, l, h);
            ResultIndex<IBR - 1>::operate(result, l, h);
        }
    };
    template <uint8_t dummyR> //inversed index
    struct ResultIndex<0, dummyR>
    {
        static inline void operate(MTR &, const MTL &, const MTH &)
        {
        }
    };

    template <uint8_t EB> //inversed index
    struct ResultBitmap
    {
        static const uint8_t BR = EB;

        template <uint8_t IBL = MTL::num, uint8_t dummyL = 0x0> //inversed index
        struct lowElementTimesHighMultivector
        {
            static const uint8_t BL = (uint8_t)(MTL::bitmap >> (8 * (MTL::num - IBL)));

            template <uint8_t IBH = MTH::num, uint8_t dummyH = 0x0>
            struct lowElementTimesHighElement
            {
                static const uint8_t BH = (uint8_t)(MTH::bitmap >> (8 * (MTH::num - IBH)));
                static const bool InPro = (BitCount<BL ^ BH>::value == ((((int)BitCount<BL>::value - (int)BitCount<BH>::value) < 0)
                                                                            ? (BitCount<BH>::value - BitCount<BL>::value)
                                                                            : (BitCount<BL>::value - BitCount<BH>::value))) && BL != 0x0 && BH != 0x0;

                static inline void operate(double &result, const EL &el, const EH &eh)
                {
                    result += (BR == (BL ^ BH) && InPro) ? (el.template element<BL>() * eh.template element<BH>() * CanonicalReorderingSign<BL, BH>::value * MetricTensorSign<BL, BH, MTR::signature>::value) : 0.0;
                    lowElementTimesHighElement<IBH - 1>::operate(result, el, eh);
                }
                template <class T>
                static inline void operate(double &result, const EL &el, const EH &eh, const T &tuple)
                {
                    result += (BR == (BL ^ BH) && InPro) ? (el.template element<BL>(tuple) * eh.template element<BH>(tuple) * CanonicalReorderingSign<BL, BH>::value * MetricTensorSign<BL, BH, MTR::signature>::value) : 0.0;
                    lowElementTimesHighElement<IBH - 1>::operate(result, el, eh, tuple);
                }
            };
            template <uint8_t dummyH>
            struct lowElementTimesHighElement<0, dummyH>
            {
                static inline void operate(double &, const EL &, const EH &)
                {
                }
                template <class T>
                static inline void operate(double &, const EL &, const EH &, const T &)
                {
                }
            };

            static inline void operate(double &result, const EL &el, const EH &eh)
            {
                lowElementTimesHighElement<>::operate(result, el, eh);
                lowElementTimesHighMultivector<IBL - 1>::operate(result, el, eh);
            }
            template <class T>
            static inline void operate(double &result, const EL &el, const EH &eh, const T &tuple)
            {
                lowElementTimesHighElement<>::operate(result, el, eh, tuple);
                lowElementTimesHighMultivector<IBL - 1>::operate(result, el, eh, tuple);
            }
        };
        template <uint8_t dummyL>
        struct lowElementTimesHighMultivector<0, dummyL>
        {
            static inline void operate(double &, const EL &, const EH &)
            {
            }
            template <class T>
            static inline void operate(double &, const EL &, const EH &, const T &)
            {
            }
        };

        static inline void operate(double &result, const EL &el, const EH &eh)
        {
            lowElementTimesHighMultivector<>::operate(result, el, eh);
        }
        template <class T>
        static inline void operate(double &result, const EL &el, const EH &eh, const T &tuple)
        {
            lowElementTimesHighMultivector<>::operate(result, el, eh, tuple);
        }
    };
};

//OuterProduct
template <class MVL, class MVH, class MVR = Multivector<0, 0, MVL::signature | MVH::signature> >
struct OuterProductResultMultivectorConfig
{
    static const uint8_t BPL = (uint8_t)MVL::bitmap;

    template <class MVPH = MVH, class MVPR = Multivector<0, 0> > //Inversed Index
    struct ElementResultConfig
    {
        static const uint8_t BPH = (uint8_t)(MVPH::bitmap);

        static const bool OutPro = (BitCount<BPL ^ BPH>::value == (BitCount<BPL>::value + BitCount<BPH>::value));

        typedef ElementResultConfig < Multivector<MVPH::num - 1, (MVPH::bitmap >> 8)>,
            Multivector < OutPro ? MVPR::num + 1 : MVPR::num,
            OutPro ? (((mv_confmap_t)(BPL ^ BPH) << (8 * MVPR::num)) ^ MVPR::bitmap) : MVPR::bitmap >> NextConfig;
        typedef typename NextConfig::Type Type;
    };
    template <class MVPR>
    struct ElementResultConfig<Multivector<0, 0>, MVPR>
    {
        typedef MVPR Type;
    };

    typedef OuterProductResultMultivectorConfig<Multivector<MVL::num - 1, (MVL::bitmap >> 8)>,
                                                MVH,
                                                typename InsertSortMultivector<typename ElementResultConfig<>::Type, MVR>::Type> NextConfig;
    typedef typename NextConfig::Type Type;
};
template <class MVH, class MVR>
struct OuterProductResultMultivectorConfig<Multivector<0, 0>, MVH, MVR>
{
    typedef MVR Type;
};

template <class EL, class EH>
struct OuterProduct
{
    typedef typename EL::MultivectorType MTL;
    typedef typename EH::MultivectorType MTH;
    typedef typename OuterProductResultMultivectorConfig<MTL, MTH>::Type MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const EL &el, const EH &eh, const T &tuple)
    {
        double result = 0.0;

        ResultBitmap<EB>::operate(result, el, eh, tuple);

        return result;
    }

    template <uint8_t IBR = MTR::num, uint8_t dummyR = 0x0> //inversed index
    struct ResultIndex
    {
        static const uint8_t BR = (uint8_t)(MTR::bitmap >> (8 * (MTR::num - IBR)));

        template <uint8_t IBL = MTL::num, uint8_t dummyL = 0x0> //inversed index
        struct lowElementTimesHighMultivector
        {
            static const uint8_t BL = (uint8_t)(MTL::bitmap >> (8 * (MTL::num - IBL)));

            template <uint8_t IBH = MTH::num, uint8_t dummyH = 0x0>
            struct lowElementTimesHighElement
            {
                static const uint8_t BH = (uint8_t)(MTH::bitmap >> (8 * (MTH::num - IBH)));
                static const bool OutPro = (BitCount<BL ^ BH>::value == (BitCount<BL>::value + BitCount<BH>::value));

                static inline void operate(MTR &result, const MTL &l, const MTH &h)
                {
                    result[MTR::num - IBR] += (BR == (BL ^ BH) && OutPro) ? (l[MTL::num - IBL] * h[MTH::num - IBH] * CanonicalReorderingSign<BL, BH>::value * MetricTensorSign<BL, BH, MTR::signature>::value) : 0.0;
                    lowElementTimesHighElement<IBH - 1>::operate(result, l, h);
                }
            };
            template <uint8_t dummyH>
            struct lowElementTimesHighElement<0, dummyH>
            {
                static inline void operate(MTR &, const MTL &, const MTH &)
                {
                }
            };

            static inline void operate(MTR &result, const MTL &l, const MTH &h)
            {
                lowElementTimesHighElement<>::operate(result, l, h);
                lowElementTimesHighMultivector<IBL - 1>::operate(result, l, h);
            }
        };
        template <uint8_t dummyL>
        struct lowElementTimesHighMultivector<0, dummyL>
        {
            static inline void operate(MTR &, const MTL &, const MTH &)
            {
            }
        };

        static inline void operate(MTR &result, const MTL &l, const MTH &h)
        {
            lowElementTimesHighMultivector<>::operate(result, l, h);
            ResultIndex<IBR - 1>::operate(result, l, h);
        }
    };
    template <uint8_t dummyR> //inversed index
    struct ResultIndex<0, dummyR>
    {
        static inline void operate(MTR &, const MTL &, const MTH &)
        {
        }
    };

    template <uint8_t EB> //inversed index
    struct ResultBitmap
    {
        static const uint8_t BR = EB;

        template <uint8_t IBL = MTL::num, uint8_t dummyL = 0x0> //inversed index
        struct lowElementTimesHighMultivector
        {
            static const uint8_t BL = (uint8_t)(MTL::bitmap >> (8 * (MTL::num - IBL)));

            template <uint8_t IBH = MTH::num, uint8_t dummyH = 0x0>
            struct lowElementTimesHighElement
            {
                static const uint8_t BH = (uint8_t)(MTH::bitmap >> (8 * (MTH::num - IBH)));
                static const bool OutPro = (BitCount<BL ^ BH>::value == (BitCount<BL>::value + BitCount<BH>::value));

                static inline void operate(double &result, const EL &el, const EH &eh)
                {
                    result += (BR == (BL ^ BH) && OutPro) ? (el.template element<BL>() * eh.template element<BH>() * CanonicalReorderingSign<BL, BH>::value * MetricTensorSign<BL, BH, MTR::signature>::value) : 0.0;
                    lowElementTimesHighElement<IBH - 1>::operate(result, el, eh);
                }
                template <class T>
                static inline void operate(double &result, const EL &el, const EH &eh, const T &tuple)
                {
                    result += (BR == (BL ^ BH) && OutPro) ? (el.template element<BL>(tuple) * eh.template element<BH>(tuple) * CanonicalReorderingSign<BL, BH>::value * MetricTensorSign<BL, BH, MTR::signature>::value) : 0.0;
                    lowElementTimesHighElement<IBH - 1>::operate(result, el, eh, tuple);
                }
            };
            template <uint8_t dummyH>
            struct lowElementTimesHighElement<0, dummyH>
            {
                static inline void operate(double &, const EL &, const EH &)
                {
                }
                template <class T>
                static inline void operate(double &, const EL &, const EH &, const T &)
                {
                }
            };

            static inline void operate(double &result, const EL &el, const EH &eh)
            {
                lowElementTimesHighElement<>::operate(result, el, eh);
                lowElementTimesHighMultivector<IBL - 1>::operate(result, el, eh);
            }
            template <class T>
            static inline void operate(double &result, const EL &el, const EH &eh, const T &tuple)
            {
                lowElementTimesHighElement<>::operate(result, el, eh, tuple);
                lowElementTimesHighMultivector<IBL - 1>::operate(result, el, eh, tuple);
            }
        };
        template <uint8_t dummyL>
        struct lowElementTimesHighMultivector<0, dummyL>
        {
            static inline void operate(double &, const EL &, const EH &)
            {
            }
            template <class T>
            static inline void operate(double &, const EL &, const EH &, const T &)
            {
            }
        };

        static inline void operate(double &result, const EL &el, const EH &eh)
        {
            lowElementTimesHighMultivector<>::operate(result, el, eh);
        }
        template <class T>
        static inline void operate(double &result, const EL &el, const EH &eh, const T &tuple)
        {
            lowElementTimesHighMultivector<>::operate(result, el, eh, tuple);
        }
    };
};

//Tertiary Expressions
template <class L, class M, class H, class OP>
struct TertiaryExpression
{
    typedef typename OP::MTR MultivectorType;

#ifdef __CPP0X__
    TertiaryExpression(TertiaryExpression &&te)
        : l_(std::move(te.l_))
        , m_(std::move(te.m_))
        , h_(std::move(te.h_))
    {
    }

    TertiaryExpression(L &&l, M &&m, H &&h)
        : l_(std::forward<L>(l))
        , m_(std::forward<M>(m))
        , h_(std::forward<H>(h))
    {
    }
#else
    TertiaryExpression(const L &l, const M &m, const H &h)
        : l_(l)
        , m_(m)
        , h_(h)
    {
    }
#endif

    MultivectorType operator()() const
    {
        return OP::evaluate(l_(), m_(), h_());
    }
    template <class T>
    MultivectorType operator()(const T &tuple) const
    {
        return OP::evaluate(l_(tuple), m_(tuple), h_(tuple));
    }

    template <uint8_t EB>
    double element() const
    {
        return OP::template evaluateElement<EB>(l_, m_, h_);
    }
    template <uint8_t EB, class T>
    double element(const T &tuple) const
    {
        return OP::template evaluateElement<EB>(l_, m_, h_, tuple);
    }

    const L l_;
    const M m_;
    const H h_;
};

} //End of namespace gealg

//creator functions for unary expressions
//reverse
#ifdef __CPP0X__
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Reverse<gealg::Expression<M> > > >
operator!(gealg::Expression<M> && m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Reverse<gealg::Expression<M> > > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Reverse<gealg::Expression<M> > > >
operator!(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Reverse<gealg::Expression<M> > > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//grade G part of multivector
#ifdef __CPP0X__
template <uint8_t G, class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Grade<gealg::Expression<M>, G> > >
grade(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Grade<gealg::Expression<M>, G> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <uint8_t G, class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Grade<gealg::Expression<M>, G> > >
grade(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Grade<gealg::Expression<M>, G> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//multivector part defined by Multivector<N, B>
#ifdef __CPP0X__
template <uint8_t N, gealg::mv_confmap_t B, class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Part<gealg::Expression<M>, gealg::Multivector<N, B, M::MultivectorType::signature> > > >
part(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Part<gealg::Expression<M>, gealg::Multivector<N, B, M::MultivectorType::signature> > > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <uint8_t N, gealg::mv_confmap_t B, class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Part<gealg::Expression<M>, gealg::Multivector<N, B, M::MultivectorType::signature> > > >
part(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Part<gealg::Expression<M>, gealg::Multivector<N, B, M::MultivectorType::signature> > > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//element I part of multivector
#ifdef __CPP0X__
template <uint8_t I, class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Element<gealg::Expression<M>, I> > >
element(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Element<gealg::Expression<M>, I> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <uint8_t I, class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Element<gealg::Expression<M>, I> > >
element(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Element<gealg::Expression<M>, I> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//creator functions for binary expressions
//plus
#ifdef __CPP0X__
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Add<gealg::Expression<L>, gealg::Expression<H> > > >
operator+(gealg::Expression<L> &&l, gealg::Expression<H> &&h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Add<gealg::Expression<L>, gealg::Expression<H> > > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(std::forward<gealg::Expression<L> >(l), std::forward<gealg::Expression<H> >(h)));
}
#else
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Add<gealg::Expression<L>, gealg::Expression<H> > > >
operator+(const gealg::Expression<L> &l, const gealg::Expression<H> &h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Add<gealg::Expression<L>, gealg::Expression<H> > > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(l, h));
}
#endif

//minus
#ifdef __CPP0X__
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Subtract<gealg::Expression<L>, gealg::Expression<H> > > >
operator-(gealg::Expression<L> &&l, gealg::Expression<H> &&h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Subtract<gealg::Expression<L>, gealg::Expression<H> > > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(std::forward<gealg::Expression<L> >(l), std::forward<gealg::Expression<H> >(h)));
}
#else
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Subtract<gealg::Expression<L>, gealg::Expression<H> > > >
operator-(const gealg::Expression<L> &l, const gealg::Expression<H> &h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Subtract<gealg::Expression<L>, gealg::Expression<H> > > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(l, h));
}
#endif

#ifdef __CPP0X__
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Multiply<typename L::MultivectorType, typename H::MultivectorType>, true> >
//gealg::Expression< gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Multiply<gealg::Expression<L>, gealg::Expression<H> > > >
operator*(gealg::Expression<L> &&l, gealg::Expression<H> &&h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Multiply<typename L::MultivectorType, typename H::MultivectorType>, true> BinaryExpressionType;
    //typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Multiply<gealg::Expression<L>, gealg::Expression<H> > > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(std::forward<gealg::Expression<L> >(l), std::forward<gealg::Expression<H> >(h)));
}
#else
//geometric product of multivectors
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Multiply<typename L::MultivectorType, typename H::MultivectorType>, true> >
//gealg::Expression< gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Multiply<gealg::Expression<L>, gealg::Expression<H> > > >
operator*(const gealg::Expression<L> &l, const gealg::Expression<H> &h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Multiply<typename L::MultivectorType, typename H::MultivectorType>, true> BinaryExpressionType;
    //typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::Multiply<gealg::Expression<L>, gealg::Expression<H> > > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(l, h));
}
#endif

//geometric product with intrinsic type double
#ifdef __CPP0X__
template <class L>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<gealg::Multivector<1, 0x00> >, gealg::Multiply<gealg::Expression<L>, gealg::Expression<gealg::Multivector<1, 0x00> > > > >
operator*(gealg::Expression<L> &&l, const double &dh)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<gealg::Multivector<1, 0x00> >, gealg::Multiply<gealg::Expression<L>, gealg::Expression<gealg::Multivector<1, 0x00> > > > BinaryExpressionType;
    gealg::Expression<gealg::Multivector<1, 0x00> > h;
    h[0] = dh;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(std::forward<gealg::Expression<L> >(l), std::move(h)));
}
#else
template <class L>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<gealg::Multivector<1, 0x00> >, gealg::Multiply<gealg::Expression<L>, gealg::Expression<gealg::Multivector<1, 0x00> > > > >
operator*(const gealg::Expression<L> &l, const double &dh)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<gealg::Multivector<1, 0x00> >, gealg::Multiply<gealg::Expression<L>, gealg::Expression<gealg::Multivector<1, 0x00> > > > BinaryExpressionType;
    gealg::Expression<gealg::Multivector<1, 0x00> > h;
    h[0] = dh;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(l, h));
}

template <class L>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<gealg::Multivector<1, 0x00> >, gealg::Expression<L>, gealg::Multiply<gealg::Expression<gealg::Multivector<1, 0x00> >, gealg::Expression<L> > > >
operator*(const double &dh, const gealg::Expression<L> &l)
{
    typedef gealg::BinaryExpression<gealg::Expression<gealg::Multivector<1, 0x00> >, gealg::Expression<L>, gealg::Multiply<gealg::Expression<gealg::Multivector<1, 0x00> >, gealg::Expression<L> > > BinaryExpressionType;
    gealg::Expression<gealg::Multivector<1, 0x00> > h;
    h[0] = dh;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(h, l));
}
#endif

//inner product of multivectors
#ifdef __CPP0X__
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::InnerProduct<gealg::Expression<L>, gealg::Expression<H> > > >
operator%(gealg::Expression<L> &&l, gealg::Expression<H> &&h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::InnerProduct<gealg::Expression<L>, gealg::Expression<H> > > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(std::forward<gealg::Expression<L> >(l), std::forward<gealg::Expression<H> >(h)));
}
#else
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::InnerProduct<gealg::Expression<L>, gealg::Expression<H> > > >
operator%(const gealg::Expression<L> &l, const gealg::Expression<H> &h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::InnerProduct<gealg::Expression<L>, gealg::Expression<H> > > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(l, h));
}
#endif

//outer product of multivectors
#ifdef __CPP0X__
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::OuterProduct<gealg::Expression<L>, gealg::Expression<H> > > >
operator^(gealg::Expression<L> &&l, gealg::Expression<H> &&h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::OuterProduct<gealg::Expression<L>, gealg::Expression<H> > > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(std::forward<gealg::Expression<L> >(l), std::forward<gealg::Expression<H> >(h)));
}
#else
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::OuterProduct<gealg::Expression<L>, gealg::Expression<H> > > >
operator^(const gealg::Expression<L> &l, const gealg::Expression<H> &h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::OuterProduct<gealg::Expression<L>, gealg::Expression<H> > > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(l, h));
}
#endif

//variable of argument tuple
template <int I, class T>
inline gealg::Expression<gealg::ExpressionVariable<I, T> > var(const T &)
{
    return gealg::Expression<gealg::ExpressionVariable<I, T> >();
}

/*std::ostream& operator<<(std::ostream& out, const gealg::uint128_t& n)
{
   return out << (uint64_t)(n>>64) << (uint64_t)n;
}*/

template <uint8_t N, gealg::mv_confmap_t B, uint8_t S>
std::ostream &operator<<(std::ostream &out, const gealg::Multivector<N, B> &mv)
{
    out << "[ ";
    for (unsigned int i = 0; i < N; ++i)
    {
        out << mv[i] << " ";
    }
    out << "] (" << std::hex << std::setfill('0') << std::setw(2 * N) << (uint64_t)B << ")";

    if (S != 0x0)
    {
        out << " {" << (int)S << "}";
    }

    out << std::dec;

    return out;
}

/*template<class ET>
std::ostream& operator<<(std::ostream& out, gealg::Expression<ET> e)
{
   typedef typename ET::MultivectorType MV;
   //typename gealg::Expression<ExprType>::result_type mv = expr();
   MV mv;

   e.e_.evaluate(*((tuple<>*)NULL));
   gealg::EvaluateExpression<gealg::Expression<ET> >::operate(mv, e);

   out << "[ ";
   for(unsigned int i=0; i<MV::num; ++i) {
      out << mv[i] << " ";
   }
   out << "] (" << std::hex << std::setfill('0') << std::setw(2*MV::num) << (uint64_t)MV::bitmap << ")";

   if(MV::signature!=0x0) {
      out << " {" << (int)MV::signature << "}";
   }
   out << std::dec;

   return out;
}*/

template <class ExprType>
std::ostream &operator<<(std::ostream &out, gealg::Expression<ExprType> expr)
{
    typedef typename gealg::Expression<ExprType>::MultivectorType MV;
    MV mv;
    expr.e_.evaluate(tuple<>());
    gealg::EvaluateExpression<gealg::Expression<ExprType> >::operate(mv, expr);

    out << "[ ";
    for (unsigned int i = 0; i < MV::num; ++i)
    {
        out << mv[i] << " ";
    }

    static const uint64_t BPA = (sizeof(gealg::mv_confmap_t) > 8) ? (MV::bitmap >> 64) : 0;
    static const uint64_t BPB = (uint64_t)MV::bitmap;
    std::stringstream bitmapStream;
    bitmapStream << std::hex << std::setfill('0');
    if (MV::num > 8)
    {
        bitmapStream << std::setw(2 * (MV::num - 8)) << BPA << std::setw(8) << BPB;
    }
    else
    {
        bitmapStream << std::setw(2 * MV::num) << BPB;
    }
    //out << "] (" << std::hex << std::setfill('0') << std::setw((MV::num>8) ? (2*(8-MV::num)) : 0 ) << BPA
    //                                              << std::setw((MV::num>8) ? 8 : (2*MV::num) ) << BPB << ")";
    out << "] (" << bitmapStream.str() << ")";

    if (MV::signature != 0x0)
    {
        out << " {" << std::hex << (int)MV::signature << "}";
    }
    out << std::dec;

    return out;
}

//Tool operators depending on basic operators
namespace gealg
{

//Unary operators
//Magnitude
template <class E>
struct Magnitude
{
    typedef typename E::MultivectorType MT;
    typedef Multivector<1, 0x00, MT::signature> MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const E &e, const T &tuple)
    {
        Expression<MTR> square_magnitude = grade<0>((!e) * e)(tuple);

        return sqrt(square_magnitude[0]);
    }
};

//Inverse
template <class E>
struct Inverse
{
    typedef typename E::MultivectorType MT;
    typedef MT MTR;

    /*template<uint8_t EB, class T>
   static inline double evaluateElement(const E& e, const T& tuple) {
      Expression<Multivector<1, 0x00, MT::signature> > square_magnitude; square_magnitude = grade<0>((!e)*e)(tuple);

      return Power<-1, BitCount<EB>::value*(BitCount<EB>::value-1)/2>::value * e.template element<EB>(tuple) / square_magnitude[0];
   }*/

    template <class T>
    static inline MTR evaluate(E &e, const T &vars)
    {
        MTR result;

        typename E::result_type m = e(vars);
        Expression<Multivector<1, 0x00, MT::signature> > square_magnitude;
        square_magnitude = grade<0>((!m) * m)(vars);
        /*double sm = 0.0;
      for(int i = 0; i<MT::num; ++i) {
         sm += m[i]*m[i];
      }*/

        DivideReversedElement<>::operate(result, m.e_, square_magnitude[0]);

        return result;
    }

    template <uint8_t I = MT::num, uint8_t dummy = 0x0> //dummy: C++ doesn't allow for full specialization in non-namespace scope, but for partial!
    struct DivideReversedElement
    {
        static inline void operate(MTR &result, const MT &m, const double &d)
        {
            static const uint8_t bitmapP = (uint8_t)(MT::bitmap >> (8 * (MT::num - I)));
            result[MT::num - I] = Power<-1, BitCount<bitmapP>::value *(BitCount<bitmapP>::value - 1) / 2>::value * m[MT::num - I] / d;
            DivideReversedElement<I - 1, 0>::operate(result, m, d);
        }
    };

    template <uint8_t dummy>
    struct DivideReversedElement<0, dummy>
    {
        static inline void operate(MTR &, const MT &, const double &)
        {
        }
    };
};

//Exponential Function
template <class E>
struct ExponentialFunction
{
    typedef typename E::MultivectorType MT;
    typedef Multivector < ((uint8_t)MT::bitmap == 0x0) ? MT::num : (MT::num + 1), ((uint8_t)MT::bitmap == 0x0) ? MT::bitmap : (MT::bitmap << 8 | 0x0), MT::signature > MTR;

#ifdef __CPP0X__
    template <class T>
    static inline MTR &&evaluate(const MT &m, const T &vars)
    {
        double s = grade<0>((!Expression<MT>(m)) * Expression<MT>(m))()[0];
        double p = 1.0;
        double r_mv = 0.0;
        double r_n = 0.0;
        SeriesTwoElements<>::operate(r_mv, r_n, p, s);

        MTR result = (part<MTR::num, MTR::bitmap>(m * r_mv))().e_;
        result.element<0x0>() += r_n;

        return std::move(result);
    }
#else
    template <class T>
    static inline MTR evaluate(E &e, const T &vars)
    {
        MTR result;

        typename E::result_type m = e(vars);
        EvaluateExponent<BitCount<MTR::bitmap>::value>::operate(result, m);

        return result;
    }
#endif

    //typedef typename MultiplyResultMultivectorConfig<MT, MT>::Type M_SQARE;
    template <uint8_t G, int dummy = 0x0>
    struct EvaluateExponent
    {
        //static inline void operate(MTR& result, const E::result_type& m) {
        static inline void operate(MTR &result, const typename E::result_type &m)
        {
            double s = grade<0>(m * m)()[0];
            double p = 1.0;
            double r_mv = 0.0;
            double r_n = 0.0;
            SeriesTwoElements<>::operate(r_n, r_mv, p, s);

            result = (part<MTR::num, MTR::bitmap>(m * r_mv))().e_;
            result[0] += r_n;
        }
    };

#ifdef _MSC_VER
    static const int BIVECNUM = 2 * MT::num;
    template <int dummy>
    struct EvaluateExponent<BIVECNUM, dummy>
    { //only valid for bivectors of algebra with dimension n<5, shouldn't work for conformal model
#else
    template <int dummy>
    struct EvaluateExponent<2 * MT::num, dummy>
    { //only valid for bivectors of algebra with dimension n<5, shouldn't work for conformal model
#endif
        static inline void operate(MTR &result, const typename E::result_type &m)
        {
            double a_square = grade<0>(m * m)()[0];
            if (a_square < 0.0)
            {
                double a = sqrt(fabs(a_square));
                result = (m * (sin(a) / a) + gealg::mv<1, 0x00>::type(cos(a)))().e_;
            }
            else if (a_square > 0.0)
            {
                double a = sqrt(a_square);
                result = (m * (sinh(a) / a) + gealg::mv<1, 0x00>::type(cosh(a)))().e_;
            }
            else
            {
                result = (m + gealg::mv<1, 0x00>::type(1.0))().e_;
            }
        }
    };
    template <int dummy>
    struct EvaluateExponent<0, dummy>
    {
        //static inline void operate(MTR& result, const E::result_type& m) {
        static inline void operate(MTR &result, const typename E::result_type &m)
        {
            result[0] = exp(m[0]);
        }
    };

    template <int I = 0, int dummy = 0>
    struct SeriesTwoElements
    {
        static void operate(double &r_n, double &r_mv, double &p, const double &s)
        {
            r_n += (1.0 / ((double)Factorial<2 * I>::value)) * p;
            r_mv += (1.0 / ((double)Factorial<2 * I + 1>::value)) * p;

            p *= s;

            //std::cout << "I " << I << ": F1: " << (1.0/((double)Factorial<2*I>::value)) << ", F2: " << (1.0/((double)Factorial<2*I+1>::value)) << ", p: " << p << std::endl;
            SeriesTwoElements<I + 1>::operate(r_n, r_mv, p, s);
        }
    };
    template <int dummy>
    struct SeriesTwoElements<6, dummy>
    {
        static void operate(double &, double &, double &, const double &)
        {
        }
    };
};

#ifndef __CUDACC__
//0-Grade functions:
template <class E, double func(double)>
struct ZeroGradeUnaryFunction
{
    typedef typename E::MultivectorType MT;
    typedef MT MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const E &e, const T &tuple)
    {
        return func(e.template element<EB>(tuple));
    }

    template <uint8_t IR = MTR::num, uint8_t dummy = 0> //Inverted index: IR
    struct FuncElement
    {
        static inline void operate(MTR &result, const MT &m)
        {
            result[MTR::num - IR] = func(m[MTR::num - IR]);

            FuncElement<IR - 1>::operate(result, m);
        }
    };
    template <uint8_t dummy> //Inverted index: IR
    struct FuncElement<0, dummy>
    {
        static inline void operate(MTR &, const MT &)
        {
        }
    };
};

//0-Grade functions:
template <class EL, class EH, double func(double, double)>
struct ZeroGradeBinaryFunction
{
    typedef typename EL::MultivectorType MTL;
    typedef typename EH::MultivectorType MTH;
    typedef MTL MTR;

    template <uint8_t EB, class T>
    static inline double evaluateElement(const EL &el, const EH &eh, const T &tuple)
    {
        return func(el.template element<EB>(tuple), eh.template element<EB>(tuple));
    }

    template <uint8_t IR = MTR::num, uint8_t dummy = 0> //Inverted index: IR
    struct FuncElement
    {
        static inline void operate(MTR &result, const MTL &l, const MTH &h)
        {
            result[MTR::num - IR] = func(l[MTR::num - IR], h[MTR::num - IR]);

            FuncElement<IR - 1>::operate(result, l, h);
        }
    };
    template <uint8_t dummy> //Inverted index: IR
    struct FuncElement<0, dummy>
    {
        static inline void operate(MTR &, const MTL &, const MTH &)
        {
        }
    };
};

//Zero grade helper function
struct Sign
{
    static inline double func(double value)
    {
        return (value >= 0) ? 1.0 : -1.0;
    }
};
#endif

} //end namespace gealg

//Magnitude
#ifdef __CPP0X__
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Magnitude<gealg::Expression<M> > > >
magnitude(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Magnitude<gealg::Expression<M> > > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Magnitude<gealg::Expression<M> > > >
magnitude(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Magnitude<gealg::Expression<M> > > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//Inverse
#ifdef __CPP0X__
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Inverse<gealg::Expression<M> >, true> >
operator~(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Inverse<gealg::Expression<M> >, true> UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::Inverse<gealg::Expression<M> >, true> >
operator~(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::Inverse<gealg::Expression<M> >, true> UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//Exponential function
#ifdef __CPP0X__
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ExponentialFunction<gealg::Expression<M> >, true> >
exp(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ExponentialFunction<gealg::Expression<M> >, true> UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ExponentialFunction<gealg::Expression<M> >, true> >
exp(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ExponentialFunction<gealg::Expression<M> >, true> UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

#ifndef __CUDACC__
//Basic standard library math function wrapper for 0-grade multivectors
//Sinus function: Only 0-Grade yet
#ifdef __CPP0X__
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, sin> > >
sin(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, sin> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, sin> > >
sin(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, sin> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//Cocosus function: Only 0-Grade yet
#ifdef __CPP0X__
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, cos> > >
cos(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, cos> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, cos> > >
cos(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, cos> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//Arcus tangens function: Only 0-Grade yet
#ifdef __CPP0X__
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, atan> > >
atan(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, atan> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, atan> > >
atan(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, atan> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//Square root function: Only 0-Grade yet
#ifdef __CPP0X__
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, sqrt> > >
sqrt(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, sqrt> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, sqrt> > >
sqrt(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, sqrt> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//tangens hyperbolicus function: Only 0-Grade yet
#ifdef __CPP0X__
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, tanh> > >
tanh(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, tanh> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, tanh> > >
tanh(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, tanh> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//sign function: Only 0-Grade yet
#ifdef __CPP0X__
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, gealg::Sign::func> > >
sgn(gealg::Expression<M> &&m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, gealg::Sign::func> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(std::forward<gealg::Expression<M> >(m)));
}
#else
template <class M>
inline gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, gealg::Sign::func> > >
sgn(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, gealg::ZeroGradeUnaryFunction<gealg::Expression<M>, gealg::Sign::func> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}
#endif

//atan2
#ifdef __CPP0X__
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::ZeroGradeBinaryFunction<gealg::Expression<L>, gealg::Expression<H>, atan2> > >
atan2(gealg::Expression<L> &&l, gealg::Expression<H> &&h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::ZeroGradeBinaryFunction<gealg::Expression<L>, gealg::Expression<H>, atan2> > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(std::forward<gealg::Expression<L> >(l), std::forward<gealg::Expression<H> >(h)));
}
#else
template <class L, class H>
inline gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::ZeroGradeBinaryFunction<gealg::Expression<L>, gealg::Expression<H>, atan2> > >
atan2(const gealg::Expression<L> &l, const gealg::Expression<H> &h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, gealg::ZeroGradeBinaryFunction<gealg::Expression<L>, gealg::Expression<H>, atan2> > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(l, h));
}
#endif

#endif //ifndef(__CUDACC__)

#endif
