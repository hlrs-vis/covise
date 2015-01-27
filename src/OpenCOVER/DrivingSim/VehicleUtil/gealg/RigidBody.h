/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __RigidBody_h
#define __RigidBody_h

namespace rigidbody
{

template <class E, int I1, int I2, int I3>
struct Inertia
{
    typedef typename E::MultivectorType MT;
    typedef gealg::Multivector<3, 0x060503> MTR;

    static MTR evaluate(const MT &m)
    {
        MTR result;

        static const uint8_t IB3 = gealg::MultivectorElementBitmapSearch<MT, 0x03>::index;
        static const uint8_t IB5 = gealg::MultivectorElementBitmapSearch<MT, 0x05>::index;
        static const uint8_t IB6 = gealg::MultivectorElementBitmapSearch<MT, 0x06>::index;

        result[0] = (IB3 == MT::num) ? 0.0 : m[IB3] * (double)I3;
        result[1] = (IB5 == MT::num) ? 0.0 : m[IB5] * (double)I2;
        result[2] = (IB6 == MT::num) ? 0.0 : m[IB6] * (double)I1;

        return result;
    }

    template <uint8_t EB>
    static double evaluateElement(const E &e)
    {
        static const int IN = (EB == 0x03) ? I3 : (EB == 0x05) ? I2 : (EB == 0x06) ? I1 : 0.0;

        return e.template element<EB>() * (double)IN;
    }

    template <uint8_t EB, class T>
    static double evaluateElement(const E &e, const T &tuple)
    {
        static const int IN = (EB == 0x03) ? I3 : (EB == 0x05) ? I2 : (EB == 0x06) ? I1 : 0.0;

        return e.template element<EB>(tuple) * (double)IN;
    }
};

//Inertia
template <int I1, int I2, int I3, class M>
gealg::Expression<gealg::UnaryExpression<gealg::Expression<M>, Inertia<gealg::Expression<M>, I1, I2, I3> > >
inertia(const gealg::Expression<M> &m)
{
    typedef gealg::UnaryExpression<gealg::Expression<M>, Inertia<gealg::Expression<M>, I1, I2, I3> > UnaryExpressionType;
    return gealg::Expression<UnaryExpressionType>(UnaryExpressionType(m));
}

template <class EL, class EH, int I1, int I2, int I3>
struct Euler
{
    typedef typename EL::MultivectorType MTL; //angular velocity
    typedef typename EH::MultivectorType MTH; //torque
    typedef typename gealg::Multivector<3, 0x060503> MTR;

#ifdef __CPP0X__
    static MTR &&evaluate(const MTL &l, const MTH &h)
    {
        MTR result;

        static const uint8_t IV3 = gealg::MultivectorElementBitmapSearch<MTL, 0x03>::index;
        static const uint8_t IV5 = gealg::MultivectorElementBitmapSearch<MTL, 0x05>::index;
        static const uint8_t IV6 = gealg::MultivectorElementBitmapSearch<MTL, 0x06>::index;
        static const uint8_t IB3 = gealg::MultivectorElementBitmapSearch<MTH, 0x03>::index;
        static const uint8_t IB5 = gealg::MultivectorElementBitmapSearch<MTH, 0x05>::index;
        static const uint8_t IB6 = gealg::MultivectorElementBitmapSearch<MTH, 0x06>::index;

        //converted euler equations to 3d-euclidean-space duality in geometric algebra: not sure about +'ses before (double)(Ix-Iy)...
        result[0] = (((IB3 == MTH::num) ? 0.0 : h[IB3]) + (double)(I2 - I1) * (((IV6 == MTL::num) ? 0.0 : l[IV6])) * (((IV5 == MTL::num) ? 0.0 : l[IV5]))) / I3;
        result[1] = (((IB5 == MTH::num) ? 0.0 : h[IB5]) + (double)(I1 - I3) * (((IV3 == MTL::num) ? 0.0 : l[IV3])) * (((IV6 == MTL::num) ? 0.0 : l[IV6]))) / I2;
        result[2] = (((IB6 == MTH::num) ? 0.0 : h[IB6]) + (double)(I3 - I2) * (((IV5 == MTL::num) ? 0.0 : l[IV5])) * (((IV3 == MTL::num) ? 0.0 : l[IV3]))) / I1;

        return std::move(result);
    }
#else
    static MTR evaluate(const MTL &l, const MTH &h)
    {
        MTR result;

        static const uint8_t IV3 = gealg::MultivectorElementBitmapSearch<MTL, 0x03>::index;
        static const uint8_t IV5 = gealg::MultivectorElementBitmapSearch<MTL, 0x05>::index;
        static const uint8_t IV6 = gealg::MultivectorElementBitmapSearch<MTL, 0x06>::index;
        static const uint8_t IB3 = gealg::MultivectorElementBitmapSearch<MTH, 0x03>::index;
        static const uint8_t IB5 = gealg::MultivectorElementBitmapSearch<MTH, 0x05>::index;
        static const uint8_t IB6 = gealg::MultivectorElementBitmapSearch<MTH, 0x06>::index;

        //converted euler equations to 3d-euclidean-space duality in geometric algebra: not sure about +'ses before (double)(Ix-Iy)...
        result[0] = (((IB3 == MTH::num) ? 0.0 : h[IB3]) + (double)(I2 - I1) * (((IV6 == MTL::num) ? 0.0 : l[IV6])) * (((IV5 == MTL::num) ? 0.0 : l[IV5]))) / I3;
        result[1] = (((IB5 == MTH::num) ? 0.0 : h[IB5]) + (double)(I1 - I3) * (((IV3 == MTL::num) ? 0.0 : l[IV3])) * (((IV6 == MTL::num) ? 0.0 : l[IV6]))) / I2;
        result[2] = (((IB6 == MTH::num) ? 0.0 : h[IB6]) + (double)(I3 - I2) * (((IV5 == MTL::num) ? 0.0 : l[IV5])) * (((IV3 == MTL::num) ? 0.0 : l[IV3]))) / I1;

        return result;
    }
#endif

    template <uint8_t EB>
    static double evaluateElement(const EL &el, const EH &eh)
    {
        return (EB == 0x03) ? ((eh.template element<0x03>() - (double)(I2 - I1) * (el.template element<0x06>()) * (el.template element<0x05>())) / I3) : (EB == 0x05) ? ((eh.template element<0x05>() - (double)(I1 - I3) * (el.template element<0x03>()) * (el.template element<0x06>())) / I2) : (EB == 0x06) ? ((eh.template element<0x06>() - (double)(I3 - I2) * (el.template element<0x05>()) * (el.template element<0x03>())) / I1) : 0.0;
    }
    template <uint8_t EB, class T>
    static double evaluateElement(const EL &el, const EH &eh, const T &tuple)
    {
        return (EB == 0x03) ? ((eh.template element<0x03>(tuple) - (double)(I2 - I1) * (el.template element<0x06>(tuple)) * (el.template element<0x05>(tuple))) / I3) : (EB == 0x05) ? ((eh.template element<0x05>(tuple) - (double)(I1 - I3) * (el.template element<0x03>(tuple)) * (el.template element<0x06>(tuple))) / I2) : (EB == 0x06) ? ((eh.template element<0x06>(tuple) - (double)(I3 - I2) * (el.template element<0x05>(tuple)) * (el.template element<0x03>(tuple))) / I1) : 0.0;
    }
};

#ifdef __CPP0X__
template <int I1, int I2, int I3, class L, class H>
gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, Euler<gealg::Expression<L>, gealg::Expression<H>, I1, I2, I3> > >
euler(gealg::Expression<L> &&l, gealg::Expression<H> &&h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, Euler<gealg::Expression<L>, gealg::Expression<H>, I1, I2, I3> > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(std::forward<gealg::Expression<L> >(l), std::forward<gealg::Expression<H> >(h)));
}
#else
template <int I1, int I2, int I3, class L, class H>
gealg::Expression<gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, Euler<gealg::Expression<L>, gealg::Expression<H>, I1, I2, I3> > >
euler(const gealg::Expression<L> &l, const gealg::Expression<H> &h)
{
    typedef gealg::BinaryExpression<gealg::Expression<L>, gealg::Expression<H>, Euler<gealg::Expression<L>, gealg::Expression<H>, I1, I2, I3> > BinaryExpressionType;
    return gealg::Expression<BinaryExpressionType>(BinaryExpressionType(l, h));
}
#endif
}

#endif
