/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __CGA_OSG_H
#define __CGA_OSG_H

#include "gaalet.h"
#include <osg/Shape>

namespace cga
{

typedef gaalet::algebra<gaalet::signature<4, 1> > cga;

//defintion of basisvectors, null basis, pseudoscalars, helper unit scalar
cga::mv<0x01>::type e1 = { 1.0 };
cga::mv<0x02>::type e2 = { 1.0 };
cga::mv<0x04>::type e3 = { 1.0 };
cga::mv<0x08>::type ep = { 1.0 };
cga::mv<0x10>::type em = { 1.0 };

cga::mv<0x00>::type one = { 1.0 };

cga::mv<0x08, 0x10>::type e0 = 0.5 * (em - ep);
cga::mv<0x08, 0x10>::type einf = em + ep;

cga::mv<0x18>::type E = ep * em;

cga::mv<0x1f>::type Ic = e1 * e2 * e3 * ep * em;
cga::mv<0x07>::type Ie = e1 * e2 * e3;

typedef cga::mv<0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x11, 0x12, 0x14>::type S_type;
typedef cga::mv<0x00, 0x03, 0x05, 0x06, 0x09, 0x0a, 0x0c, 0x0f, 0x11, 0x12, 0x14, 0x17>::type D_type;

namespace osg
{

    class Sphere : public cga::mv<1, 2, 4, 8, 16>::type, public ::osg::Sphere
    {
    public:
        //initialization
        Sphere()
            : multivector({ 0.0, 0.0, 0.0, 0.0, 1.0 })
            , ::osg::Sphere()
        {
        }

        Sphere(std::initializer_list<element_t> s)
            : multivector(s)
            , ::osg::Sphere(::osg::Vec3(data[0] / data[4], data[1] / data[4], data[2] / data[4]), 2.0 * sqrt((-data[3] / data[4] + (data[0] * data[0] + data[1] * data[1] + data[2] * data[2]) / (data[4] * data[4]))))
        {
            auto x = part<1, 2, 4>(*this) * (!((-1.0) * (*this & einf)));
            setCenter(::osg::Vec3(x.element<1>(), x.element<2>(), x.element<4>()));
            auto r = sqrt(eval((x & x) - 2.0 * ((-1.0) * (*this & e0))));
            setRadius(r);
        }

        template <class E>
        Sphere(const gaalet::expression<E> &e_)
            : multivector(e_)
            , ::osg::Sphere()
        {
            auto x = part<1, 2, 4>(*this) * (!((-1.0) * (*this & einf)));
            setCenter(::osg::Vec3(x.element<1>(), x.element<2>(), x.element<4>()));
            auto r = sqrt(eval((x & x) - 2.0 * ((-1.0) * (*this & e0))));
            setRadius(r);
        }

        template <class E>
        void update(const expression<E> &e_)
        {
            this->multivector::operator=(multivector(e_));

            auto x = part<1, 2, 4>(*this) * (!((-1.0) * (*this & einf)));
            setCenter(::osg::Vec3(x.element<1>(), x.element<2>(), x.element<4>()));
            auto r = sqrt(eval((x & x) - 2.0 * ((-1.0) * (*this & e0))));
            setRadius(r);
        }
    };

    class Point : public cga::mv<1, 2, 4, 8, 16>::type, public ::osg::Sphere
    {
    public:
        //initialization
        Point()
            : multivector({ 0.0, 0.0, 0.0, 0.0, 1.0 })
            , ::osg::Sphere()
        {
            setRadius(0.05);
        }

        Point(std::initializer_list<element_t> s)
            : multivector(s)
            , ::osg::Sphere()
        {
            auto x = part<1, 2, 4>(*this) * (!((-1.0) * (*this & einf)));
            setCenter(::osg::Vec3(x.element<1>(), x.element<2>(), x.element<4>()));
            setRadius(0.05);
        }

        template <class E>
        Point(const gaalet::expression<E> &e_)
            : multivector(e_)
            , ::osg::Sphere()
        {
            auto x = part<1, 2, 4>(*this) * (!((-1.0) * (*this & einf)));
            setCenter(::osg::Vec3(x.element<1>(), x.element<2>(), x.element<4>()));
            setRadius(0.05);
        }

        template <class E>
        void update(const expression<E> &e_)
        {
            this->multivector::operator=(multivector(e_));

            auto x = part<1, 2, 4>(*this) * (!((-1.0) * (*this & einf)));
            setCenter(::osg::Vec3(x.element<1>(), x.element<2>(), x.element<4>()));
        }
    };
} //end namspace osg

} //end namspace cga

#endif
