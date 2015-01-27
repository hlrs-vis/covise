/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef __GAALET_STREAMING_H
#define __GAALET_STREAMING_H

#include "expression.h"
#include "multivector.h"

//expression streaming
template <typename E, typename clist>
struct UnpackElementsToStream
{
    static void unpack(std::ostream &os, const E &e)
    {
        os << e.template element<clist::head>() << ' ';
        UnpackElementsToStream<E, typename clist::tail>::unpack(os, e);
    }
};

template <typename E>
struct UnpackElementsToStream<E, gaalet::cl_null>
{
    static void unpack(std::ostream &, const E &)
    {
    }
};

template <typename clist>
struct UnpackConfigurationListToStream
{
    static void unpack(std::ostream &os)
    {
        os << clist::head << ' ';
        UnpackConfigurationListToStream<typename clist::tail>::unpack(os);
    }
};

template <>
struct UnpackConfigurationListToStream<gaalet::cl_null>
{
    static void unpack(std::ostream &)
    {
    }
};

template <class E>
std::ostream &operator<<(std::ostream &os, const gaalet::expression<E> &e_)
{
    const E &e(e_);

    os << "[ " << std::dec;
    UnpackElementsToStream<E, typename E::clist>::unpack(os, e);
    os << "] { " << std::hex;
    UnpackConfigurationListToStream<typename E::clist>::unpack(os);
    os << '}';

    return os;
}

#endif
