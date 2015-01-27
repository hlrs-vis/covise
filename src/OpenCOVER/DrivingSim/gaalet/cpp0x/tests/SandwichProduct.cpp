/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

template <typename L, typename R>
inline auto sandwich(const gaalet::expression<L> &l, const gaalet::expression<R> &r) -> decltype((r * l * (~r)))
{
    return (r * l * (~r));
}

/*template<typename L, typename R> inline
auto sandwich(const gaalet::expression<L>& l_, const gaalet::expression<R>& r_) -> decltype(R()*L()*(~R()))
{
   const L& l(l_);
   const R& r(r_);

   return (r*l*(~r));
}*/

namespace gaalet
{

template <class L, class R>
struct geometric_product<L, geometric_product<R, reverse<L> > > : public expression<geometric_product<L, geometric_product<R, reverse<L> > > >
{
    typedef typename element_type_combination_traits<typename L::element_t, typename R::element_t>::element_t element_t;
    typedef typename metric_combination_traits<typename L::metric, typename R::metric>::metric metric;

    typedef typename R::clist clist;

    geometric_product(const L &l_, const R &r_)
        : l(l_)
        , r(r_)
    {
    }

    template <conf_t conf>
    element_t element() const
    {
        return -1;
    }

protected:
    const L &l;
    const R &r;
};
}

typedef gaalet::algebra<gaalet::signature<3, 0> > em;

int main()
{
    em::mv<1, 2, 4>::type v = { 1.0, 2.0, 3.0 };
    em::mv<0, 3, 5, 6>::type R = { cos(-0.5 * 0.5 * M_PI), sin(-0.5 * 0.5 * M_PI), 0.0, 0.0 };

    auto RvrR = R * v * ~R;
    std::cout << "R*v*~R: " << RvrR << std::endl;
    std::cout << "<R*v*~R>_1: " << grade<1>(R * v * ~R) << std::endl;

    const auto &s = sandwich(v, R);
    std::cout << "sandwich(v,R): " << s << std::endl;

    auto a = R * (v * ~R);

    std::cout << "a: " << a << std::endl;
}
