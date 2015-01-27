/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"
#include "ginac/ginac.h"

namespace gaalet
{

template <>
struct null_element<GiNaC::ex>
{
    static GiNaC::ex value()
    {
        return GiNaC::symbol("0");
    }
};

}; //end namespace gaalet

typedef gaalet::algebra<gaalet::signature<3, 0>, GiNaC::ex> gem;

int main()
{
    gem::mv<0>::type s = { GiNaC::symbol("s") };
    gem::mv<0>::type t = { GiNaC::symbol("t") };

    auto u = s + t;

    std::cout << "s: " << s << std::endl;
    std::cout << "t: " << t << std::endl;
    std::cout << "u: " << u << std::endl;

    gem::mv<1, 2, 4>::type a = { GiNaC::symbol("a1"), GiNaC::symbol("a2"), GiNaC::symbol("a3") };
    gem::mv<1, 2, 4>::type b = { GiNaC::symbol("b1"), GiNaC::symbol("b2"), GiNaC::symbol("b3") };

    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "a+b: " << a + b << std::endl;
    std::cout << "a*b: " << a *b << std::endl;
}
