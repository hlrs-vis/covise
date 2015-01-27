/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"
#include <cmath>

int main()
{
    typedef gaalet::algebra<gaalet::signature<4, 1> > cm;
    cm::mv<0x01>::type e1 = { 1.0 };
    cm::mv<0x02>::type e2 = { 1.0 };
    cm::mv<0x04>::type e3 = { 1.0 };
    cm::mv<0x08>::type ep = { 1.0 };
    cm::mv<0x10>::type em = { 1.0 };

    cm::mv<0x00>::type one = { 1.0 };
    std::cout << "sin(one): " << sin(one) << std::endl;

    cm::mv<0x08, 0x10>::type e0 = 0.5 * (em - ep);
    cm::mv<0x08, 0x10>::type einf = em + ep;

    cm::mv<0x18>::type E = ep * em;

    cm::mv<0x1f>::type I = e1 * e2 * e3 * ep * em;
    auto I_expr = e1 * e2 * e3 * ep * em;
    auto I_mv = eval(e1 * e2 * e3 * ep * em);
    std::cout << "I_expr: " << I_expr << ", I_mv[0]: " << I_mv[0] << std::endl;
    cm::mv<0x07>::type i = e1 * e2 * e3;

    std::cout << "e0*e0: " << e0 *e0 << std::endl;
    std::cout << "einf*einf: " << einf *einf << std::endl;
    std::cout << "ep*ep: " << ep *ep << std::endl;
    std::cout << "em*em: " << em *em << std::endl;
    std::cout << "E: " << E << std::endl;
    std::cout << "ep*em: " << ep *em << std::endl;
    std::cout << "em*ep: " << em *ep << std::endl;
    std::cout << "e0*einf: " << e0 *einf << std::endl;
    std::cout << "einf*e0: " << einf *e0 << std::endl;
    std::cout << "e0&einf: " << (e0 & einf) << std::endl;
    std::cout << "einf&e0: " << (einf & e0) << std::endl;
    std::cout << "e0^einf: " << (e0 ^ einf) << std::endl;
    std::cout << "einf^e0: " << (einf ^ e0) << std::endl;
    std::cout << "E*einf: " << (E * einf) << std::endl;
    std::cout << "einf*E: " << (einf * E) << std::endl;

    std::cout << "e1*E: " << e1 *E << std::endl;
    std::cout << "e1^E: " << (e1 ^ E) << std::endl;
    std::cout << "(e1^E)*einf: " << ((e1 ^ E) * einf) << std::endl;

    //combination:
    typedef gaalet::algebra<gaalet::signature<3, 0> > tdem;
    tdem::mv<0x01, 0x02, 0x04>::type v;
    auto V = v * i;
    typedef decltype(V) V_type;
    std::cout << "V=v*i: " << V << ", signature: (" << V_type::metric::p << ", " << V_type::metric::q << ")" << std::endl;

    //violates static_assert:
    //auto V2 = gaalet::algebra<gaalet::signature<3,2>>::mv<0x01>::type()*i;
    //typedef decltype(V2) V2_type;
    //std::cout << "V2=v*i: " << V2 << ", signature: (" << V2_type::metric::p << ", " << V2_type::metric::q <<")" << std::endl;
}
