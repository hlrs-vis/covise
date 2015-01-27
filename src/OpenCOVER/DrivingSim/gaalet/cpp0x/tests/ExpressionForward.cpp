/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

typedef gaalet::algebra<gaalet::signature<3, 0> > em;

template <typename E>
void printExpr(const gaalet::expression<E> &expr)
{
    std::cout << "Expression: " << expr << std::endl;
}

template <typename E>
void printExprC(gaalet::expression<E> expr)
{
    std::cout << "Expression: " << expr << std::endl;
}

template <typename B>
auto Rotor(const gaalet::expression<B> &b) -> decltype(exp(b))
{
    //std::cout << "Rotor(): " << exp(b) << std::endl;
    return exp(b);
}

template <typename E>
auto mag(const gaalet::expression<E> &e) -> decltype((e * ~e))
{
    return (e * ~e);
}

int main()
{
    em::mv<1, 2, 4>::type a = { 1, 2, 3 };
    em::mv<3, 5, 6>::type b = { -0.5 * M_PI * 0.5, 0, 0 };

    std::cout << "a*b: " << a *b << std::endl;

    printExpr(a);
    auto R = Rotor(b);
    std::cout << "R: " << R << std::endl;
    //em::mv<0>::type s = {-0.5};
    //auto R = exp(s*b);
    //auto R = exp(em::mv<0>::type({-0.5})*b);
    //double s = -0.5;
    //auto R = exp(s*b);
    //auto R = exp(-0.5*b);
    printExpr(R);
    auto c = R * a * ~R;
    printExpr(c);

    std::cout << "mag(a): " << mag(a) << std::endl;
    std::cout << "mag(a): " << mag(a).element<0x00>() << std::endl;
    auto maga = eval(mag(a));
    std::cout << "mag(a): " << maga << std::endl;
    auto magR = mag(R);
    std::cout << "magR: " << magR.element<0x00>() << std::endl;
    //eval(magR);
    std::cout << "mag(R): " << magR << std::endl;
    printExpr(mag(b));

    printExprC(c);
}
