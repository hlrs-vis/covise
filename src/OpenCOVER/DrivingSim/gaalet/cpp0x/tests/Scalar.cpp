/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

template <class E>
void eval_copy(E expr)
{
    std::cout << "eval_copy(): " << expr << ", expr size: " << sizeof(expr) << std::endl;
}

int main()
{
    gaalet::mv<1, 2, 4>::type a = { 1, 2, 3 };
    gaalet::mv<1, 2, 4>::type b = { 3, 4, 5 };

    gaalet::mv<0, 3, 5, 6>::type R = { cos(0.25 * M_PI), sin(0.25 * M_PI), 0.0, 0.0 };

    std::cout << "scalar(a,a): " << scalar(a, a) << std::endl;

    auto expr_grade = grade<0>(a * a);
    std::cout << "grade<0>(a,a): " << expr_grade << std::endl;
    eval_copy(expr_grade);

    auto expr_scalar = scalar(a, a);

    auto m = eval(expr_scalar);
    std::cout << "m: " << m << std::endl;

    m = eval(expr_scalar);
    std::cout << "m: " << m << std::endl;

    m = eval(expr_scalar);
    std::cout << "m: " << m << std::endl;
    eval_copy(expr_scalar);
}
