/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"
#include <cmath>
#include <sstream>
#include <typeinfo>

typedef gaalet::algebra<gaalet::signature<3, 0> > em;

template <class O>
std::string hexdump(const O &object)
{
    std::stringstream dump;
    for (int i = 0; i < sizeof(object); ++i)
    {
        dump << std::hex << (unsigned int)*((char *)(&object) + i) << " ";
    }
    return dump.str();
};

int main()
{
    em::mv<1, 2, 4>::type a(1, 2, 3);
    em::mv<1, 2, 4>::type b(3, 4, 5);

    em::mv<0, 3, 5, 6>::type R(cos(0.25 * M_PI), sin(0.25 * M_PI), 0.0, 0.0);

    std::cout << "grade<0>(a*a): " << grade<0>(a * a) << std::endl;
    std::cout << "scalar(a,a): " << scalar(a, a) << std::endl;

    typeof(grade<0>(a * a)) expr_grade = grade<0>(a * a);
    typeof(eval(expr_grade)) n = eval(expr_grade);
    std::cout << "n: " << n << std::endl;
    n = eval(expr_grade);
    std::cout << "n: " << n << std::endl;
    n = eval(expr_grade);
    std::cout << "n: " << n << std::endl;

    typeof(scalar(a, a)) expr_scalar = scalar(a, a);
    typeof(eval(expr_scalar)) m = eval(expr_scalar);
    std::cout << "m: " << m << std::endl;
    m = eval(expr_scalar);
    std::cout << "m: " << m << std::endl;
    m = eval(expr_scalar);
    std::cout << "m: " << m << std::endl;

    typedef typeof(grade<0>(a * a)) type_grade;
    typedef typeof(scalar(a, a)) type_scalar;

    std::string name_grade = typeid(type_grade).name();
    std::string name_scalar = typeid(type_scalar).name();
    std::cout << "type grade<0>(a*a): " << name_grade << ", size: " << sizeof(name_grade) << std::endl;
    std::cout << "type scalar(a,a): " << name_scalar << ", size: " << sizeof(name_scalar) << std::endl;
    if (name_grade == name_scalar)
        std::cout << "names equal!" << std::endl;

    std::cout << "pointer to a: " << std::hex << (&a) << std::endl;
    std::cout << "hexdump grade: " << hexdump(expr_grade) << std::endl;
    std::cout << "hexdump scalar: " << hexdump(expr_scalar) << std::endl;
}
