/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

typedef gaalet::algebra<gaalet::signature<3, 0> > em;

int main()
{
    em::mv<1, 2, 4>::type a(1, 2, 3);
    em::mv<1, 2, 4>::type b(4, 5, 6);

    typedef gaalet::geometric_product<em::mv<1, 2, 4>::type, em::mv<1, 2, 4>::type> c_type;

    std::cout << "clist size: " << c_type::clist::size << std::endl;
    std::cout << "element 0: " << gaalet::get_element<0, c_type::clist>::value << std::endl;
    std::cout << "element 1: " << gaalet::get_element<1, c_type::clist>::value << std::endl;
    std::cout << "element 2: " << gaalet::get_element<2, c_type::clist>::value << std::endl;
    std::cout << "element 3: " << gaalet::get_element<3, c_type::clist>::value << std::endl;

    gaalet::multivector<c_type::clist, em, gaalet::default_element_t> c = a * b;
    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "c: " << c << std::endl;
}
