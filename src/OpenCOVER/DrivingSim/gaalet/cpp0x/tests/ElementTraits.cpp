/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"
#include <iostream>
#include <typeinfo>

int main()
{
    std::cout << "float - float: " << typeid(gaalet::element_type_combination_traits<float, float>::element_t).name() << std::endl;
    std::cout << "double - double: " << typeid(gaalet::element_type_combination_traits<double, double>::element_t).name() << std::endl;
    std::cout << "int - int: " << typeid(gaalet::element_type_combination_traits<int, int>::element_t).name() << std::endl;
    std::cout << "long long - long long: " << typeid(gaalet::element_type_combination_traits<long long, long long>::element_t).name() << std::endl;
    std::cout << "unsigned int - unsigned int: " << typeid(gaalet::element_type_combination_traits<unsigned int, unsigned int>::element_t).name() << std::endl;
    std::cout << "float - double: " << typeid(gaalet::element_type_combination_traits<float, double>::element_t).name() << std::endl;
    std::cout << "double - float: " << typeid(gaalet::element_type_combination_traits<double, float>::element_t).name() << std::endl;
    std::cout << "int - float: " << typeid(gaalet::element_type_combination_traits<int, float>::element_t).name() << std::endl;
    std::cout << "float - int: " << typeid(gaalet::element_type_combination_traits<float, int>::element_t).name() << std::endl;
    std::cout << "int - double: " << typeid(gaalet::element_type_combination_traits<int, double>::element_t).name() << std::endl;
    std::cout << "double - int: " << typeid(gaalet::element_type_combination_traits<double, int>::element_t).name() << std::endl;

    std::cout << "int - long long: " << typeid(gaalet::element_type_combination_traits<int, long long>::element_t).name() << std::endl;
    std::cout << "long long - int: " << typeid(gaalet::element_type_combination_traits<long long, int>::element_t).name() << std::endl;
    std::cout << "float - long long: " << typeid(gaalet::element_type_combination_traits<float, long long>::element_t).name() << std::endl;
    std::cout << "long long - float: " << typeid(gaalet::element_type_combination_traits<long long, float>::element_t).name() << std::endl;
    std::cout << "double - long long: " << typeid(gaalet::element_type_combination_traits<double, long long>::element_t).name() << std::endl;
    std::cout << "long long - double: " << typeid(gaalet::element_type_combination_traits<long long, double>::element_t).name() << std::endl;

    std::cout << "int - unsigned int: " << typeid(gaalet::element_type_combination_traits<int, unsigned int>::element_t).name() << std::endl;
    std::cout << "unsigned int - int: " << typeid(gaalet::element_type_combination_traits<unsigned int, int>::element_t).name() << std::endl;
    std::cout << "int - unsigned long long: " << typeid(gaalet::element_type_combination_traits<int, unsigned long long>::element_t).name() << std::endl;
    std::cout << "unsigned long long - int: " << typeid(gaalet::element_type_combination_traits<unsigned long long, int>::element_t).name() << std::endl;
    std::cout << "float - unsigned long long: " << typeid(gaalet::element_type_combination_traits<float, unsigned long long>::element_t).name() << std::endl;
    std::cout << "unsigned long long - float: " << typeid(gaalet::element_type_combination_traits<unsigned long long, float>::element_t).name() << std::endl;

    typedef gaalet::algebra<gaalet::signature<4, 1> > emd;
    typedef gaalet::algebra<gaalet::signature<4, 1>, int> emi;

    emd::mv<1, 2, 4>::type a = { 1.2, 2.4, 3.3 };
    emi::mv<1, 2, 4>::type b = { 1, 2, 3 };
    emi::mv<1, 2, 4>::type c = { 4, 5, 6 };

    std::cout << "a: " << a << std::endl;
    std::cout << "b: " << b << std::endl;
    std::cout << "c: " << c << std::endl;

    std::cout << "a*a: " << a *a << std::endl;
    std::cout << "a*b: " << a *b << std::endl;
    std::cout << "b*c: " << b *c << std::endl;
};
