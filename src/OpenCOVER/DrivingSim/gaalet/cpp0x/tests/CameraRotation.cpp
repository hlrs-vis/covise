/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

typedef gaalet::algebra<gaalet::signature<3, 0> > em;

int main()
{
    //Pseudoscalar for G(3,0):
    em::mv<7>::type I = { 1.0 };
    //Position vector declaration of camera and look-at point:
    em::mv<1, 2, 4>::type L = { 0.0, 0.0, 0.5 }, P = { 5.0, 0.0, 1.0 };
    //Camera rotation angle:
    double phi = 0.5 * M_PI;

    //View direction vector expression defined and evaluated:
    em::mv<1, 2, 4>::type t = L - P;
    std::cout << !magnitude(t) << std::endl;
    //Expression defined, not evaluated:
    auto S_t = t * !magnitude(t) * I;
    //Expression defined, including another expression:
    auto R_t = exp(-0.5 * phi * S_t);

    //Rotating vector definition:
    em::mv<1, 2, 4>::type a = { 0.0, 0.0, 1.0 };
    //Expression defined and evaluated into multivector:
    auto b = eval(grade<1>(R_t * a * (~R_t)));

    std::cout << "a: " << a << ", b: " << b << std::endl;
}
