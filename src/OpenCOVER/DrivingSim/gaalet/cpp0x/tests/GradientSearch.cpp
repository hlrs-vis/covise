/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "gaalet.h"

typedef gaalet::algebra<gaalet::signature<3, 0> > em;

int main()
{
    em::mv<1, 2, 4>::type t_0 = { 1.0, 0.0, 0.0 };

    em::mv<1, 2, 4>::type t_t = { 1.0, 2.0, 1.0 };

    em::mv<3, 5, 6>::type m = { 0.0, 0.0, 0.0 };
    em::mv<3, 5, 6>::type m_old = { 1.0, 0.0, 0.0 };

    auto R = exp(-0.5 * m_old);
    em::mv<3, 5, 6>::type gradU_old = 1.0 / 8.0 * (R * t_0 * ~R * t_t - t_t * R * t_0 * ~R);

    double maxerr = 1e-8;
    double mag_g = 1.0;

    while (mag_g >= maxerr)
    {
        auto R = exp(-0.5 * m);
        auto gradU = 1.0 / 8.0 * (R * t_0 * ~R * t_t - t_t * R * t_0 * ~R);

        auto D_m = eval(m_old - m);
        m_old = m;
        auto D_gradU = eval(gradU_old - gradU);
        gradU_old = gradU;
        mag_g = eval(magnitude(D_gradU & D_gradU));
        //BB Two point step size
        //double alpha = eval((D_m&D_gradU)*!(D_gradU&D_gradU));
        double alpha = eval((D_m & D_m) * !(D_m & D_gradU));

        m = m - alpha * gradU;

        std::cout << "R: " << R << ", t: " << grade<1>(R * t_0 * ~R) << ", t_t: " << t_t << ", alpha: " << alpha << ", mag_g: " << mag_g << std::endl;
    }
}
