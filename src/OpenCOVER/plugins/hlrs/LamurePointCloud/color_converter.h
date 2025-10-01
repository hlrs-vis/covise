// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#ifndef COLOR_CONVERTER_HPP
#define COLOR_CONVERTER_HPP

#include <cmath>
#include <array>
#include <algorithm>

#if WIN32
#if defined(LAMURE_QUALITY_MEASUREMENT_LIBRARY)
#define QUALITY_MEASUREMENT_DLL __declspec(dllexport)
#else
#define QUALITY_MEASUREMENT_DLL __declspec(dllimport)
#endif
#else
#define QUALITY_MEASUREMENT_DLL
#endif

// color conversion code taken from:
// https://github.com/THEjoezack/ColorMine/tree/master/ColorMine/ColorSpaces/Conversions

namespace lamure
{
    namespace qm
    {

        struct col3
        {
            col3() : col{0.0, 0.0, 0.0} {}
            col3(double a, double b, double c) : col{a, b, c}
            {
            }

            double const operator[](int i) const { return col[i]; };
            double &operator[](int i) { return col[i]; };

            double col[3];
        };

        col3 const white_reference(95.047, 100.000, 108.883);
        double const epsilon = 0.008856;
        double const kappa = 903.3;

        class color_converter
        {
        public:
            static col3 const rgb_to_xyz(col3 const &rgbCol);
            static col3 const xyz_to_lab(col3 const &xyzCol);

            static double const calc_delta_E(col3 const &c1, col3 const &c2);

        private:
            static double const pivot_RGB(double n);
            static double const pivot_XYZ(double n);
        };

    }
}

#endif
