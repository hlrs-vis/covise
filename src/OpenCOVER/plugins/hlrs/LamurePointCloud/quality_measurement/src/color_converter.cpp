// Copyright (c) 2014-2018 Bauhaus-Universitaet Weimar
// This Software is distributed under the Modified BSD License, see license.txt.
//
// Virtual Reality and Visualization Research Group 
// Faculty of Media, Bauhaus-Universitaet Weimar
// http://www.uni-weimar.de/medien/vr

#include <lamure/qm/color_converter.h>

namespace lamure {
namespace qm {

double const color_converter::pivot_RGB(double n)
{
	return (n > 0.04045 ? std::pow((n + 0.055) / 1.055, 2.4) : n / 12.92) * 100.0;
}

col3 const color_converter::rgb_to_xyz(col3 const& rgbCol)
{

	double r = pivot_RGB(rgbCol[0] / 255.0f);
	double g = pivot_RGB(rgbCol[1] / 255.0f);
	double b = pivot_RGB(rgbCol[2] / 255.0f);

	col3 xyz_col;

	xyz_col[0] = r * 0.4124 + g * 0.3576 + b * 0.1805;
	xyz_col[1] = r * 0.2126 + g * 0.7152 + b * 0.0722;
	xyz_col[2] = r * 0.0193 + g * 0.1192 + b * 0.9505;
	
	return xyz_col;	
}

double const color_converter::pivot_XYZ(double n)
{
        return n > epsilon ? std::pow(n, 1.0/3.0) : (kappa * n + 16) / 116;
}

col3 const color_converter::xyz_to_lab(col3 const& xyzCol)
{
	double x = pivot_XYZ(xyzCol[0] / white_reference[0]);
	double y = pivot_XYZ(xyzCol[1] / white_reference[1]);
	double z = pivot_XYZ(xyzCol[2] / white_reference[2]);

	col3 lab_col;
	
	lab_col[0] = std::max(0.0, 116*y -16);
	lab_col[1] = 500 * (x - y);
	lab_col[2] = 200 * (y - z);

	return lab_col;
}


double const color_converter::calc_delta_E(col3 const& c1, col3 const& c2)
{
	return std::sqrt(
				std::pow(c1[0]-c2[0],2) + 
				std::pow(c1[1]-c2[1],2) +
				std::pow(c1[2]-c2[2],2) 
			 );	
}

}
}