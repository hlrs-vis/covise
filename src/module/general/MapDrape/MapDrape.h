/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

        if (p_type_->getValue() == TYPE_MOVE_PLUGIN)
 * License: LGPL 2+ */


#ifndef _MAPDRAPE_H
#define _MAPDRAPE_H

#include <api/coSimpleModule.h>
using namespace covise;
#include <api/coHideParam.h>
#include <util/coviseCompat.h>
#include <proj_api.h>
#include <gdal_priv.h>


class MapDrape : public coSimpleModule
{
private:
    /// compute callback
    virtual int compute(const char *port);


	coStringParam *p_mapping_from_;
	coStringParam *p_mapping_to_;
	coFloatVectorParam *p_offset_;
    // Draping geotif
    coFileBrowserParam *p_heightfield_;

	GDALDataset  *heightDataset;
	GDALRasterBand  *heightBand;

    // ports
    coInputPort *p_geo_in_;
    coOutputPort *p_geo_out_;

	void transformCoordinates(int numCoords, float * xIn, float * yIn, float * zIn, float * xOut, float * yOut, float * zOut);

	void closeImage();

	void openImage(std::string & name);

	float getAlt(int x, int y);


protected:

	projPJ pj_from, pj_to;
	std::string dir;

public:
    /// constructor
    MapDrape(int argc, char *argv[]);
};
#endif // _MAPDRAPE_H
