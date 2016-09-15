#ifndef PRT_POINT_READER_H
#define PRT_POINT_READER_H

#include <visionaray/bvh.h>
#include <visionaray/math/math.h>
#include <visionaray/aligned_vector.h>

#include "PointRayTracerGlobals.h"

//using sphere_type   = visionaray::basic_sphere<float>;
using color_type    = visionaray::vector<3, visionaray::unorm<8>>;

using point_vector  = visionaray::aligned_vector<sphere_type, 32>;
using color_vector  = visionaray::aligned_vector<color_type, 32>;

class PointReader {
public:

    static PointReader *instance();

    bool readFile(std::string filename,
                  float pointSize,
                  std::vector<host_bvh_type> &bvh_vector,
                  visionaray::aabb& bbox,
                  bool useCache,
                  bool cutUTMdata = false);

private:    
    PointReader();

    bool loadBvh(std::string filename, host_bvh_type &bvh);
    bool storeBvh(std::string filename, host_bvh_type &bvh);
};


#endif // PRT_POINT_READER_H
