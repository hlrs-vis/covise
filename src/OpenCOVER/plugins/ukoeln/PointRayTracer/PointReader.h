#ifndef PRT_POINT_READER_H
#define PRT_POINT_READER_H

#include <visionaray/bvh.h>
#include <visionaray/math/math.h>
#include <visionaray/aligned_vector.h>

// TODO: consolidate with PointRayTracerGlobals.h (?)
using sphere_type   = visionaray::basic_sphere<float>;
using color_type    = visionaray::vector<3, visionaray::unorm<8>>;

using point_vector  = visionaray::aligned_vector<sphere_type, 32>;
using color_vector  = visionaray::aligned_vector<color_type, 32>;

class PointReader {
public:
    PointReader();

    bool readFile(std::string filename,
                  float pointSize,
                  point_vector& points,
                  color_vector& colors,
                  visionaray::aabb& bbox,
                  bool cutUTMdata = false);

private:

    /*
    char* m_filename;

    visionaray::aligned_vector<sphere_type>                     m_points;
    visionaray::aligned_vector<visionaray::vector<3, visionaray::unorm<8>>, 32>         m_colors;
    */
};


#endif // PRT_POINT_READER_H
