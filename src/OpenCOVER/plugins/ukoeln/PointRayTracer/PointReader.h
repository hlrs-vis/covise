#ifndef PRT_POINT_READER_H
#define PRT_POINT_READER_H

#include <visionaray/bvh.h>
#include <visionaray/math/math.h>
#include <visionaray/aligned_vector.h>

using sphere_type   = visionaray::basic_sphere<float>;

class PointReader {
public:
    PointReader();

    bool readFile(std::string filename,
                  float pointSize,
                  visionaray::aligned_vector<sphere_type>& points,
                  visionaray::aligned_vector<visionaray::vector<3, visionaray::unorm<8>>, 32>& colors,
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
