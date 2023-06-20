#ifndef MEASURE_MATERIALS_H
#define MEASURE_MATERIALS_H

#include <osg/Material>
namespace material{

enum Color
{
    White, Red, Green, Blue, LAST
};

osg::Material *get(Color c);

}

#endif // MEASURE_MATERIALS_H