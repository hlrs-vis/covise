#ifndef MEASURE_MATERIALS_H
#define MEASURE_MATERIALS_H

#include <osg/Material>
#include <util/coExport.h>

namespace opencover{

namespace material{

enum Color
{
    White, Red, Green, Blue, LAST
};

PLUGIN_UTILEXPORT osg::Material *get(Color c);

}
}

#endif // MEASURE_MATERIALS_H