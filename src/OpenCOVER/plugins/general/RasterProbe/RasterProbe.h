/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * Author: Siva Karthikeya Mandarapu
 * License: LGPL 2+ */

#ifndef RASTER_PROBE_H
#define RASTER_PROBE_H

#include <vector>

#include <cover/coVRPlugin.h>
#include <cover/ui/Owner.h>

namespace opencover
{
namespace ui
{
class Button;
}
}

class VrmlNodeRasterProbe;

// Samples the RasterProbe enabled last under the pointer, every frame.
class RasterProbe : public opencover::coVRPlugin, public opencover::ui::Owner
{
public:
    RasterProbe();

    bool init() override;
    void preFrame() override;

    static void addProbe(VrmlNodeRasterProbe *probe);
    static void removeProbe(VrmlNodeRasterProbe *probe);

private:
    opencover::ui::Button *m_probeButton = nullptr;
    VrmlNodeRasterProbe *m_lastProbe = nullptr;

    static std::vector<VrmlNodeRasterProbe *> s_probes;
};
#endif
