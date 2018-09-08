#pragma once


#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif


#if VV_HAVE_TEEM


class vvVolDesc;


namespace virvo
{

namespace nrrd
{

void load(vvVolDesc* vd);

} // nrrd

} // virvo


#endif // VV_HAVE_TEEM


