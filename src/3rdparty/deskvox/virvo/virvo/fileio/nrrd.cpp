#include "exceptions.h"
#include "nrrd.h"

#include <virvo/vvvoldesc.h>
#include <virvo/private/vvlog.h>

#if VV_HAVE_TEEM
#include <teem/nrrd.h>
#endif

#include <string>
#include <string.h> // memcpy


#if VV_HAVE_TEEM

using namespace virvo;


struct ScopedNrrd
{

  ScopedNrrd()
    : ptr(nrrdNew())
    , io(nrrdIoStateNew())
  {
  }

  ~ScopedNrrd()
  {
    nrrdNuke(ptr);
    nrrdIoStateNix(io);
  }

  Nrrd* ptr;
  NrrdIoState* io;

};


void nrrd::load(vvVolDesc* vd)
{

  ScopedNrrd nrrd;

  // load decodes data and automatically toggles
  // endianness according to endian tag in file
  if (nrrdLoad(nrrd.ptr, vd->getFilename(), nrrd.io))
  {
    char* err = biffGetDone(NRRD);
    std::string errstr = err;
    free(err);
    throw fileio::exception(errstr);
  }

  if (nrrd.ptr->dim != 3)
  {
    throw fileio::wrong_dimensions();
  }

  vd->frames = 1;
  switch (nrrd.ptr->type)
  {
  case nrrdTypeChar:
  case nrrdTypeUChar:
    vd->bpc  = 1;
    vd->setChan(1);
    break;
  case nrrdTypeShort:
  case nrrdTypeUShort:
    vd->bpc  = 2;
    vd->setChan(1);
    break;
  case nrrdTypeInt:
  case nrrdTypeUInt:
    vd->bpc  = 4;
    vd->setChan(1);
    break;
  default:
    throw fileio::unsupported_datatype();
  }

  size_t size[3];
  nrrdAxisInfoGet_nva(nrrd.ptr, nrrdAxisInfoSize, size);

  vd->vox[0] = size[0];
  vd->vox[1] = size[1];
  vd->vox[2] = size[2];

  uint8_t* raw = new uint8_t[vd->getFrameBytes()];
  memcpy(raw, nrrd.ptr->data, vd->getFrameBytes());
  vd->addFrame(raw, vvVolDesc::ARRAY_DELETE);

}


#endif // VV_HAVE_TEEM


