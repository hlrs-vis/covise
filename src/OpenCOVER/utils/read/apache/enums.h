#pragma once
#include "export.h"

namespace apache {
enum class ARROWUTIL IO {
  MEM_MAP,  // Memory-mapped file os specific => zero-copy => better for random
            // access
  FILE      // Readable file => better for sequential access / streaming
};
}
