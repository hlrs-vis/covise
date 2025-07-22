#pragma once
namespace apache {
enum class IO {
  MEM_MAP,  // Memory-mapped file os specific => zero-copy => better for random
            // access
  FILE      // Readable file => better for sequential access / streaming
};
}
