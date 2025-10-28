#pragma once

#include "ITimedependable.h"
#include "IUpdateable.h"

namespace core::interface {
class ISystem : public ITimedependable, public IUpdateable {
 public:
  virtual ~ISystem() = default;

  // Initialize the system
  virtual void init() = 0;

  // Enable or disable the system
  virtual void enable(bool on) = 0;

  // Check if the system is enabled
  virtual bool isEnabled() const = 0;
};
}  // namespace core::interface
