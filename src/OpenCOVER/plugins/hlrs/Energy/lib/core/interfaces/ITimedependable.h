#pragma once
namespace core::interface {
class ITimedependable {
 public:
  virtual ~ITimedependable() = default;
  virtual void updateTime(int timestep) = 0;
};
}  // namespace core::interface
