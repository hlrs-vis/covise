#pragma once
namespace core::interface {
class ITimedependable {
 public:
  virtual ~ITimedependable() = default;
  ITimedependable(const ITimedependable&) = delete;
  ITimedependable operator=(const ITimedependable&) = delete;
  virtual void updateTime(int timestep) = 0;
};
}  // namespace core::interface
