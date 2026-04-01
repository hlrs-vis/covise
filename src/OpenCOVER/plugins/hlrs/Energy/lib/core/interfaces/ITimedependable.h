#pragma once
namespace core::interface {
class ITimedependable {
 public:
  ITimedependable() = default;
  virtual ~ITimedependable() = default;
  ITimedependable(const ITimedependable&) = delete;
  ITimedependable(ITimedependable&&) = delete;
  ITimedependable operator=(const ITimedependable&) = delete;
  ITimedependable operator=(ITimedependable&&) = delete;
  virtual void updateTime(int timestep) = 0;
};
}  // namespace core::interface
