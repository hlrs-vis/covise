#pragma once

#include <memory>
#include <stdexcept>
#include "object.h"
#include "object_type.h"

namespace core::simulation {

inline std::unique_ptr<Object> createObject(ObjectType type, const std::string& name,
                                            const Data& data) {
  switch (type) {
    // for custom logic inherit from Object and do something with it here like this
    // case ObjectType::Bus:
    //   return std::make_unique<power::Bus>(name, data);
    case ObjectType::Bus:
    case ObjectType::Generator:
    case ObjectType::Building:
    case ObjectType::Cable:
    case ObjectType::Transformator:
    case ObjectType::Consumer:
    case ObjectType::Producer:
      return std::make_unique<Object>(name, data);
    default:
      throw std::runtime_error("Unknown ObjectType: " + std::to_string(static_cast<int>(type)));
  }
}
}  // namespace core::simulation
