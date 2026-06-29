#pragma once

namespace prototype::core::simulation {

enum class ObjectType {
  Bus,
  Generator,
  Transformator,
  Cable,
  Building,
  Consumer,
  Producer,
  // Add more types as needed
  Unknown
};

}
