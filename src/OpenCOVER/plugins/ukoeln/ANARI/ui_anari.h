
#pragma once

// cover
#include <cover/ui/Menu.h>
// helium
#include "helium/utility/AnariAny.h"
// anari
#include "anari/anari_cpp.hpp"
// std
#include <string>
#include <vector>

class Renderer;

namespace ui_anari {

using Any = helium::AnariAny;

struct ParameterInfo
{
  std::string name;
  Any value;
  Any min;
  Any max;
  std::string description;

  // valid values if this parameter is ANARI_STRING_LIST
  std::vector<std::string> stringValues;
  // which string is selected in 'stringValues', if applicable
  int currentSelection{0};
};

using Parameter = ParameterInfo;
using ParameterList = std::vector<Parameter>;

ParameterList parseParameters(
    anari::Device d, ANARIDataType objectType, const char *subtype);

void buildUI(std::shared_ptr<Renderer> renderer, Parameter &p, opencover::ui::Menu *menu);

} // namespace ui_anari
