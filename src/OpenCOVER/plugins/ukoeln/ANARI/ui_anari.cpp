
// cover
#include <cover/ui/Button.h>
#include <cover/ui/EditField.h>
#include <cover/ui/Slider.h>
// ours
#include "Renderer.h"
#include "ui_anari.h"
// std
#include <iostream>

namespace ui = opencover::ui;

namespace ui_anari {

// Helper functions ///////////////////////////////////////////////////////////

static ui_anari::Any parseValue(ANARIDataType type, const void *mem)
{
  if (type == ANARI_STRING)
    return ui_anari::Any(ANARI_STRING, "");
  else if (anari::isObject(type)) {
    ANARIObject nullHandle = ANARI_INVALID_HANDLE;
    return ui_anari::Any(type, &nullHandle);
  } else if (mem)
    return ui_anari::Any(type, mem);
  else
    return {};
}

static bool UI_stringList_callback(
    void *_stringList, int index, const char **out_text)
{
  auto &stringList = *(std::vector<std::string> *)_stringList;
  *out_text = stringList[index].c_str();
  return true;
}

ParameterList parseParameters(
    anari::Device d, ANARIDataType objectType, const char *subtype)
{
  ParameterList retval;

  auto *parameter = (const ANARIParameter *)anariGetObjectInfo(
      d, objectType, subtype, "parameter", ANARI_PARAMETER_LIST);

  for (; parameter && parameter->name != nullptr; parameter++) {
    Parameter p;

    auto *description = (const char *)anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "description",
        ANARI_STRING);

    const void *defaultValue = anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "default",
        parameter->type);

    const void *minValue = anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "minimum",
        parameter->type);

    const void *maxValue = anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "maximum",
        parameter->type);

    const auto **stringValues = (const char **)anariGetParameterInfo(d,
        objectType,
        subtype,
        parameter->name,
        parameter->type,
        "value",
        ANARI_STRING_LIST);

    p.name = parameter->name;
    p.description = description ? description : "";
    p.value = parseValue(parameter->type, defaultValue);

    if (minValue)
      p.min = parseValue(parameter->type, minValue);
    if (maxValue)
      p.max = parseValue(parameter->type, maxValue);

    for (; stringValues && *stringValues; stringValues++)
      p.stringValues.push_back(*stringValues);

    retval.push_back(p);
  }

  return retval;
}

ui::Element *buildUI(Renderer::SP renderer, Parameter &p, ui::Menu *menu)
{
  ANARIDataType type = p.value.type();
  const char *name = p.name.c_str();
  void *value = p.value.data();

  const bool bounded = p.min || p.max;
  const bool showTooltip = !p.description.empty();

  ui::Element *result{nullptr};

  switch (type) {
  case ANARI_BOOL: {
      auto *button = new ui::Button(menu, name);
      button->setState(*(bool *)value);
      button->setCallback([=](bool value) {
          renderer->setParameter(p.name, value);
      });
      result=button;
    }
    break;
  case ANARI_INT32: {
      if (bounded) {
        auto *slider = new ui::Slider(menu, name);
        slider->setIntegral(true);
        if (p.min && p.max) {
          slider->setBounds(p.min.get<int>(), p.max.get<int>());
        } else {
          int min =
              p.min ? p.min.get<int>() : std::numeric_limits<int>::lowest();
          int max =
              p.max ? p.max.get<int>() : std::numeric_limits<int>::max();
          slider->setBounds(min, max);
        }
        slider->setValue(p.value.get<int>());
        slider->setCallback([=](int value, bool /*release*/) {
            renderer->setParameter(p.name, value);
        });
        result=slider;
      } else {
        auto *edit = new ui::EditField(menu, name);
        edit->setValue(p.value.get<int>());
        edit->setCallback([=](std::string value) {
            renderer->setParameter(p.name, std::stoi(value));
        });
        result=edit;
      }
    }
    break;
  case ANARI_FLOAT32: {
      if (bounded) {
        auto *slider = new ui::Slider(menu, name);
        slider->setIntegral(false);
        if (p.min && p.max) {
          slider->setBounds(p.min.get<float>(), p.max.get<float>());
        } else {
          float min =
              p.min ? p.min.get<float>() : std::numeric_limits<float>::lowest();
          float max =
              p.max ? p.max.get<float>() : std::numeric_limits<float>::max();
          slider->setBounds(min, max);
        }
        slider->setValue(p.value.get<float>());
        slider->setCallback([=](double value, bool /*release*/) {
            renderer->setParameter(p.name, (float)value);
        });
        result=slider;
      } else {
        auto *edit = new ui::EditField(menu, name);
        edit->setValue(p.value.get<float>());
        edit->setCallback([=](std::string value) {
            renderer->setParameter(p.name, std::stof(value));
        });
        result=edit;
      }
    }
    break;
  default: {
      std::cerr << "ui_anari::buildUI -- not implemented yet for (type:) "
        << anari::toString(type) << " (name:) "
        << name << '\n';
    }
    break;
  }

  return result;
}

} // namespace ui_anari
