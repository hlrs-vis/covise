/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "WSTools.h"

#include "WSIntVectorParameter.h"
#include "WSIntSliderParameter.h"
#include "WSIntScalarParameter.h"
#include "WSFloatVectorParameter.h"
#include "WSFloatSliderParameter.h"
#include "WSFloatScalarParameter.h"
#include "WSFileBrowserParameter.h"
#include "WSChoiceParameter.h"
#include "WSColormapChoiceParameter.h"
#include "WSBooleanParameter.h"
#include "WSStringParameter.h"
#include "WSLink.h"
#include "WSColormap.h"

#include <cassert>

QString covise::WSTools::fromCovise(const QString &from)
{
#ifndef YAC
    if (from.size() == 1 && from[0] == '\001')
        return "";
    QString newString = from;
    return newString.replace(QChar('\177'), ' ');
#else
    return from;
#endif
}

QString covise::WSTools::toCovise(const QString &from)
{
#ifndef YAC
    if (from == "")
        return QChar('\001');
    QString newString = from;
    return newString.replace(' ', QChar('\177'));
#else
    return from;
#endif
}

bool covise::WSTools::setParameterFromString(covise::WSParameter *parameter, const QString &value)
{
    if (parameter == 0)
    {
        std::cerr << "WSTools::setParameterFromString err: called with parameter = 0" << std::endl;
        return false;
    }

    bool changed = false;

    if (parameter->getType() == "Boolean")
    {
        changed = dynamic_cast<WSBooleanParameter *>(parameter)->setValue(value.toLower() != "false");
    }
    else if (parameter->getType() == "Choice")
    {
        QStringList list = value.split(' ', QString::SkipEmptyParts);
        int selection = list.takeFirst().toInt() - 1;
        if (list.size() > 1)
            changed = dynamic_cast<WSChoiceParameter *>(parameter)->setValue(list, selection);
        else
            changed = dynamic_cast<WSChoiceParameter *>(parameter)->setValue(selection);
    }
    else if (parameter->getType() == "ColormapChoice")
    {
        QStringList list = value.split(' ', QString::SkipEmptyParts);
        int selection = list.takeFirst().toInt() - 1;
        if (list.size() > 1)
        {
            int numberOfMaps = list.takeFirst().toInt();
            QList<covise::WSColormap> colormaps;
            for (int currentMap = 0; currentMap < numberOfMaps; ++currentMap)
            {
                QList<covise::WSColormapPin> pins;
                QString name = list.takeFirst();
                int numberOfPins = list.takeFirst().toInt();

                assert(numberOfPins * 5 <= list.size());

                for (int p = 0; p < numberOfPins; ++p)
                {
                    covise::WSColormapPin pin;
                    pin.r = list.takeFirst().toFloat();
                    pin.g = list.takeFirst().toFloat();
                    pin.b = list.takeFirst().toFloat();
                    pin.a = list.takeFirst().toFloat();
                    pin.position = list.takeFirst().toFloat();
                    pins.push_back(pin);
                }
                colormaps.push_back(covise::WSColormap(name, pins));
            }
            changed = dynamic_cast<WSColormapChoiceParameter *>(parameter)->setValue(colormaps, selection);
        }
        else
        {
            changed = dynamic_cast<WSColormapChoiceParameter *>(parameter)->setValue(selection);
        }
    }
    else if (parameter->getType() == "FileBrowser")
    {
        changed = dynamic_cast<WSFileBrowserParameter *>(parameter)->setValue(value);
    }
    else if (parameter->getType() == "IntScalar")
    {
        changed = dynamic_cast<WSIntScalarParameter *>(parameter)->setValue(value.toInt());
    }
    else if (parameter->getType() == "IntSlider")
    {
        QStringList vList = value.split(' ', QString::SkipEmptyParts);
        if (vList.size() > 1)
        {
            changed = dynamic_cast<WSIntSliderParameter *>(parameter)->setValue(vList[2].toInt(),
                                                                                vList[0].toInt(),
                                                                                vList[1].toInt());
        }
        else
        {
            changed = dynamic_cast<WSIntSliderParameter *>(parameter)->setValue(vList[0].toInt());
        }
    }
    else if (parameter->getType() == "IntVector")
    {
        QStringList vList = value.split(' ', QString::SkipEmptyParts);
        QVector<int> values;
        foreach (QString v, vList)
            values.push_back(v.toInt());
        changed = dynamic_cast<WSIntVectorParameter *>(parameter)->setValue(values);
    }
    else if (parameter->getType() == "FloatScalar")
    {
        changed = dynamic_cast<WSFloatScalarParameter *>(parameter)->setValue(value.toFloat());
    }
    else if (parameter->getType() == "FloatSlider")
    {
        QStringList vList = value.split(' ', QString::SkipEmptyParts);
        if (vList.size() > 1)
        {
            changed = dynamic_cast<WSFloatSliderParameter *>(parameter)->setValue(vList[2].toFloat(),
                                                                                  vList[0].toFloat(),
                                                                                  vList[1].toFloat());
        }
        else
        {
            changed = dynamic_cast<WSFloatSliderParameter *>(parameter)->setValue(vList[0].toFloat());
        }
    }
    else if (parameter->getType() == "FloatVector")
    {
        QStringList vList = value.split(' ', QString::SkipEmptyParts);
        QVector<float> values;
        foreach (QString v, vList)
            values.push_back(v.toFloat());
        changed = dynamic_cast<WSFloatVectorParameter *>(parameter)->setValue(values);
    }
    else if (parameter->getType() == "String")
    {
        changed = dynamic_cast<WSStringParameter *>(parameter)->setValue(value);
    }
    else
    {
        std::cerr << "WSMessageHandler::setParameterFromString err: unsupported parameter type " << qPrintable(parameter->getType()) << std::endl;
        return false;
    }

    return changed;
}
