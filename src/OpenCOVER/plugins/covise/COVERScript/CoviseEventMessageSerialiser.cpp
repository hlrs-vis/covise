/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "CoviseEventMessageSerialiser.h"

#include <wslib/WSCoviseStub.h>

using namespace covise;

QDomElement CoviseEventMessageSerialiser::serialise(const covise__Event *event, QDomDocument &doc)
{
    if (event->type == "LinkAdd")
        return serialise(dynamic_cast<const covise::covise__LinkAddEvent *>(event), doc);
    else if (event->type == "ModuleAdd")
        return serialise(dynamic_cast<const covise::covise__ModuleAddEvent *>(event), doc);
    else if (event->type == "ModuleDel")
        return serialise(dynamic_cast<const covise::covise__ModuleDelEvent *>(event), doc);
    else if (event->type == "ModuleDied")
        return serialise(dynamic_cast<const covise::covise__ModuleDiedEvent *>(event), doc);
    else if (event->type == "ParameterChange")
        return serialise(dynamic_cast<const covise::covise__ParameterChangeEvent *>(event), doc);
    else if (event->type == "OpenNet")
        return serialise(dynamic_cast<const covise::covise__OpenNetEvent *>(event), doc);
    else if (event->type == "OpenNetDone")
        return serialise(dynamic_cast<const covise::covise__OpenNetDoneEvent *>(event), doc);
    else if (event->type == "Quit")
        return serialise(dynamic_cast<const covise::covise__QuitEvent *>(event), doc);
    else
        return QDomElement();
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__Port *p, QDomDocument &doc)
{
    QDomElement port = doc.createElement("port");

    QDomElement name = doc.createElement("name");
    port.appendChild(name);
    name.appendChild(doc.createTextNode(QString::fromStdString(p->name)));

    QDomElement portType = doc.createElement("portType");
    port.appendChild(portType);
    portType.appendChild(doc.createTextNode(QString::fromStdString(p->portType)));

    QDomElement id = doc.createElement("id");
    port.appendChild(id);
    id.appendChild(doc.createTextNode(QString::fromStdString(p->id)));

    QDomElement moduleID = doc.createElement("moduleID");
    port.appendChild(moduleID);
    moduleID.appendChild(doc.createTextNode(QString::fromStdString(p->moduleID)));

    QDomElement types = doc.createElement("types");
    port.appendChild(types);
    for (std::vector<std::string>::const_iterator t = p->types.begin(); t != p->types.end(); ++t)
    {
        QDomElement type = doc.createElement("type");
        type.appendChild(doc.createTextNode(QString::fromStdString(*t)));
        types.appendChild(type);
    }

    return port;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__Link *l, QDomDocument &doc)
{

    QDomElement link = doc.createElement("link");

    QDomElement id = doc.createElement("id");
    link.appendChild(id);
    id.appendChild(doc.createTextNode(QString::fromStdString(l->id)));

    QDomElement from = doc.createElement("from");
    link.appendChild(from);
    from.appendChild(serialise(&(l->from), doc));

    QDomElement to = doc.createElement("to");
    link.appendChild(to);
    to.appendChild(serialise(&(l->to), doc));

    return link;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__Parameter *const param, QDomDocument &doc)
{

    QDomElement parameter = doc.createElement("parameter");

    QDomElement name = doc.createElement("name");
    parameter.appendChild(name);
    name.appendChild(doc.createTextNode(QString::fromStdString(param->name)));

    QDomElement type = doc.createElement("type");
    parameter.appendChild(type);
    type.appendChild(doc.createTextNode(QString::fromStdString(param->type)));

    QDomElement description = doc.createElement("description");
    parameter.appendChild(description);
    description.appendChild(doc.createTextNode(QString::fromStdString(param->description)));

    QDomElement mapped = doc.createElement("mapped");
    parameter.appendChild(mapped);
    mapped.appendChild(doc.createTextNode(param->mapped ? "true" : "false"));

    if (dynamic_cast<const covise__BooleanParameter *>(param) != 0)
    {
        const covise__BooleanParameter *p = dynamic_cast<const covise__BooleanParameter *>(param);

        QDomElement value = doc.createElement("value");
        parameter.appendChild(value);
        value.appendChild(doc.createTextNode(p->value ? "true" : "false"));
    }
    else if (dynamic_cast<const covise__ChoiceParameter *>(param) != 0)
    {
        const covise__ChoiceParameter *p = dynamic_cast<const covise__ChoiceParameter *>(param);

        QDomElement selected = doc.createElement("selected");
        parameter.appendChild(selected);
        selected.appendChild(doc.createTextNode(QString::number(p->selected)));

        QDomElement choices = doc.createElement("choices");
        parameter.appendChild(choices);
        for (std::vector<std::string>::const_iterator c = p->choices.begin(); c != p->choices.end(); ++c)
        {
            QDomElement choice = doc.createElement("choice");
            choices.appendChild(choice);
            choice.appendChild(doc.createTextNode(QString::fromStdString(*c)));
        }
    }
    else if (dynamic_cast<const covise__FileBrowserParameter *>(param) != 0)
    {
        const covise__FileBrowserParameter *p = dynamic_cast<const covise__FileBrowserParameter *>(param);

        QDomElement value = doc.createElement("value");
        parameter.appendChild(value);
        value.appendChild(doc.createTextNode(QString::fromStdString(p->value)));
    }
    else if (dynamic_cast<const covise__FloatScalarParameter *>(param) != 0)
    {
        const covise__FloatScalarParameter *p = dynamic_cast<const covise__FloatScalarParameter *>(param);

        QDomElement value = doc.createElement("value");
        parameter.appendChild(value);
        value.appendChild(doc.createTextNode(QString::number(p->value)));
    }
    else if (dynamic_cast<const covise__FloatSliderParameter *>(param) != 0)
    {
        const covise__FloatSliderParameter *p = dynamic_cast<const covise__FloatSliderParameter *>(param);

        QDomElement value = doc.createElement("value");
        parameter.appendChild(value);
        value.appendChild(doc.createTextNode(QString::number(p->value)));

        QDomElement min = doc.createElement("min");
        parameter.appendChild(min);
        min.appendChild(doc.createTextNode(QString::number(p->min)));

        QDomElement max = doc.createElement("max");
        parameter.appendChild(max);
        max.appendChild(doc.createTextNode(QString::number(p->max)));
    }
    else if (dynamic_cast<const covise__FloatVectorParameter *>(param) != 0)
    {
        const covise__FloatVectorParameter *p = dynamic_cast<const covise__FloatVectorParameter *>(param);

        QDomElement values = doc.createElement("values");
        parameter.appendChild(values);

        for (std::vector<float>::const_iterator v = p->value.begin(); v != p->value.end(); ++v)
        {
            QDomElement value = doc.createElement("value");
            values.appendChild(value);
            value.appendChild(doc.createTextNode(QString::number(*v)));
        }
    }
    else if (dynamic_cast<const covise__IntScalarParameter *>(param) != 0)
    {
        const covise__IntScalarParameter *p = dynamic_cast<const covise__IntScalarParameter *>(param);

        QDomElement value = doc.createElement("value");
        parameter.appendChild(value);
        value.appendChild(doc.createTextNode(QString::number(p->value)));
    }
    else if (dynamic_cast<const covise__IntSliderParameter *>(param) != 0)
    {
        const covise__IntSliderParameter *p = dynamic_cast<const covise__IntSliderParameter *>(param);

        QDomElement value = doc.createElement("value");
        parameter.appendChild(value);
        value.appendChild(doc.createTextNode(QString::number(p->value)));

        QDomElement min = doc.createElement("min");
        parameter.appendChild(min);
        min.appendChild(doc.createTextNode(QString::number(p->min)));

        QDomElement max = doc.createElement("max");
        parameter.appendChild(max);
        max.appendChild(doc.createTextNode(QString::number(p->max)));
    }
    else if (dynamic_cast<const covise__IntVectorParameter *>(param) != 0)
    {
        const covise__IntVectorParameter *p = dynamic_cast<const covise__IntVectorParameter *>(param);

        QDomElement values = doc.createElement("values");
        parameter.appendChild(values);

        for (std::vector<int>::const_iterator v = p->value.begin(); v != p->value.end(); ++v)
        {
            QDomElement value = doc.createElement("value");
            values.appendChild(value);
            value.appendChild(doc.createTextNode(QString::number(*v)));
        }
    }
    else if (dynamic_cast<const covise__StringParameter *>(param) != 0)
    {
        const covise__StringParameter *p = dynamic_cast<const covise__StringParameter *>(param);

        QDomElement value = doc.createElement("value");
        parameter.appendChild(value);
        value.appendChild(doc.createTextNode(QString::fromStdString(p->value)));
    }
    else
    {
        return QDomElement();
    }

    return parameter;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__Module *mod, QDomDocument &doc)
{

    QDomElement module = doc.createElement("module");

    QDomElement name = doc.createElement("name");
    module.appendChild(name);
    name.appendChild(doc.createTextNode(QString::fromStdString(mod->name)));

    QDomElement category = doc.createElement("category");
    module.appendChild(category);
    category.appendChild(doc.createTextNode(QString::fromStdString(mod->category)));

    QDomElement host = doc.createElement("host");
    module.appendChild(host);
    host.appendChild(doc.createTextNode(QString::fromStdString(mod->host)));

    QDomElement description = doc.createElement("description");
    module.appendChild(description);
    description.appendChild(doc.createTextNode(QString::fromStdString(mod->description)));

    QDomElement instance = doc.createElement("instance");
    module.appendChild(instance);
    instance.appendChild(doc.createTextNode(QString::fromStdString(mod->instance)));

    QDomElement id = doc.createElement("id");
    module.appendChild(id);
    id.appendChild(doc.createTextNode(QString::fromStdString(mod->id)));

    QDomElement title = doc.createElement("title");
    module.appendChild(title);
    title.appendChild(doc.createTextNode(QString::fromStdString(mod->title)));

    QDomElement position = doc.createElement("position");
    module.appendChild(position);
    position.appendChild(serialise(&(mod->position), doc));

    QDomElement parameters = doc.createElement("parameters");
    module.appendChild(parameters);

    for (std::vector<covise__Parameter *>::const_iterator p = mod->parameters.begin();
         p != mod->parameters.end(); ++p)
    {
        parameters.appendChild(serialise(*p, doc));
    }

    QDomElement inputPorts = doc.createElement("inputPorts");
    module.appendChild(inputPorts);

    for (std::vector<covise__Port>::const_iterator p = mod->inputPorts.begin();
         p != mod->inputPorts.end(); ++p)
    {
        inputPorts.appendChild(serialise(&(*p), doc));
    }

    QDomElement outputPorts = doc.createElement("outputPorts");
    module.appendChild(outputPorts);

    for (std::vector<covise__Port>::const_iterator p = mod->outputPorts.begin();
         p != mod->outputPorts.end(); ++p)
    {
        outputPorts.appendChild(serialise(&(*p), doc));
    }

    return module;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__Point *p, QDomDocument &doc)
{
    QDomElement point = doc.createElement("point");

    QDomElement x = doc.createElement("x");
    point.appendChild(x);
    x.appendChild(doc.createTextNode(QString::number(p->x)));

    QDomElement y = doc.createElement("y");
    point.appendChild(y);
    y.appendChild(doc.createTextNode(QString::number(p->y)));

    return point;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__LinkAddEvent *e, QDomDocument &doc)
{
    QDomElement event = doc.createElement("coviseEvent");

    event.setAttribute("type", "LinkAdd");
    event.appendChild(serialise(&(e->link), doc));

    return event;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__ModuleAddEvent *e, QDomDocument &doc)
{
    QDomElement event = doc.createElement("coviseEvent");

    event.setAttribute("type", "ModuleAdd");
    event.appendChild(serialise(&(e->module), doc));

    return event;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__ModuleDelEvent *e, QDomDocument &doc)
{
    QDomElement event = doc.createElement("coviseEvent");

    event.setAttribute("type", "ModuleDel");
    QDomElement moduleID = doc.createElement("moduleID");
    event.appendChild(moduleID);
    moduleID.appendChild(doc.createTextNode(QString::fromStdString(e->moduleID)));

    return event;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__ModuleDiedEvent *e, QDomDocument &doc)
{
    QDomElement event = doc.createElement("coviseEvent");

    event.setAttribute("type", "ModuleDied");
    QDomElement moduleID = doc.createElement("moduleID");
    event.appendChild(moduleID);
    moduleID.appendChild(doc.createTextNode(QString::fromStdString(e->moduleID)));

    return event;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__ParameterChangeEvent *e, QDomDocument &doc)
{
    QDomElement event = doc.createElement("coviseEvent");

    event.setAttribute("type", "ParameterChange");

    QDomElement moduleID = doc.createElement("moduleID");
    event.appendChild(moduleID);
    moduleID.appendChild(doc.createTextNode(QString::fromStdString(e->moduleID)));

    event.appendChild(serialise(e->parameter, doc));

    return event;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__OpenNetEvent *e, QDomDocument &doc)
{
    QDomElement event = doc.createElement("coviseEvent");

    event.setAttribute("type", "OpenNet");

    QDomElement mapname = doc.createElement("mapname");
    event.appendChild(mapname);
    mapname.appendChild(doc.createTextNode(QString::fromStdString(e->mapname)));

    return event;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__OpenNetDoneEvent *e, QDomDocument &doc)
{
    QDomElement event = doc.createElement("coviseEvent");

    event.setAttribute("type", "OpenNetDone");

    QDomElement mapname = doc.createElement("mapname");
    event.appendChild(mapname);
    mapname.appendChild(doc.createTextNode(QString::fromStdString(e->mapname)));

    return event;
}

QDomElement CoviseEventMessageSerialiser::serialise(const covise__QuitEvent *, QDomDocument &doc)
{
    QDomElement event = doc.createElement("coviseEvent");
    event.setAttribute("type", "Quit");
    return event;
}
