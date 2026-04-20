#pragma once
#include "Parser.h"
#include "GridRenderer.h"
#include <lib/core/interfaces/IEnergyGrid.h>
#include <lib/core/ClassLogger.h>

using grid_ptr = std::unique_ptr<core::interface::IEnergyGrid>;

struct GridParser : DataPackageParser<grid_ptr>, core::ClassLogger
{
    GridParser(core::interface::ILogger &logger, const std::string &name, GridRenderConfig &config)
        : core::ClassLogger(logger, name)
        , m_config(config)
    {
    }

    GridRenderConfig m_config;
};
