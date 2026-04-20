#pragma once
#include "DataPackage.h"

template <typename T>
struct DataPackageParser
{
    typedef T Type;
    virtual T operator()(CSVDataMap &map) = 0;
    virtual T operator()(const ArrowDataMap &map) = 0;
    virtual T operator()(const ArrowData &data) = 0;
    virtual T operator()(CSVData &data) = 0;
};
