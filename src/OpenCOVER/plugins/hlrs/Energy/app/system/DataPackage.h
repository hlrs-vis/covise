#pragma once
#include <memory>
#include <utils/read/csv/csv.h>
#include <utils/read/apache/arrow.h>
#include <variant>
#include <map>

using CSVData = std::shared_ptr<opencover::utils::read::CSVStream>;
// using CSVDataMap = opencover::utils::read::StreamMap;
using CSVDataMap = std::map<std::string, CSVData>;
using ArrowData = std::shared_ptr<arrow::Table>;
using ArrowDataMap = std::map<std::string, ArrowData>;

using DataPackage = std::variant<CSVData, ArrowData>;
using DataPackages = std::variant<CSVDataMap, ArrowDataMap>;
