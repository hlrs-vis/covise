#pragma once
#include <initializer_list>

enum class Storage {
    CSV,
    ARROW,
    PSQL
};

constexpr auto FILE_STORAGE_RANGE = { Storage::ARROW, Storage::CSV };
constexpr auto FULL_STORAGE_RANGE = { Storage::ARROW, Storage::CSV , Storage::PSQL};
