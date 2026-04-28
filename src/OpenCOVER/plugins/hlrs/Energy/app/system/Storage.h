#pragma once
#include <array>

enum class Storage
{
    CSV,
    ARROW,
    PSQL,
    UNKNOWN
};

inline constexpr std::array<Storage, 2> FILE_STORAGE_RANGE = { Storage::ARROW, Storage::CSV };
inline constexpr std::array<Storage, 3> FULL_STORAGE_RANGE = { Storage::ARROW, Storage::CSV, Storage::PSQL };
