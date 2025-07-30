//
// Created by thwdpc on 7/24/25.
//

// #include <fmt/format.h>
#pragma once
#include <fmt/std.h>

#include <stdexcept>

#include "cassert"
#include "depthai/build/version.hpp"
// #include "utility/spdlog-fmt.hpp"

// clang-format off
// Give me back my rust stuff
#define TODO throw std::runtime_error("TODO unimplemented");
#define TODO_V(msg, ...) throw std::runtime_error(fmt::format("TODO unimplemented: {}", fmt::format(msg, ##__VA_ARGS__)));

// Only use this one for internal errors. Clearly invalid states that shouldn't happen.
#define DAI_CHECK_IN(A) \
    if(!(A)) { /* NOLINT(readability-simplify-boolean-expr) */ \
        throw std::runtime_error(fmt::format( \
            "Internal error occured. Please report." \
            " commit: {}" \
            " | dev_v: {}" \
            " | boot_v: {}" \
            " | rvc3_v: {}" \
            " | file: {}:{}", \
            dai::build::COMMIT, \
            dai::build::DEVICE_VERSION, \
            dai::build::BOOTLOADER_VERSION, \
            dai::build::DEVICE_RVC3_VERSION, \
            __FILE__,\
            __LINE__ \
            )); \
    }

#define DAI_CHECK(A, M) \
    if(!(A)) { \
        throw std::runtime_error( M ); \
    }

#define DAI_CHECK_V(A, M, ...) \
    if(!(A)) { \
        throw std::runtime_error(fmt::format( M, ##__VA_ARGS__ )); \
    }

// clang-format on
