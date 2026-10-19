#pragma once

#include <algorithm>
#include <cstdint>

namespace strata::prefill::cuda_length_chunks {

// Opt-in, measured on four V100s. The caller passes the uncached segment length,
// and the configured ceiling always wins. No hardware/load-dependent choices.
inline int64_t cap(int64_t tokens, int64_t configured) {
    if (tokens <= 0 || configured <= 0) return 0;
    const int64_t measured = tokens <= 8192 ? 2048 : tokens <= 49152 ? 4096 : 8192;
    return std::min(configured, measured);
}

}  // namespace strata::prefill::cuda_length_chunks
