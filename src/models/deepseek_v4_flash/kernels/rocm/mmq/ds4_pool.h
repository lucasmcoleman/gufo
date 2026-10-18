// SPDX-License-Identifier: MIT
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Selects the scratch pool for HIP contexts created after this call: nonzero
// uses one LIFO arena in managed memory (single stream, no graph capture),
// zero the default hipMallocAsync pool. Call before the first ds4_mmq entry.
void ds4_pool_use_stack_arena(int enable);

#ifdef __cplusplus
}  // extern "C"
#endif
