// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused kernel: performs exp(A + B) + C
// Processes tiles in 2D blocks (BLOCK_H x BLOCK_W tiles)
// Uses DST_TILES tiles per DST acquire cycle to maximize register usage
// With 8 DST registers and 2 needed per tile, we can process 4 tiles at once

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/tile_move_copy.h"

// Block configuration (passed via defines from Python)
#ifndef BLOCK_H
#define BLOCK_H 8
#endif
#ifndef BLOCK_W
#define BLOCK_W 8
#endif

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;  // A
    constexpr auto cb_in1 = tt::CBIndex::c_1;  // B
    constexpr auto cb_in2 = tt::CBIndex::c_2;  // C
    constexpr auto cb_out = tt::CBIndex::c_16;

    constexpr uint32_t block_h = BLOCK_H;
    constexpr uint32_t block_w = BLOCK_W;
    constexpr uint32_t block_size = block_h * block_w;

    // 8 DST registers available, 2 needed per tile for binary ops = 4 tiles max
    constexpr uint32_t num_dst_regs = 8;
    constexpr uint32_t dst_regs_per_tile = 2; // chain of binary & unary ops
    constexpr uint32_t tiles_per_dst_cycle = num_dst_regs / dst_regs_per_tile;

    // Initialize the SFPU path (unary operations + copy_tile)
    init_sfpu(cb_in0, cb_out);

    // Process tiles in blocks
    uint32_t num_blocks = n_tiles / block_size;
    uint32_t remaining_tiles = n_tiles % block_size;

    for (uint32_t block = 0; block < num_blocks; block++) {
        // Wait for entire block of inputs from reader kernel
        cb_wait_front(cb_in0, block_size);
        cb_wait_front(cb_in1, block_size);
        cb_wait_front(cb_in2, block_size);

        // Process block in DST-sized chunks
        for (uint32_t dst_cycle = 0; dst_cycle < block_size / tiles_per_dst_cycle; dst_cycle++) {
            uint32_t base_t = dst_cycle * tiles_per_dst_cycle;

            // Acquire DST registers for this batch
            tile_regs_acquire();

            // Copy A tiles to even DST registers
            copy_tile_init(cb_in0);
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                copy_tile(cb_in0, base_t + i, i * 2);  // A[i] -> DST[i*2]
            }

            // Copy B tiles to odd DST registers
            copy_tile_init(cb_in1);
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                copy_tile(cb_in1, base_t + i, i * 2 + 1);  // B[i] -> DST[i*2+1]
            }


            // First add: A + B for all tiles
            add_binary_tile_init();
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                add_binary_tile(i * 2, i * 2 + 1, i * 2);  // DST[i*2] + DST[i*2+1] -> DST[i*2]
            }

            // Exp: exp(A + B) for all tiles
            exp_tile_init();
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                exp_tile(i * 2);
            }

            // Copy C tiles to odd DST registers
            copy_tile_init(cb_in2);
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                copy_tile(cb_in2, base_t + i, i * 2 + 1);  // C[i] -> DST[i*2+1]
            }

            // Second add: exp(A + B) + C for all tiles
            add_binary_tile_init();
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                add_binary_tile(i * 2, i * 2 + 1, i * 2);  // DST[i*2] + DST[i*2+1] -> DST[i*2]
            }

            // Commit and wait for results
            tile_regs_commit();
            tile_regs_wait();

            // Pack all output tiles from this DST cycle
            cb_reserve_back(cb_out, tiles_per_dst_cycle);
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                pack_tile(i * 2, cb_out);
            }
            tile_regs_release();

            cb_push_back(cb_out, tiles_per_dst_cycle);
        }

        // Pop entire block of inputs
        cb_pop_front(cb_in0, block_size);
        cb_pop_front(cb_in1, block_size);
        cb_pop_front(cb_in2, block_size);
    }

    // Handle remaining tiles (if n_tiles not divisible by block_size)
    if (remaining_tiles > 0) {
        cb_wait_front(cb_in0, remaining_tiles);
        cb_wait_front(cb_in1, remaining_tiles);
        cb_wait_front(cb_in2, remaining_tiles);

        // Process remaining tiles in DST-sized chunks
        uint32_t remaining_dst_cycles = remaining_tiles / tiles_per_dst_cycle;
        uint32_t final_remainder = remaining_tiles % tiles_per_dst_cycle;

        for (uint32_t dst_cycle = 0; dst_cycle < remaining_dst_cycles; dst_cycle++) {
            uint32_t base_t = dst_cycle * tiles_per_dst_cycle;

            tile_regs_acquire();

            copy_tile_init(cb_in0);
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                copy_tile(cb_in0, base_t + i, i * 2);
            }
            copy_tile_init(cb_in1);
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                copy_tile(cb_in1, base_t + i, i * 2 + 1);
            }

            add_binary_tile_init();
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                add_binary_tile(i * 2, i * 2 + 1, i * 2);
            }

            exp_tile_init();
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                exp_tile(i * 2);
            }

            copy_tile_init(cb_in2);
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                copy_tile(cb_in2, base_t + i, i * 2 + 1);
            }

            add_binary_tile_init();
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                add_binary_tile(i * 2, i * 2 + 1, i * 2);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, tiles_per_dst_cycle);
            for (uint32_t i = 0; i < tiles_per_dst_cycle; i++) {
                pack_tile(i * 2, cb_out);
            }
            tile_regs_release();

            cb_push_back(cb_out, tiles_per_dst_cycle);
        }

        // Handle final remainder (< tiles_per_dst_cycle)
        if (final_remainder > 0) {
            uint32_t base_t = remaining_dst_cycles * tiles_per_dst_cycle;

            tile_regs_acquire();

            copy_tile_init(cb_in0);
            for (uint32_t i = 0; i < final_remainder; i++) {
                copy_tile(cb_in0, base_t + i, i * 2);
            }
            copy_tile_init(cb_in1);
            for (uint32_t i = 0; i < final_remainder; i++) {
                copy_tile(cb_in1, base_t + i, i * 2 + 1);
            }

            add_binary_tile_init();
            for (uint32_t i = 0; i < final_remainder; i++) {
                add_binary_tile(i * 2, i * 2 + 1, i * 2);
            }

            exp_tile_init();
            for (uint32_t i = 0; i < final_remainder; i++) {
                exp_tile(i * 2);
            }

            copy_tile_init(cb_in2);
            for (uint32_t i = 0; i < final_remainder; i++) {
                copy_tile(cb_in2, base_t + i, i * 2 + 1);
            }

            add_binary_tile_init();
            for (uint32_t i = 0; i < final_remainder; i++) {
                add_binary_tile(i * 2, i * 2 + 1, i * 2);
            }

            tile_regs_commit();
            tile_regs_wait();

            cb_reserve_back(cb_out, final_remainder);
            for (uint32_t i = 0; i < final_remainder; i++) {
                pack_tile(i * 2, cb_out);
            }
            tile_regs_release();

            cb_push_back(cb_out, final_remainder);
        }

        cb_pop_front(cb_in0, remaining_tiles);
        cb_pop_front(cb_in1, remaining_tiles);
        cb_pop_front(cb_in2, remaining_tiles);
    }
}
}  // namespace NAMESPACE
