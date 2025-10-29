#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 64x64, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  ttcore.global @lhs = tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> [0]
  ttcore.global @rhs = tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> [1]
  func.func @matmul(%arg0: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {d2m.stream = true}, %arg1: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {d2m.stream = true}, %arg2: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {d2m.stream = false}) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.empty() : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    %stream = "d2m.stream_layout"(%arg0, %0) : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    %1 = d2m.empty() : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    %stream_0 = "d2m.stream_layout"(%arg1, %1) : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>

    %2 = d2m.generic {block_factors = [], grid = #ttcore.grid<2x2>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%stream, %stream_0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
        outs(%arg2 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)  {
    ^datamovement0(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
      %c2 = arith.constant 2 : index
      %c2_1 = arith.constant 2 : index
      %c2_2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %3 = ttcore.get_global @lhs : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      %c1_3 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c2_2 step %c1_3 {
        %c0_4 = arith.constant 0 : index
        %c1_5 = arith.constant 1 : index
        scf.for %arg4 = %c0_4 to %c1 step %c1_5 {
          %4 = d2m.reserve %cb0 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
          %c0_i64 = arith.constant 0 : i64
          %5 = arith.index_cast %c0_i64 : i64 to index
          %6 = arith.cmpi eq, %core1, %5 : index
          scf.if %6 {
            %7 = arith.muli %core0, %c1 : index
            %8 = arith.addi %7, %arg4 : index
            %tx = d2m.dma %3 [%8, %arg3], %4 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
            d2m.dma_wait %tx
            %c1_i64 = arith.constant 1 : i64
            %9 = arith.index_cast %c1_i64 : i64 to index
            %10 = arith.subi %c2, %9 : index
            %c0_i64_6 = arith.constant 0 : i64
            %11 = arith.index_cast %c0_i64_6 : i64 to index
            d2m.semaphore_wait %sem0, %10 reset %11
            %c1_i64_7 = arith.constant 1 : i64
            %c1_i64_8 = arith.constant 1 : i64
            %c1_i64_9 = arith.constant 1 : i64
            %12 = arith.index_cast %c1_i64_9 : i64 to index
            %13 = arith.subi %c2_1, %12 : index
            %14 = arith.index_cast %c1_i64_7 : i64 to index
            %15 = arith.index_cast %c1_i64_8 : i64 to index
            %tx_10 = d2m.dma %4, %4 core[%core0, %14] mcast[%15, %13] : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
            d2m.dma_wait %tx_10
            %c1_i64_11 = arith.constant 1 : i64
            %c1_i64_12 = arith.constant 1 : i64
            %c1_i64_13 = arith.constant 1 : i64
            %c1_i64_14 = arith.constant 1 : i64
            %16 = arith.index_cast %c1_i64_14 : i64 to index
            %17 = arith.subi %c2_1, %16 : index
            %18 = arith.index_cast %c1_i64_11 : i64 to index
            %19 = arith.index_cast %c1_i64_12 : i64 to index
            %20 = arith.index_cast %c1_i64_13 : i64 to index
            d2m.semaphore_set %sem1, %18, core[%core0, %19] mcast[%20, %17]
          } else {
            %c1_i64 = arith.constant 1 : i64
            %c0_i64_6 = arith.constant 0 : i64
            %7 = arith.index_cast %c1_i64 : i64 to index
            %8 = arith.index_cast %c0_i64_6 : i64 to index
            d2m.semaphore_inc %sem0, %7, core[%core0, %8]
            %c1_i64_7 = arith.constant 1 : i64
            %c0_i64_8 = arith.constant 0 : i64
            %9 = arith.index_cast %c1_i64_7 : i64 to index
            %10 = arith.index_cast %c0_i64_8 : i64 to index
            d2m.semaphore_wait %sem1, %9 reset %10
          }
        }
      }
      d2m.yield %arg2 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
    }, {
    ^datamovement1(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
      %c2 = arith.constant 2 : index
      %c2_1 = arith.constant 2 : index
      %c2_2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c1_3 = arith.constant 1 : index
      %3 = ttcore.get_global @rhs : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
      %core0 = d2m.core_index(0) : index
      %core1 = d2m.core_index(1) : index
      %c0 = arith.constant 0 : index
      %c1_4 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c2_2 step %c1_4 {
        %c0_5 = arith.constant 0 : index
        %c1_6 = arith.constant 1 : index
        scf.for %arg4 = %c0_5 to %c1 step %c1_6 {
          %c0_7 = arith.constant 0 : index
          %c1_8 = arith.constant 1 : index
          scf.for %arg5 = %c0_7 to %c1_3 step %c1_8 {
            %4 = d2m.reserve %cb1 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
            %c0_i64 = arith.constant 0 : i64
            %5 = arith.index_cast %c0_i64 : i64 to index
            %6 = arith.cmpi eq, %core0, %5 : index
            scf.if %6 {
              %7 = arith.muli %core1, %c1_3 : index
              %8 = arith.addi %7, %arg5 : index
              %tx = d2m.dma %3 [%arg3, %8], %4 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
              d2m.dma_wait %tx
              %c1_i64 = arith.constant 1 : i64
              %9 = arith.index_cast %c1_i64 : i64 to index
              %10 = arith.subi %c2, %9 : index
              %c0_i64_9 = arith.constant 0 : i64
              %11 = arith.index_cast %c0_i64_9 : i64 to index
              d2m.semaphore_wait %sem2, %10 reset %11
              %c1_i64_10 = arith.constant 1 : i64
              %c1_i64_11 = arith.constant 1 : i64
              %12 = arith.index_cast %c1_i64_11 : i64 to index
              %13 = arith.subi %c2_1, %12 : index
              %c1_i64_12 = arith.constant 1 : i64
              %14 = arith.index_cast %c1_i64_10 : i64 to index
              %15 = arith.index_cast %c1_i64_12 : i64 to index
              %tx_13 = d2m.dma %4, %4 core[%14, %core1] mcast[%13, %15] : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> !d2m.mem_tx
              d2m.dma_wait %tx_13
              %c1_i64_14 = arith.constant 1 : i64
              %c1_i64_15 = arith.constant 1 : i64
              %c1_i64_16 = arith.constant 1 : i64
              %16 = arith.index_cast %c1_i64_16 : i64 to index
              %17 = arith.subi %c2_1, %16 : index
              %c1_i64_17 = arith.constant 1 : i64
              %18 = arith.index_cast %c1_i64_14 : i64 to index
              %19 = arith.index_cast %c1_i64_15 : i64 to index
              %20 = arith.index_cast %c1_i64_17 : i64 to index
              d2m.semaphore_set %sem3, %18, core[%19, %core1] mcast[%17, %20]
            } else {
              %c1_i64 = arith.constant 1 : i64
              %c0_i64_9 = arith.constant 0 : i64
              %7 = arith.index_cast %c1_i64 : i64 to index
              %8 = arith.index_cast %c0_i64_9 : i64 to index
              d2m.semaphore_inc %sem2, %7, core[%8, %core1]
              %c1_i64_10 = arith.constant 1 : i64
              %c0_i64_11 = arith.constant 0 : i64
              %9 = arith.index_cast %c1_i64_10 : i64 to index
              %10 = arith.index_cast %c0_i64_11 : i64 to index
              d2m.semaphore_wait %sem3, %9 reset %10
            }
          }
        }
      }
      d2m.yield %arg2 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
    }, {
    ^compute0(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %sem0: !d2m.semaphore, %sem1: !d2m.semaphore, %sem2: !d2m.semaphore, %sem3: !d2m.semaphore):
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1_2 = arith.constant 1 : index
      scf.for %arg3 = %c0 to %c2 step %c1_2 {
        %c0_3 = arith.constant 0 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg4 = %c0_3 to %c1 step %c1_4 {
          %3 = d2m.wait %cb0 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
          %c0_5 = arith.constant 0 : index
          %c1_6 = arith.constant 1 : index
          scf.for %arg5 = %c0_5 to %c1_1 step %c1_6 {
            %4 = d2m.wait %cb1 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
            %5 = d2m.reserve %cb2 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
            %6 = d2m.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
            %7 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %4 : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) outs(%6 : tensor<2x2x!ttcore.tile<32x32, f32>>) {
            ^bb0(%in: !ttcore.tile<32x32, f32>, %in_7: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
              %9 = "d2m.tile_matmul"(%in, %in_7, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
              linalg.yield %9 : !ttcore.tile<32x32, f32>
            } -> tensor<2x2x!ttcore.tile<32x32, f32>>
            d2m.store %5, %7 : tensor<2x2x!ttcore.tile<32x32, f32>>
            %8 = d2m.wait %cb2 : <tensor<2x2x!ttcore.tile<32x32, f32>>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
          }
        }
      }
    d2m.yield %arg2 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
    } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    return %2 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
  }
}
