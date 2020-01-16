/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef _OPENMP

#include <alpaka/block/sync/Traits.hpp>

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

#ifdef SPEC_FAKE_OMP_TARGET_CPU
#include <mutex>
#include <alpaka/core/BarrierThread.hpp>
#endif

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The OpenMP barrier block synchronization.
            class BlockSyncBarrierOmp : public concepts::Implements<ConceptBlockSync, BlockSyncBarrierOmp>
            {
            public:
                //-----------------------------------------------------------------------------
#ifndef SPEC_FAKE_OMP_TARGET_CPU
                ALPAKA_FN_HOST BlockSyncBarrierOmp() :
                    m_generation(0u)
                {
                }
#else
                ALPAKA_FN_HOST BlockSyncBarrierOmp(int numThreads) :
                    m_barrier(numThreads),
                    m_generation(0u)
                {
                }
#endif
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierOmp(BlockSyncBarrierOmp const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierOmp(BlockSyncBarrierOmp &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockSyncBarrierOmp const &) -> BlockSyncBarrierOmp & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockSyncBarrierOmp &&) -> BlockSyncBarrierOmp & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~BlockSyncBarrierOmp() = default;

                std::uint8_t mutable m_generation;
                int mutable m_result[2];

#ifdef SPEC_FAKE_OMP_TARGET_CPU
                mutable core::threads::BarrierThread<int> m_barrier;
                std::thread::id m_masterThread;
                mutable std::mutex m_mtxAtomic;
#endif

            };

            namespace traits
            {
                //#############################################################################
                template<>
                struct SyncBlockThreads<
                    BlockSyncBarrierOmp>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto syncBlockThreads(
                        block::sync::BlockSyncBarrierOmp const & blockSync)
                    -> void
                    {
                        alpaka::ignore_unused(blockSync);

                        // NOTE: This waits for all threads in all blocks.
                        // If multiple blocks are executed in parallel this is not optimal.
#ifndef SPEC_FAKE_OMP_TARGET_CPU
                        #pragma omp barrier
#else
                        blockSync.m_barrier.wait();
#endif
                    }
                };

#ifndef SPEC_FAKE_OMP_TARGET_CPU
                namespace detail
                {
                    //#############################################################################
                    template<
                        typename TOp>
                    struct AtomicOp;
                    //#############################################################################
                    template<>
                    struct AtomicOp<
                        block::sync::op::Count>
                    {
                        void operator()(int& result, bool value)
                        {
#ifdef SPEC_FAKE_OMP_TARGET_CPU
                            std::cout << "WARN: atomic count in SPEC_FAKE_OMP_TARGET_CPU kernel." << std::endl;
#else
                            #pragma omp atomic
#endif
                            result += static_cast<int>(value);
                        }
                    };
                    //#############################################################################
                    template<>
                    struct AtomicOp<
                        block::sync::op::LogicalAnd>
                    {
                        void operator()(int& result, bool value)
                        {
#ifdef SPEC_FAKE_OMP_TARGET_CPU
                            std::cout << "WARN: atomic and in SPEC_FAKE_OMP_TARGET_CPU kernel." << std::endl;
#else
                            #pragma omp atomic
#endif
                            result &= static_cast<int>(value);
                        }
                    };
                    //#############################################################################
                    template<>
                    struct AtomicOp<
                        block::sync::op::LogicalOr>
                    {
                        void operator()(int& result, bool value)
                        {
#ifdef SPEC_FAKE_OMP_TARGET_CPU
                            std::cout << "WARN: atomic or in SPEC_FAKE_OMP_TARGET_CPU kernel." << std::endl;
#else
                            #pragma omp atomic
#endif
                            result |= static_cast<int>(value);
                        }
                    };
                }
#endif

                //#############################################################################
                template<
                    typename TOp>
                struct SyncBlockThreadsPredicate<
                    TOp,
                    BlockSyncBarrierOmp>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_NO_HOST_ACC_WARNING
                    ALPAKA_FN_ACC static auto syncBlockThreadsPredicate(
                        block::sync::BlockSyncBarrierOmp const & blockSync,
                        int predicate)
                    -> int
                    {
#ifdef SPEC_FAKE_OMP_TARGET_CPU
                        std::lock_guard<std::mutex> lock(blockSync.m_mtxAtomic);
                        std::cout << "Error: SyncBlockThreadsPredicate in SPEC_FAKE_OMP_TARGET_CPU kernel." << std::endl;
                        return 0;
#else
                        // The first thread initializes the value.
                        // There is an implicit barrier at the end of omp single.
                        // NOTE: This code is executed only once for all OpenMP threads.
                        // If multiple blocks with multiple threads are executed in parallel
                        // this reduction is executed only for one block!
                        #pragma omp single
                        {
                            ++blockSync.m_generation;
                            blockSync.m_result[blockSync.m_generation % 2u] = TOp::InitialValue;
                        }

                        auto const generationMod2(blockSync.m_generation % 2u);
                        int& result(blockSync.m_result[generationMod2]);
                        bool const predicateBool(predicate != 0);

                        detail::AtomicOp<TOp>()(result, predicateBool);

                        // Wait for all threads to write their predicate into the vector.
                        // NOTE: This waits for all threads in all blocks.
                        // If multiple blocks are executed in parallel this is not optimal.
                        #pragma omp barrier

                        return blockSync.m_result[generationMod2];
#endif
                    }
                };
            }
        }
    }
}

#endif
