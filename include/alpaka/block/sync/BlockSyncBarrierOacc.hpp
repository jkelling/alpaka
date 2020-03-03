/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef _OPENACC

#include <alpaka/block/sync/Traits.hpp>

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>

namespace alpaka
{
    namespace block
    {
        namespace sync
        {
            //#############################################################################
            //! The OpenMP barrier block synchronization.
            class BlockSyncBarrierOacc : public concepts::Implements<ConceptBlockSync, BlockSyncBarrierOacc>
            {
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierOacc() :
                    m_generation(0u)
                {}
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierOacc(BlockSyncBarrierOacc const &) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierOacc(BlockSyncBarrierOacc &&) = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockSyncBarrierOacc const &) -> BlockSyncBarrierOacc & = delete;
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST auto operator=(BlockSyncBarrierOacc &&) -> BlockSyncBarrierOacc & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~BlockSyncBarrierOacc() = default;

                std::uint8_t mutable m_generation;
                int mutable m_result[2];
            };

            namespace traits
            {
                //#############################################################################
                template<>
                struct SyncBlockThreads<
                    BlockSyncBarrierOacc>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto syncBlockThreads(
                        block::sync::BlockSyncBarrierOacc const & blockSync)
                    -> void
                    {
                        alpaka::ignore_unused(blockSync);

                        // #pragma omp barrier
                    }
                };

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
                            #pragma acc atomic update
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
                            #pragma acc atomic update
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
                            #pragma acc atomic update
                            result |= static_cast<int>(value);
                        }
                    };
                }
            }
        }
    }
}

#endif
