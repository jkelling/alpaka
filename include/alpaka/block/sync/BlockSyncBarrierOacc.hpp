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

#include <alpaka/acc/AccCpuOacc.hpp>

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
            //! The OpenACC barrier block synchronization.
            //! Traits are specialized on acc::oacc::detail::AccCpuOaccWorker
            class BlockSyncBarrierOacc : public concepts::Implements<ConceptBlockSync, BlockSyncBarrierOacc>
            {
            public:
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST BlockSyncBarrierOacc()
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

                std::uint8_t mutable m_generation = 0u;
                int mutable m_syncCounter[4] {0,0,0,0};
            };

        }
    }
}

#endif
