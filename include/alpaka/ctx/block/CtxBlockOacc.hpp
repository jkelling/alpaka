/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OACC_ENABLED

#if _OPENACC < 201306
    #error If ALPAKA_ACC_ANY_BT_OACC_ENABLED is set, the compiler has to support OpenACC xx or higher!
#endif

// Base classes.
#include <alpaka/idx/gb/IdxGbOaccBuiltIn.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynOacc.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStOacc.hpp>
#include <alpaka/block/sync/BlockSyncBarrierOacc.hpp>

// Specialized traits.
#include <alpaka/idx/Traits.hpp>

#include <limits>
#include <typeinfo>

namespace alpaka
{
    namespace kernel
    {
        template<
            typename TDim,
            typename TIdx,
            typename TKernelFnObj,
            typename... TArgs>
        class TaskKernelOacc;
    }
    namespace ctx
    {
        //#############################################################################
        //! The OpenACC block context.
        template<
            typename TDim,
            typename TIdx>
        class CtxBlockOacc final :
            public idx::gb::IdxGbOaccBuiltIn<TDim, TIdx>::BlockShared,
            public block::shared::dyn::BlockSharedMemDynOacc::BlockShared,
            public block::shared::st::BlockSharedMemStOacc::BlockShared<CtxBlockOacc<TDim, TIdx>>,
            public block::sync::BlockSyncBarrierOacc::BlockShared
        {
        public:
            // Partial specialization with the correct TDim and TIdx is not allowed.
            template<
                typename TDim2,
                typename TIdx2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::kernel::TaskKernelOacc;

        protected:
            //-----------------------------------------------------------------------------
            CtxBlockOacc(
                TIdx const & gridBlockIdx,
                TIdx const & blockSharedMemDynSizeBytes) :
                    idx::gb::IdxGbOaccBuiltIn<TDim, TIdx>::BlockShared(gridBlockIdx),
                    block::shared::dyn::BlockSharedMemDynOacc::BlockShared(static_cast<std::size_t>(blockSharedMemDynSizeBytes)),
                    //! \TODO can with some TMP determine the amount of statically alloced smem from the kernelFuncObj?
                    block::shared::st::BlockSharedMemStOacc::BlockShared<CtxBlockOacc<TDim, TIdx>>(staticMemBegin()),
                    block::sync::BlockSyncBarrierOacc::BlockShared()
            {}

        public:
            //-----------------------------------------------------------------------------
            CtxBlockOacc(CtxBlockOacc const &) = delete;
            //-----------------------------------------------------------------------------
            CtxBlockOacc(CtxBlockOacc &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(CtxBlockOacc const &) -> CtxBlockOacc & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(CtxBlockOacc &&) -> CtxBlockOacc & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~CtxBlockOacc() = default;
        };
    }
}

#endif
