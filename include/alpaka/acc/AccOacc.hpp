/* Copyright 2019 Axel Huebl, Benjamin Worpitz, René Widera
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
#include <alpaka/workdiv/WorkDivMembers.hpp>
#include <alpaka/idx/gb/IdxGbOaccBuiltIn.hpp>
#include <alpaka/idx/bt/IdxBtOaccBuiltIn.hpp>
#include <alpaka/atomic/AtomicOaccBuiltIn.hpp>
#include <alpaka/atomic/AtomicHierarchy.hpp>
#include <alpaka/math/MathStdLib.hpp>
#include <alpaka/block/shared/dyn/BlockSharedMemDynOacc.hpp>
#include <alpaka/block/shared/st/BlockSharedMemStOacc.hpp>
#include <alpaka/block/sync/BlockSyncBarrierOacc.hpp>
#include <alpaka/rand/RandStdLib.hpp>
#include <alpaka/time/TimeStdLib.hpp>

// Specialized traits.
#include <alpaka/acc/Traits.hpp>
#include <alpaka/kernel/Traits.hpp>
#include <alpaka/dev/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

// Implementation details.
#include <alpaka/core/ClipCast.hpp>
#include <alpaka/core/Concepts.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevOacc.hpp>

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
    namespace acc
    {
        // define max gang/worker num because there is no standart way in OpenACC to
        // get this information
#ifndef ALPAKA_OACC_MAX_GANG_NUM
        constexpr size_t oaccMaxGangNum = std::numeric_limits<unsigned int>::max();
#else
        constexpr size_t oaccMaxGangNum = ALPAKA_OACC_MAX_GANG_NUM;
#endif
#if defined(ALPAKA_OFFLOAD_MAX_BLOCK_SIZE) && ALPAKA_OFFLOAD_MAX_BLOCK_SIZE>0
        constexpr size_t oaccMaxWorkerNum = ALPAKA_OFFLOAD_MAX_BLOCK_SIZE;
#else
        constexpr size_t oaccMaxWorkerNum = 1;
#endif

        //#############################################################################
        //! The OpenACC accelerator.
        template<
            typename TDim,
            typename TIdx>
        class AccOacc :
            public workdiv::WorkDivMembers<TDim, TIdx>,
            public idx::bt::IdxBtOaccBuiltIn<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicOaccBuiltIn,    // grid atomics
                atomic::AtomicOaccBuiltIn,    // block atomics
                atomic::AtomicOaccBuiltIn     // thread atomics
            >,
            public math::MathStdLib,
            public rand::RandStdLib,
            public time::TimeStdLib,
            public idx::gb::IdxGbOaccBuiltIn<TDim, TIdx>, // dummy
            public block::shared::dyn::BlockSharedMemDynOacc, // dummy
            public block::shared::st::BlockSharedMemStOacc, // dummy
            public block::sync::BlockSyncBarrierOacc, // dummy
            public concepts::Implements<ConceptAcc, AccOacc<TDim, TIdx>>
        {

        protected:
            //-----------------------------------------------------------------------------
            AccOacc(
                vec::Vec<TDim, TIdx> const & gridBlockExtent,
                vec::Vec<TDim, TIdx> const & blockThreadExtent,
                vec::Vec<TDim, TIdx> const & threadElemExtent,
                TIdx const & blockThreadIdx) :
                    workdiv::WorkDivMembers<TDim, TIdx>(gridBlockExtent, blockThreadExtent, threadElemExtent),
                    idx::gb::IdxGbOaccBuiltIn<TDim, TIdx>(),
                    idx::bt::IdxBtOaccBuiltIn<TDim, TIdx>(blockThreadIdx),
                    atomic::AtomicHierarchy<
                        atomic::AtomicOaccBuiltIn,    // grid atomics
                        atomic::AtomicOaccBuiltIn,    // block atomics
                        atomic::AtomicOaccBuiltIn     // thread atomics
                    >(),
                    math::MathStdLib(),
                    block::shared::dyn::BlockSharedMemDynOacc(),
                    //! \TODO can with some TMP determine the amount of statically alloced smem from the kernelFuncObj?
                    block::shared::st::BlockSharedMemStOacc(),
                    block::sync::BlockSyncBarrierOacc(),
                    rand::RandStdLib(),
                    time::TimeStdLib()
            {}

        public:
            //-----------------------------------------------------------------------------
            AccOacc(AccOacc const &) = delete;
            //-----------------------------------------------------------------------------
            AccOacc(AccOacc &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AccOacc const &) -> AccOacc & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AccOacc &&) -> AccOacc & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AccOacc() = default;
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenACC accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccOacc<TDim, TIdx>>
            {
                using type = acc::AccOacc<TDim, TIdx>;
            };
            //#############################################################################
            //! The CPU OpenACC accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccOacc<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevOacc const & dev)
                -> acc::AccDevProps<TDim, TIdx>
                {
                    alpaka::ignore_unused(dev);

#ifdef ALPAKA_CI
                    auto const blockThreadCountMax(alpaka::core::clipCast<TIdx>(std::min(4, oaccMaxWorkerNum)));
                    auto const gridBlockCountMax(alpaka::core::clipCast<TIdx>(std::min(4, oaccMaxGangNum)));
#else
                    auto const blockThreadCountMax(alpaka::core::clipCast<TIdx>(oaccMaxWorkerNum));
                    auto const gridBlockCountMax(alpaka::core::clipCast<TIdx>(oaccMaxGangNum));
#endif
                    return {
                        // m_multiProcessorCount
                        static_cast<TIdx>(gridBlockCountMax),
                        // m_gridBlockExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_gridBlockCountMax
                        std::numeric_limits<TIdx>::max(),
                        // m_blockThreadExtentMax
                        vec::Vec<TDim, TIdx>::all(blockThreadCountMax),
                        // m_blockThreadCountMax
                        blockThreadCountMax,
                        // m_threadElemExtentMax
                        vec::Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                        // m_threadElemCountMax
                        std::numeric_limits<TIdx>::max()};
                }
            };
            //#############################################################################
            //! The OpenACC accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccOacc<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccOacc<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenACC accelerator device type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DevType<
                acc::AccOacc<TDim, TIdx>>
            {
                using type = dev::DevOacc;
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenACC accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                acc::AccOacc<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace kernel
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenACC accelerator execution task type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TWorkDiv,
                typename TKernelFnObj,
                typename... TArgs>
            struct CreateTaskKernel<
                acc::AccOacc<TDim, TIdx>,
                TWorkDiv,
                TKernelFnObj,
                TArgs...>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto createTaskKernel(
                    TWorkDiv const & workDiv,
                    TKernelFnObj const & kernelFnObj,
                    TArgs && ... args)
                {
                    return
                        kernel::TaskKernelOacc<
                            TDim,
                            TIdx,
                            TKernelFnObj,
                            TArgs...>(
                                workDiv,
                                kernelFnObj,
                                std::forward<TArgs>(args)...);
                }
            };
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenACC execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct PltfType<
                acc::AccOacc<TDim, TIdx>>
            {
                using type = pltf::PltfOacc;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenACC accelerator idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                acc::AccOacc<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif