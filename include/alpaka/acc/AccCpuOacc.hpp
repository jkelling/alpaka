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
#include <alpaka/workdiv/WorkDivOaccBuiltIn.hpp>
#include <alpaka/idx/gb/IdxGbOaccBuiltIn.hpp>
#include <alpaka/idx/bt/IdxBtOaccBuiltIn.hpp>
#include <alpaka/atomic/AtomicNoOp.hpp>
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
        class TaskKernelCpuOacc;
    }
    namespace acc
    {
        //#############################################################################
        //! The OpenACC accelerator.
        template<
            typename TDim,
            typename TIdx>
        class AccCpuOacc final :
            public workdiv::WorkDivOaccBuiltIn<TDim, TIdx>,
            public idx::gb::IdxGbOaccBuiltIn<TDim, TIdx>,
            public idx::bt::IdxBtOaccBuiltIn<TDim, TIdx>,
            public atomic::AtomicHierarchy<
                atomic::AtomicNoOp,   // grid atomics
                atomic::AtomicNoOp,    // block atomics
                atomic::AtomicNoOp     // thread atomics
            >,
            public math::MathStdLib,
            public block::shared::dyn::BlockSharedMemDynOacc,
            public block::shared::st::BlockSharedMemStOacc,
            public block::sync::BlockSyncBarrierOacc,
            public rand::RandStdLib,
            public time::TimeStdLib,
            public concepts::Implements<ConceptAcc, AccCpuOacc<TDim, TIdx>>
        {
        public:
            // Partial specialization with the correct TDim and TIdx is not allowed.
            template<
                typename TDim2,
                typename TIdx2,
                typename TKernelFnObj,
                typename... TArgs>
            friend class ::alpaka::kernel::TaskKernelCpuOacc;

        private:
            //-----------------------------------------------------------------------------
            AccCpuOacc(
                vec::Vec<TDim, TIdx> const & gridBlockExtent,
                vec::Vec<TDim, TIdx> const & blockThreadExtent,
                vec::Vec<TDim, TIdx> const & threadElemExtent,
                TIdx const & teamOffset,
                TIdx const & blockSharedMemDynSizeBytes) :
                    workdiv::WorkDivOaccBuiltIn<TDim, TIdx>(threadElemExtent, blockThreadExtent, gridBlockExtent),
                    idx::gb::IdxGbOaccBuiltIn<TDim, TIdx>(teamOffset),
                    idx::bt::IdxBtOaccBuiltIn<TDim, TIdx>(),
                    atomic::AtomicHierarchy<
                        atomic::AtomicNoOp,// atomics between grids
                        atomic::AtomicNoOp, // atomics between blocks
                        atomic::AtomicNoOp  // atomics between threads
                    >(),
                    math::MathStdLib(),
                    block::shared::dyn::BlockSharedMemDynOacc(static_cast<std::size_t>(blockSharedMemDynSizeBytes)),
                    //! \TODO can with some TMP determine the amount of statically alloced smem from the kernelFuncObj?
                    block::shared::st::BlockSharedMemStOacc(staticMemBegin()),
                    block::sync::BlockSyncBarrierOacc(),
                    rand::RandStdLib(),
                    time::TimeStdLib(),
                    m_gridBlockIdx(vec::Vec<TDim, TIdx>::zeros())
            {}

        public:
            //-----------------------------------------------------------------------------
            AccCpuOacc(AccCpuOacc const &) = delete;
            //-----------------------------------------------------------------------------
            AccCpuOacc(AccCpuOacc &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AccCpuOacc const &) -> AccCpuOacc & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(AccCpuOacc &&) -> AccCpuOacc & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AccCpuOacc() = default;

        private:
            // getIdx
            vec::Vec<TDim, TIdx> m_gridBlockIdx;    //!< The index of the currently executed block.
        };
    }

    namespace acc
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator accelerator type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct AccType<
                acc::AccCpuOacc<TDim, TIdx>>
            {
                using type = acc::AccCpuOacc<TDim, TIdx>;
            };
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator device properties get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccDevProps<
                acc::AccCpuOacc<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccDevProps(
                    dev::DevOacc const & dev)
                -> acc::AccDevProps<TDim, TIdx>
                {
                    alpaka::ignore_unused(dev);

#if defined(ALPAKA_OFFLOAD_MAX_BLOCK_SIZE) && ALPAKA_OFFLOAD_MAX_BLOCK_SIZE>0
                    auto const blockThreadCount = ALPAKA_OFFLOAD_MAX_BLOCK_SIZE;
#else
                    auto const blockThreadCount = ::omp_get_max_threads();
#endif
#ifdef ALPAKA_CI
                    auto const blockThreadCountMax(alpaka::core::clipCast<TIdx>(std::min(4, blockThreadCount)));
                    auto const gridBlockCountMax(alpaka::core::clipCast<TIdx>(std::min(4, ::omp_get_max_threads())));
#else
                    auto const blockThreadCountMax(alpaka::core::clipCast<TIdx>(blockThreadCount));
                    auto const gridBlockCountMax(alpaka::core::clipCast<TIdx>(::omp_get_max_threads())); //! \todo fix max block size for target
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
            //! The CPU OpenMP 4.0 accelerator name trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetAccName<
                acc::AccCpuOacc<TDim, TIdx>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAccName()
                -> std::string
                {
                    return "AccCpuOacc<" + std::to_string(TDim::value) + "," + typeid(TIdx).name() + ">";
                }
            };
        }
    }
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CPU OpenMP 4.0 accelerator device type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DevType<
                acc::AccCpuOacc<TDim, TIdx>>
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
            //! The CPU OpenMP 4.0 accelerator dimension getter trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                acc::AccCpuOacc<TDim, TIdx>>
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
            //! The CPU OpenMP 4.0 accelerator execution task type trait specialization.
            template<
                typename TDim,
                typename TIdx,
                typename TWorkDiv,
                typename TKernelFnObj,
                typename... TArgs>
            struct CreateTaskKernel<
                acc::AccCpuOacc<TDim, TIdx>,
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
                        kernel::TaskKernelCpuOacc<
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
            //! The CPU OpenMP 4.0 execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct PltfType<
                acc::AccCpuOacc<TDim, TIdx>>
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
            //! The CPU OpenMP 4.0 accelerator idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                acc::AccCpuOacc<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
