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
        // define max gang/worker num because there is no standart way in OpenACC to
        // get this information
#ifndef ALPAKA_OACC_MAX_GANG_NUM
        constexpr size_t oaccMaxGangNum = std::numeric_limits<unsigned int>::max();
#else
        constexpr size_t oaccMaxGangNum = ALPAKA_OACC_MAX_GANG_NUM;
#endif
#if defined(ALPAKA_OFFLOAD_MAX_BLOCK_SIZE) && ALPAKA_OFFLOAD_MAX_BLOCK_SIZE>0 && 0
        constexpr size_t oaccMaxWorkerNum = ALPAKA_OFFLOAD_MAX_BLOCK_SIZE;
#else
        constexpr size_t oaccMaxWorkerNum = 1;
#endif

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
                TIdx const & gridBlockIdx,
                TIdx const & blockSharedMemDynSizeBytes) :
                    workdiv::WorkDivOaccBuiltIn<TDim, TIdx>(threadElemExtent, blockThreadExtent, gridBlockExtent),
                    idx::gb::IdxGbOaccBuiltIn<TDim, TIdx>(gridBlockIdx),
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
                    time::TimeStdLib()
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

    namespace acc
    {
        namespace oacc
        {
            namespace detail
            {
                template<
                    typename TDim,
                    typename TIdx>
                struct AccCpuOaccWorker
                {
                    acc::AccCpuOacc<TDim, TIdx>& m_acc;
                    const TIdx m_blockThreadIdx;

                        operator acc::AccCpuOacc<TDim, TIdx>& () {return m_acc;}
                        operator acc::AccCpuOacc<TDim, TIdx> const & () const {return m_acc;}

                        AccCpuOaccWorker(acc::AccCpuOacc<TDim, TIdx>& acc, const TIdx& wIdx) :
                            m_acc(acc),
                            m_blockThreadIdx(wIdx)
                    {}
                };
            }
        }
    }

    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The OpenACC accelerator block thread index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                acc::oacc::detail::AccCpuOaccWorker<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                static auto getIdx(
                    acc::oacc::detail::AccCpuOaccWorker<TDim, TIdx> const &idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    return idx::mapIdx<TDim::value>(
                        vec::Vec<dim::DimInt<1u>, TIdx>(idx.m_blockThreadIdx),
                        workdiv::getWorkDiv<Block, Threads>(workDiv));
                }
            };

            template<
                typename TIdx>
            struct GetIdx<
                acc::oacc::detail::AccCpuOaccWorker<dim::DimInt<1u>, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current thread in the block.
                template<
                    typename TWorkDiv>
                static auto getIdx(
                    acc::oacc::detail::AccCpuOaccWorker<dim::DimInt<1u>, TIdx> const & idx,
                    TWorkDiv const &)
                -> vec::Vec<dim::DimInt<1u>, TIdx>
                {
                    return idx.m_blockThreadIdx;
                }
            };
        }
    }
}

#endif
