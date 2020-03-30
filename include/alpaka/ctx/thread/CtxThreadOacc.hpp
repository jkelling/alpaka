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
#include <alpaka/ctx/block/CtxBlockOacc.hpp>
#include <alpaka/acc/AccCpuOacc.hpp>

// Specialized traits.
#include <alpaka/idx/Traits.hpp>
#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/block/sync/Traits.hpp>

#include <limits>
#include <typeinfo>

namespace alpaka
{
    namespace ctx
    {
        //#############################################################################
        //! The OpenACC block context.
        template<
            typename TDim,
            typename TIdx>
        class CtxThreadOacc final :
            public acc::AccCpuOacc<TDim, TIdx>
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
            CtxThreadOacc(
                vec::Vec<TDim, TIdx> const & gridBlockExtent,
                vec::Vec<TDim, TIdx> const & blockThreadExtent,
                vec::Vec<TDim, TIdx> const & threadElemExtent,
                TIdx const & blockThreadIdx,
                ctx::CtxBlockOacc<TDim, TIdx>& blockShared) :
                    acc::AccCpuOacc<TDim, TIdx>(threadElemExtent, blockThreadExtent, gridBlockExtent, blockThreadIdx),
                    m_blockShared(blockShared)
            {}

        public:
            //-----------------------------------------------------------------------------
            CtxThreadOacc(CtxThreadOacc const &) = delete;
            //-----------------------------------------------------------------------------
            CtxThreadOacc(CtxThreadOacc &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(CtxThreadOacc const &) -> CtxThreadOacc & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(CtxThreadOacc &&) -> CtxThreadOacc & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~CtxThreadOacc() = default;

            ctx::CtxBlockOacc<TDim, TIdx>& m_blockShared;
        };
    }

    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator grid block index get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetIdx<
                ctx::CtxThreadOacc<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                template<
                    typename TWorkDiv>
                static auto getIdx(
                    ctx::CtxThreadOacc<TDim, TIdx> const & idx,
                    TWorkDiv const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    // // \TODO: Would it be faster to precompute the index and cache it inside an array?
                    return idx::mapIdx<TDim::value>(
                        vec::Vec<dim::DimInt<1u>, TIdx>(idx.m_blockShared.m_gridBlockIdx),
                        workdiv::getWorkDiv<Grid, Blocks>(workDiv));
                }
            };

            template<
                typename TIdx>
            struct GetIdx<
                ctx::CtxThreadOacc<dim::DimInt<1u>, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The index of the current block in the grid.
                template<
                    typename TWorkDiv>
                static auto getIdx(
                    ctx::CtxThreadOacc<dim::DimInt<1u>, TIdx> const & idx,
                    TWorkDiv const &)
                -> vec::Vec<dim::DimInt<1u>, TIdx>
                {
                    return idx.m_blockShared.m_gridBlockIdx;
                }
            };
        }
    }

    namespace block
    {
        namespace shared
        {
            namespace dyn
            {
                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T,
                        typename TDim,
                        typename TIdx>
                    struct GetMem<
                        T,
                        ctx::CtxThreadOacc<TDim, TIdx>>
                    {
                        //-----------------------------------------------------------------------------
                        static auto getMem(
                            ctx::CtxThreadOacc<TDim, TIdx> const &mem)
                        -> T *
                        {
                            return reinterpret_cast<T*>(mem.m_blockShared.dynMemBegin());
                        }
                    };
                }
            }
        }

        namespace shared
        {
            namespace st
            {
                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T,
                        typename TDim,
                        typename TIdx,
                        std::size_t TuniqueId>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        ctx::CtxThreadOacc<TDim, TIdx>>
                    {
                        //-----------------------------------------------------------------------------
                        static auto allocVar(
                            ctx::CtxThreadOacc<TDim, TIdx> const &smem)
                        -> T &
                        {
                            return smem.m_blockShared.template alloc<T>();
                        }
                    };
                }
            }
        }

        namespace sync
        {
            namespace traits
            {
                //#############################################################################
                template<
                    typename TDim,
                    typename TIdx>
                struct SyncBlockThreads<
                    ctx::CtxThreadOacc<TDim, TIdx>>
                {
                    //-----------------------------------------------------------------------------
                    //! Execute op with single thread (any idx, last thread to
                    //! arrive at barrier executes) syncing before and after
                    template<
                        typename TOp>
                    ALPAKA_FN_HOST static auto masterOpBlockThreads(
                        ctx::CtxThreadOacc<TDim, TIdx> const & acc,
                        TOp &&op = [](){})
                    -> void
                    {
                        const auto slot = (acc.m_generation&1)<<1;
                        const int workerNum = static_cast<int>(workdiv::getWorkDiv<Block, Threads>(acc).prod());
                        int sum;
                        #pragma acc atomic capture
                        {
                            ++acc.m_blockShared.m_syncCounter[slot];
                            sum = acc.m_blockShared.m_syncCounter[slot];
                        }
                        if(sum == workerNum)
                        {
                            ++acc.m_blockShared.m_generation;
                            const int nextSlot = (acc.m_blockShared.m_generation&1)<<1;
                            acc.m_blockShared.m_syncCounter[nextSlot] = 0;
                            acc.m_blockShared.m_syncCounter[nextSlot+1] = 0;
                            op();
                        }
                        while(sum < workerNum)
                        {
                            #pragma acc atomic read
                            sum = acc.m_blockShared.m_syncCounter[slot];
                        }
                        #pragma acc atomic capture
                        {
                            ++acc.m_syncCounter[slot];
                            sum = acc.m_blockShared.m_syncCounter[slot];
                        }
                        while(sum < workerNum)
                        {
                            #pragma acc atomic read
                            sum = acc.m_blockShared.m_syncCounter[slot+1];
                        }
                    }

                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto syncBlockThreads(
                        ctx::CtxThreadOacc<TDim, TIdx> const & acc)
                    -> void
                    {
                        masterOpBlockThreads<>(acc);
                    }
                };
            }
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
                ctx::CtxThreadOacc<TDim, TIdx>>
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
                ctx::CtxThreadOacc<TDim, TIdx>>
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
                ctx::CtxThreadOacc<TDim, TIdx>,
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
            //! The CPU OpenACC execution task platform type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct PltfType<
                ctx::CtxThreadOacc<TDim, TIdx>>
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
                ctx::CtxThreadOacc<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
