/* Copyright 2019 Axel Huebl, Benjamin Worpitz
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

#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/idx/Traits.hpp>

#include <alpaka/core/Unused.hpp>
#include <alpaka/vec/Vec.hpp>

namespace alpaka
{
    namespace workdiv
    {
        //#############################################################################
        //! The GPU CUDA accelerator work division.
        template<
            typename TDim,
            typename TIdx>
        class WorkDivOaccBuiltIn : public concepts::Implements<ConceptWorkDiv, WorkDivOaccBuiltIn<TDim, TIdx>>
        {
        public:
            //-----------------------------------------------------------------------------
            WorkDivOaccBuiltIn(
                vec::Vec<TDim, TIdx> const & threadElemExtent,
                vec::Vec<TDim, TIdx> const & blockThreadExtent,
                vec::Vec<TDim, TIdx> const & gridBlockExtent) :
                    m_threadElemExtent(threadElemExtent),
                    m_blockThreadExtent(blockThreadExtent),
                    m_gridBlockExtent(gridBlockExtent)
            {
                // printf("WorkDivOaccBuiltIn ctor threadElemExtent %d\n", int(threadElemExtent[0]));
            }
            //-----------------------------------------------------------------------------
            WorkDivOaccBuiltIn(WorkDivOaccBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            WorkDivOaccBuiltIn(WorkDivOaccBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(WorkDivOaccBuiltIn const &) -> WorkDivOaccBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(WorkDivOaccBuiltIn &&) -> WorkDivOaccBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~WorkDivOaccBuiltIn() = default;

        public:
            // \TODO: Optimize! Add WorkDivCudaBuiltInNoElems that has no member m_threadElemExtent as well as AccGpuCudaRtNoElems.
            // Use it instead of AccGpuCudaRt if the thread element extent is one to reduce the register usage.
            vec::Vec<TDim, TIdx> const m_threadElemExtent, m_blockThreadExtent, m_gridBlockExtent;
        };
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                workdiv::WorkDivOaccBuiltIn<TDim, TIdx>>
            {
                using type = TDim;
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                workdiv::WorkDivOaccBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
    namespace workdiv
    {
        namespace traits
        {
            //#############################################################################
            //! The GPU CUDA accelerator work division grid block extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivOaccBuiltIn<TDim, TIdx>,
                origin::Grid,
                unit::Blocks>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                static auto getWorkDiv(
                    WorkDivOaccBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    return workDiv.m_gridBlockExtent;
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division block thread extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivOaccBuiltIn<TDim, TIdx>,
                origin::Block,
                unit::Threads>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of threads in each dimension of a block.
                static auto getWorkDiv(
                    WorkDivOaccBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    return workDiv.m_blockThreadExtent;
                    // return vec::Vec<TDim, TIdx>(static_cast<TIdx>(omp_get_num_threads()));
                }
            };

            //#############################################################################
            //! The GPU CUDA accelerator work division thread element extent trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct GetWorkDiv<
                WorkDivOaccBuiltIn<TDim, TIdx>,
                origin::Thread,
                unit::Elems>
            {
                //-----------------------------------------------------------------------------
                //! \return The number of blocks in each dimension of the grid.
                static auto getWorkDiv(
                    WorkDivOaccBuiltIn<TDim, TIdx> const & workDiv)
                -> vec::Vec<TDim, TIdx>
                {
                    return workDiv.m_threadElemExtent;
                }
            };
        }
    }
}

#endif
