/* Copyright 2020 Axel Huebl, Jeffrey Kelling, Benjamin Worpitz, René Widera
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

#include <alpaka/idx/Traits.hpp>
#include <alpaka/workdiv/Traits.hpp>
#include <alpaka/workdiv/WorkDivMembers.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/idx/MapIdx.hpp>
#include <alpaka/core/Concepts.hpp>

namespace alpaka
{
    namespace bt
    {
        //#############################################################################
        //! The OpenACC accelerator ND index provider.
        template<
            typename TDim,
            typename TIdx>
        class IdxBtOacc : public concepts::Implements<ConceptIdxBt, IdxBtOacc<TDim, TIdx>>
        {
        public:
            //-----------------------------------------------------------------------------
            IdxBtOacc(TIdx blockThreadIdx) : m_blockThreadIdx(blockThreadIdx) {};
            //-----------------------------------------------------------------------------
            IdxBtOacc(IdxBtOacc const &) = delete;
            //-----------------------------------------------------------------------------
            IdxBtOacc(IdxBtOacc &&) = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxBtOacc const & ) -> IdxBtOacc & = delete;
            //-----------------------------------------------------------------------------
            auto operator=(IdxBtOacc &&) -> IdxBtOacc & = delete;
            //-----------------------------------------------------------------------------
            ~IdxBtOacc() = default;

            const TIdx m_blockThreadIdx;
        };
    }

    namespace traits
    {
        //#############################################################################
        //! The OpenACC accelerator index dimension get trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct DimType<
            bt::IdxBtOacc<TDim, TIdx>>
        {
            using type = TDim;
        };

        //#############################################################################
        //! The OpenACC accelerator block thread index get trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct GetIdx<
            bt::IdxBtOacc<TDim, TIdx>,
            origin::Block,
            unit::Threads>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current thread in the block.
            template<
                typename TWorkDiv>
            static auto getIdx(
                bt::IdxBtOacc<TDim, TIdx> const &idx,
                TWorkDiv const & workDiv)
            -> Vec<TDim, TIdx>
            {
                return mapIdx<TDim::value>(
                    Vec<DimInt<1u>, TIdx>(idx.m_blockThreadIdx),
                    getWorkDiv<Block, Threads>(workDiv));
            }
        };

        template<
            typename TIdx>
        struct GetIdx<
            bt::IdxBtOacc<DimInt<1u>, TIdx>,
            origin::Block,
            unit::Threads>
        {
            //-----------------------------------------------------------------------------
            //! \return The index of the current thread in the block.
            template<
                typename TWorkDiv>
            static auto getIdx(
                bt::IdxBtOacc<DimInt<1u>, TIdx> const & idx,
                TWorkDiv const &)
            -> Vec<DimInt<1u>, TIdx>
            {
                return idx.m_blockThreadIdx;
            }
        };

        //#############################################################################
        //! The OpenACC accelerator block thread index idx type trait specialization.
        template<
            typename TDim,
            typename TIdx>
        struct IdxType<
            bt::IdxBtOacc<TDim, TIdx>>
        {
            using type = TIdx;
        };
    }
}

#endif
