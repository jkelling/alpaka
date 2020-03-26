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

#include <alpaka/idx/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/workdiv/WorkDivOaccBuiltIn.hpp>
#include <alpaka/core/Positioning.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/idx/MapIdx.hpp>

namespace alpaka
{

    namespace idx
    {
        namespace bt
        {
            //#############################################################################
            //! The CUDA accelerator ND index provider.
            template<
                typename TDim,
                typename TIdx>
            class IdxBtOaccBuiltIn : public concepts::Implements<ConceptIdxBt, IdxBtOaccBuiltIn<TDim, TIdx>>
            {
            public:
                //-----------------------------------------------------------------------------
                IdxBtOaccBuiltIn() = default;
                //-----------------------------------------------------------------------------
                IdxBtOaccBuiltIn(IdxBtOaccBuiltIn const &) = delete;
                //-----------------------------------------------------------------------------
                IdxBtOaccBuiltIn(IdxBtOaccBuiltIn &&) = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxBtOaccBuiltIn const & ) -> IdxBtOaccBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                auto operator=(IdxBtOaccBuiltIn &&) -> IdxBtOaccBuiltIn & = delete;
                //-----------------------------------------------------------------------------
                /*virtual*/ ~IdxBtOaccBuiltIn() = default;
            };
        }
    }

    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The OpenACC accelerator index dimension get trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct DimType<
                idx::bt::IdxBtOaccBuiltIn<TDim, TIdx>>
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
            //! The OpenACC accelerator block thread index idx type trait specialization.
            template<
                typename TDim,
                typename TIdx>
            struct IdxType<
                idx::bt::IdxBtOaccBuiltIn<TDim, TIdx>>
            {
                using type = TIdx;
            };
        }
    }
}

#endif
