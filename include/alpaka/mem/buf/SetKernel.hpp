/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            //#############################################################################
            //! any device ND memory set kernel.
            class MemSetKernel
            {
            public:
                //-----------------------------------------------------------------------------
                //! The kernel entry point.
                //!
                //! \tparam TAcc The accelerator environment to be executed on.
                //! \tparam TElem element type.
                //! \tparam TExtent extent type.
                //! \param acc The accelerator to be executed on.
                //! \param val value to set.
                //! \param dst target mem ptr.
                //! \param extent area to set.
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TAcc,
                    typename TElem,
                    typename TExtent,
                    typename TPitch>
                ALPAKA_FN_ACC auto operator()(
                    TAcc const & acc,
                    TElem const val,
                    TElem * dst,
                    TExtent extent,
                    TPitch pitch) const
                -> void
                {
                    using Idx = typename idx::traits::IdxType<TExtent>::type;
                    auto const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc));
                    auto const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc));
                    auto const idxThreadFirstElem = idx::getIdxThreadFirstElem(acc, gridThreadIdx, threadElemExtent);
                    auto idx = idx::mapIdxPitch<1u, dim::Dim<TAcc>::value>(idxThreadFirstElem, pitch)[0];
                    constexpr auto lastDim = dim::Dim<TAcc>::value - 1;
                    const auto lastIdx = idx +
                        std::min(threadElemExtent[lastDim], static_cast<Idx>(extent[lastDim]-idxThreadFirstElem[lastDim]));

                    if ([&idxThreadFirstElem, &extent](){
                            for(auto i = 0u; i < dim::Dim<TAcc>::value; ++i)
                                if(idxThreadFirstElem[i] >= extent[i])
                                    return false;
                            return true;
                        }())
                    {
                        // assuming elements = {1,1,...,1,n}
                        for(; idx<lastIdx; ++idx)
                        {
                            dst[idx] = val;
                        }
                    }
                }
            };
        }
    }
}
