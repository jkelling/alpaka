/* Copyright 2019 Benjamin Worpitz, René Widera
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

#include <alpaka/block/shared/dyn/Traits.hpp>

#include <type_traits>
#include <array>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace dyn
            {
                //#############################################################################
                //! The GPU CUDA block shared memory allocator.
                class BlockSharedMemDynOacc : public concepts::Implements<ConceptBlockSharedDyn, BlockSharedMemDynOacc>
                {
                    mutable std::array<char, 30<<10> m_mem; // ! static 30kB
                    std::size_t m_dynSize;

                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOacc(size_t sizeBytes) : m_dynSize(sizeBytes) {}
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOacc(BlockSharedMemDynOacc const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemDynOacc(BlockSharedMemDynOacc &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynOacc const &) -> BlockSharedMemDynOacc & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemDynOacc &&) -> BlockSharedMemDynOacc & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemDynOacc() = default;

                    char* dynMemBegin() const {return m_mem.data();}
                    char* staticMemBegin() const {return m_mem.data()+m_dynSize;}
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T>
                    struct GetMem<
                        T,
                        BlockSharedMemDynOacc>
                    {
                        //-----------------------------------------------------------------------------
                        static auto getMem(
                            block::shared::dyn::BlockSharedMemDynOacc const &mem)
                        -> T *
                        {
                            return reinterpret_cast<T*>(mem.dynMemBegin());
                        }
                    };
                }
            }
        }
    }
}

#endif
