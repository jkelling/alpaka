/* Copyright 2019 Benjamin Worpitz, Erik Zenker, René Widera
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

#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/block/sync/Traits.hpp>

#include <type_traits>
#include <cstdint>
#include <omp.h>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace st
            {
                //#############################################################################
                //! The GPU CUDA block shared memory allocator.
                class BlockSharedMemStOacc : public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStOacc>
                {
                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOacc() = default;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOacc(BlockSharedMemStOacc const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOacc(BlockSharedMemStOacc &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemStOacc const &) -> BlockSharedMemStOacc & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemStOacc &&) -> BlockSharedMemStOacc & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemStOacc() = default;

                    template<
                        typename TAcc>
                    class BlockShared : public concepts::Implements<ConceptBlockSharedSt, BlockShared<TAcc>>
                    {
                        mutable unsigned int m_allocdBytes = 0;
                        mutable char* m_mem;

                        public:

                        BlockShared(char* mem) : m_mem(mem) {}
                        //-----------------------------------------------------------------------------
                        BlockShared(BlockSharedMemStOacc const &) = delete;
                        //-----------------------------------------------------------------------------
                        BlockShared(BlockSharedMemStOacc &&) = delete;
                        //-----------------------------------------------------------------------------
                        auto operator=(BlockShared const &) -> BlockShared & = delete;
                        //-----------------------------------------------------------------------------
                        auto operator=(BlockShared &&) -> BlockShared & = delete;
                        //-----------------------------------------------------------------------------
                        /*virtual*/ ~BlockShared() = default;

                        template<typename T>
                        T& alloc() const
                        {
                           block::sync::traits::SyncBlockThreads<TAcc>::masterOpBlockThreads(
                               static_cast<TAcc>(*this),
                               [this](){
                                   char* buf = &m_mem[m_allocdBytes];
                                   new (buf) T();
                                   m_allocdBytes += sizeof(T);
                                   }
                               );
                           return *reinterpret_cast<T*>(&m_mem[m_allocdBytes-sizeof(T)]);
                        }
                    };
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename TAcc>
                    struct FreeMem<
                        BlockSharedMemStOacc::BlockShared<TAcc>>
                    {
                        //-----------------------------------------------------------------------------
                        static auto freeMem(
                            BlockSharedMemStOacc::BlockShared<TAcc> const &)
                        -> void
                        {
                            // Nothing to do. Block shared memory is automatically freed when all threads left the block.
                        }
                    };
                }
            }
        }
    }
}

#endif
