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
                template<
                    typename TAcc>
                class BlockSharedMemStOacc : public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStOacc<TAcc>>
                {
                    mutable unsigned int m_allocdBytes = 0;
                    mutable char* m_mem;

                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOacc(char* mem) : m_mem(mem) {}
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

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T,
                        typename TAcc,
                        std::size_t TuniqueId>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        BlockSharedMemStOacc<TAcc>>
                    {
                        //-----------------------------------------------------------------------------
                        static auto allocVar(
                            block::shared::st::BlockSharedMemStOacc<TAcc> const &smem)
                        -> T &
                        {
                            return smem.template alloc<T>();
                        }
                    };
                    //#############################################################################
                    template<
                        typename TAcc>
                    struct FreeMem<
                        BlockSharedMemStOacc<TAcc>>
                    {
                        //-----------------------------------------------------------------------------
                        static auto freeMem(
                            block::shared::st::BlockSharedMemStOacc<TAcc> const &)
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
