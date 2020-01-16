/* Copyright 2019 Benjamin Worpitz, Erik Zenker, René Widera
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_ANY_BT_OMP5_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_ANY_BT_OMP5_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/block/sync/BlockSyncBarrierOmp.hpp>

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
                class BlockSharedMemStOmp5 : public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStOmp5>
                {
                    mutable unsigned int m_allocdBytes = 0;
                    mutable char* m_mem;
#ifdef SPEC_FAKE_OMP_TARGET_CPU
                    sync::BlockSyncBarrierOmp& m_sync;
#endif

                public:
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOmp5(char* mem
#ifdef SPEC_FAKE_OMP_TARGET_CPU
                            , sync::BlockSyncBarrierOmp& sync
#endif
                            ) : m_mem(mem)
#ifdef SPEC_FAKE_OMP_TARGET_CPU
                                , m_sync(sync)
#endif
                    {}
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOmp5(BlockSharedMemStOmp5 const &) = delete;
                    //-----------------------------------------------------------------------------
                    BlockSharedMemStOmp5(BlockSharedMemStOmp5 &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemStOmp5 const &) -> BlockSharedMemStOmp5 & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(BlockSharedMemStOmp5 &&) -> BlockSharedMemStOmp5 & = delete;
                    //-----------------------------------------------------------------------------
                    /*virtual*/ ~BlockSharedMemStOmp5() = default;

                    template<class T>
                    T& alloc() const
                    {
#ifndef SPEC_FAKE_OMP_TARGET_CPU
                        #pragma omp barrier
                        if(omp_get_thread_num() == 0)
#else
                        sync::syncBlockThreads(m_sync);
                        if(m_sync.m_masterThread == std::this_thread::get_id())
#endif
                        {
                            char* buf = &m_mem[m_allocdBytes];
                            new (buf) T();
                            m_allocdBytes += sizeof(T);
                            // printf("BlockSharedMemStOmp4::alloc (master) m_allocdBytes=%d\n", m_allocdBytes);

                        }
#ifndef SPEC_FAKE_OMP_TARGET_CPU
                        #pragma omp barrier
#else
                        sync::syncBlockThreads(m_sync);
#endif
                        return *reinterpret_cast<T*>(&m_mem[m_allocdBytes-sizeof(T)]);
                    }
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T,
                        std::size_t TuniqueId>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        BlockSharedMemStOmp5>
                    {
                        //-----------------------------------------------------------------------------
                        static auto allocVar(
                            block::shared::st::BlockSharedMemStOmp5 const &smem)
                        -> T &
                        {
                            return smem.alloc<T>();
                        }
                    };
                    //#############################################################################
                    template<>
                    struct FreeMem<
                        BlockSharedMemStOmp5>
                    {
                        //-----------------------------------------------------------------------------
                        static auto freeMem(
                            block::shared::st::BlockSharedMemStOmp5 const &)
                        -> void
                        {
                            // Nothing to do. CUDA block shared memory is automatically freed when all threads left the block.
                        }
                    };
                }
            }
        }
    }
}

#endif
