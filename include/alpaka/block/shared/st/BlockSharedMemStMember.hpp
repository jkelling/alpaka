/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/block/shared/st/Traits.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Vectorize.hpp>

#include <type_traits>
#include <cstdint>

namespace alpaka
{
    namespace block
    {
        namespace shared
        {
            namespace st
            {
                namespace detail
                {
                    //#############################################################################
                    //! Implementation of static block shared memory provider.
                    template<unsigned int TDataAlignBytes = core::vectorization::defaultAlignment>
                    class BlockSharedMemStMemberImpl
                    {
                    public:
                        //-----------------------------------------------------------------------------
#ifndef NDEBUG
                        BlockSharedMemStMemberImpl(uint8_t* mem, unsigned int capacity) : m_mem(mem), m_capacity(capacity) {}
#else
                        BlockSharedMemStMemberImpl(uint8_t* mem, unsigned int) : m_mem(mem) {}
#endif
                        //-----------------------------------------------------------------------------
                        BlockSharedMemStMemberImpl(BlockSharedMemStMemberImpl const &) = delete;
                        //-----------------------------------------------------------------------------
                        BlockSharedMemStMemberImpl(BlockSharedMemStMemberImpl &&) = delete;
                        //-----------------------------------------------------------------------------
                        auto operator=(BlockSharedMemStMemberImpl const &) -> BlockSharedMemStMemberImpl & = delete;
                        //-----------------------------------------------------------------------------
                        auto operator=(BlockSharedMemStMemberImpl &&) -> BlockSharedMemStMemberImpl & = delete;
                        //-----------------------------------------------------------------------------
                        /*virtual*/ ~BlockSharedMemStMemberImpl() = default;

                        template <typename T>
                        void alloc() const
                        {
                            uint8_t* buf = &m_mem[m_allocdBytes];
                            new (buf) T();
                            m_allocdBytes += alignPitch<T>();
#if (defined ALPAKA_DEBUG_OFFLOAD_ASSUME_HOST) && (! defined NDEBUG)
                            ALPAKA_ASSERT(m_allocdBytes < m_capacity);
#endif
                        }

#if BOOST_COMP_GNUC
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases required alignment of target type"
#endif
                        template <typename T>
                        T& getLatestVar() const
                        {
                           return *reinterpret_cast<T*>(&m_mem[m_allocdBytes-alignPitch<T>()]);
                        }
#if BOOST_COMP_GNUC
    #pragma GCC diagnostic pop
#endif

                        void free() const
                        {
                            m_allocdBytes = 0u;
                        }

                    private:
                        mutable unsigned int m_allocdBytes = 0;
                        mutable uint8_t* m_mem;
#ifndef NDEBUG
                        const unsigned int m_capacity;
#endif

                        template<typename T>
                        static constexpr unsigned int alignPitch()
                        {
                            return (static_cast<unsigned int>(sizeof(T))/TDataAlignBytes
                                + static_cast<unsigned int>(sizeof(T))%TDataAlignBytes>0)*TDataAlignBytes;
                        }
                    };
                }
                //#############################################################################
                //! Static block shared memory provider using a pointer to
                //! externally allocated fixed-size memory, likely provided by
                //! BlockSharedMemDynMember.
                template<unsigned int TDataAlignBytes = core::vectorization::defaultAlignment>
                class BlockSharedMemStMember :
                    public detail::BlockSharedMemStMemberImpl<TDataAlignBytes>,
                    public concepts::Implements<ConceptBlockSharedSt, BlockSharedMemStMember<TDataAlignBytes>>
                {
                public:
                    using detail::BlockSharedMemStMemberImpl<TDataAlignBytes>::BlockSharedMemStMemberImpl;
                };

                namespace traits
                {
                    //#############################################################################
                    template<
                        typename T,
                        unsigned int TDataAlignBytes,
                        std::size_t TuniqueId>
                    struct AllocVar<
                        T,
                        TuniqueId,
                        BlockSharedMemStMember<TDataAlignBytes>>
                    {
                        //-----------------------------------------------------------------------------
                        static auto allocVar(
                            block::shared::st::BlockSharedMemStMember<TDataAlignBytes> const &smem)
                        -> T &
                        {
                            smem.template alloc<T>();
                            return smem.template getLatestVar<T>();
                        }
                    };
                    //#############################################################################
                    template<
                        unsigned int TDataAlignBytes>
                    struct FreeMem<
                        BlockSharedMemStMember<TDataAlignBytes>>
                    {
                        //-----------------------------------------------------------------------------
                        static auto freeMem(
                            block::shared::st::BlockSharedMemStMember<TDataAlignBytes> const &mem)
                        -> void
                        {
                            mem.free();
                        }
                    };
                }
            }
        }
    }
}
