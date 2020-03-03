/* Copyright 2020 Jeffrey Kelling
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef _OPENACC

#include <alpaka/atomic/Traits.hpp>
#include <alpaka/atomic/Op.hpp>

namespace alpaka
{
    namespace atomic
    {
        //#############################################################################
        //! The OpenACC accelerator's atomic ops.
        //
        //  Atomics can be used in the blocks and threads hierarchy levels.
        //  Atomics are not guaranteed to be safe between devices or grids.
        class AtomicOaccBuiltIn
        {
        public:
            //-----------------------------------------------------------------------------
            AtomicOaccBuiltIn() = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST AtomicOaccBuiltIn(AtomicOaccBuiltIn const &) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST AtomicOaccBuiltIn(AtomicOaccBuiltIn &&) = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(AtomicOaccBuiltIn const &) -> AtomicOaccBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator=(AtomicOaccBuiltIn &&) -> AtomicOaccBuiltIn & = delete;
            //-----------------------------------------------------------------------------
            /*virtual*/ ~AtomicOaccBuiltIn() = default;
        };

        namespace traits
        {

// check for OpenMP 3.1+
// "omp atomic capture" is not supported before OpenMP 3.1
#if 0

            //#############################################################################
            //! The OpenMP accelerators atomic operation: ADD
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic capture
                    {
                        old = ref;
                        ref += value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenMP accelerators atomic operation: SUB
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic update capture
                    {
                        old = ref;
                        ref -= value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenMP accelerators atomic operation: EXCH
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic update capture
                    {
                        old = ref;
                        ref = value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenMP accelerators atomic operation: AND
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic update capture
                    {
                        old = ref;
                        ref &= value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenMP accelerators atomic operation: OR
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic update capture
                    {
                        old = ref;
                        ref |= value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenMP accelerators atomic operation: XOR
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicOmpBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOmpBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic update capture
                    {
                        old = ref;
                        ref ^= value;
                    }
                    return old;
                }
            };

#endif // _OPENMP >= 201107

            //#############################################################################
            //! The OpenACC accelerators atomic operation: ADD
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Add,
                atomic::AtomicOaccBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOaccBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic update capture
                    {
                        old = ref;
                        ref += value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenACC accelerators atomic operation: SUB
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Sub,
                atomic::AtomicOaccBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOaccBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic update capture
                    {
                        old = ref;
                        ref -= value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenACC accelerators atomic operation: EXCH
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Exch,
                atomic::AtomicOaccBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOaccBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic update capture
                    {
                        old = ref;
                        ref = value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenACC accelerators atomic operation: AND
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::And,
                atomic::AtomicOaccBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOaccBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic update capture
                    {
                        old = ref;
                        ref &= value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenACC accelerators atomic operation: OR
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Or,
                atomic::AtomicOaccBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOaccBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic update capture
                    {
                        old = ref;
                        ref |= value;
                    }
                    return old;
                }
            };

            //#############################################################################
            //! The OpenACC accelerators atomic operation: XOR
            template<
                typename T,
                typename THierarchy>
            struct AtomicOp<
                op::Xor,
                atomic::AtomicOaccBuiltIn,
                T,
                THierarchy>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto atomicOp(
                    atomic::AtomicOaccBuiltIn const &,
                    T * const addr,
                    T const & value)
                -> T
                {
                    T old;
                    auto & ref(*addr);
                    // atomically update ref, but capture the original value in old
                    #pragma acc atomic update capture
                    {
                        old = ref;
                        ref ^= value;
                    }
                    return old;
                }
            };
        }
    }
}

#endif
