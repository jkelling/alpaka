/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Erik Zenker, Matthias Werner
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_BT_OMP4_ENABLED

#if _OPENMP < 201307
    #error If ALPAKA_ACC_CPU_BT_OMP4_ENABLED is set, the compiler has to support OpenMP 4.0 or higher!
#endif

#include <alpaka/queue/QueueOmp4Blocking.hpp>

#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/dev/DevOmp4.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/mem/view/Traits.hpp>

#include <alpaka/vec/Vec.hpp>
#include <alpaka/core/Assert.hpp>
#include <alpaka/core/Omp4.hpp>

#include <set>
#include <tuple>
#include <utility>

namespace alpaka
{
    namespace mem
    {
        namespace view
        {
            namespace omp4
            {
                namespace detail
                {
                    template<
                        typename TDim,
                        typename TVal,
                        typename TView,
                        template<std::size_t> class Fn,
                        std::size_t DIM = TDim::value
                        >
                    struct VecFromDimTrait
                    {
                        static_assert(DIM > 0, "DIM !> 0");
                        template<typename... TArgs>
                        static auto vecFromDimTrait(TView const &view, TArgs... args)
                            -> vec::Vec<TDim, TVal>
                        {
                            return VecFromDimTrait<TDim, TVal, TView, Fn, DIM-1>
                                ::vecFromDimTrait(view,
                                        static_cast<TVal>(Fn<DIM-1>::get(view)),
                                        std::forward<TArgs>(args)...);
                        }
                    };

                    template<
                        typename TDim,
                        typename TVal,
                        typename TView,
                        template<std::size_t> class Fn
                        >
                    struct VecFromDimTrait<
                        TDim,
                        TVal,
                        TView,
                        Fn,
                        0u
                        >
                    {
                        template<typename... TArgs>
                        static auto vecFromDimTrait(TView const &, TArgs... args)
                            -> vec::Vec<TDim, TVal>
                        {
                            return vec::Vec<TDim, TVal>(std::forward<TArgs>(args)...);
                        }
                    };

                    template<std::size_t TIdx>
                    struct MyGetExtent
                    {
                        template<typename TExtent>
                        static inline size_t get (TExtent const & extent)
                        {
                            return static_cast<size_t>(extent::getExtent<TIdx>(extent));
                        }
                    };
                    template<std::size_t TIdx>
                    struct MyGetPitch
                    {
                        template<typename TPitch>
                        static inline size_t get (TPitch const & pitch)
                        {
                            return static_cast<size_t>(view::getPitchBytes<TIdx>(pitch));
                        }
                    };

                    //#############################################################################
                    //! The Omp4 memory copy trait.
                    template<
                        typename TDim,
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyOmp4
                    {
                        static_assert(
                            !std::is_const<TViewDst>::value,
                            "The destination view can not be const!");

                        static_assert(
                            dim::Dim<TViewSrc>::value == TDim::value,
                            "The source view is required to have dimensionality TDim!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TViewSrc>::value,
                            "The source and the destination view are required to have the same dimensionality!");
                        static_assert(
                            dim::Dim<TViewDst>::value == dim::Dim<TExtent>::value,
                            "The views and the extent are required to have the same dimensionality!");
                        // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyOmp4(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),
                                m_extent(VecFromDimTrait<
                                        TDim, size_t, TExtent,
                                        MyGetExtent>::vecFromDimTrait(extent)),
                                m_dstPitchBytes(VecFromDimTrait<
                                        TDim, size_t, TViewDst,
                                        MyGetPitch>::vecFromDimTrait(viewDst)),
                                m_srcPitchBytes(VecFromDimTrait<
                                        TDim, size_t, TViewSrc,
                                        MyGetPitch>::vecFromDimTrait(viewSrc)),
#if 0
                                m_extent(alpaka::extent::getExtentVec(extent)),
                                m_dstExtent(alpaka::extent::getExtentVec(viewDst)),
                                m_srcExtent(alpaka::extent::getExtentVec(viewSrc)),
#endif
                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            const auto dstExtent(VecFromDimTrait<
                                    TDim, size_t, TViewDst,
                                    MyGetExtent>::vecFromDimTrait(viewDst));
                            const auto srcExtent(VecFromDimTrait<
                                    TDim, size_t, TViewSrc,
                                    MyGetExtent>::vecFromDimTrait(viewSrc));
                            for(auto i = static_cast<decltype(TDim::value)>(0u); i < TDim::value; ++i)
                            {
                                ALPAKA_ASSERT(m_extent[i] <= dstExtent[i]);
                                ALPAKA_ASSERT(m_extent[i] <= srcExtent[i]);
                            }
                            std::cout << "TaskCopyOmp4<" << TDim::value << ",...>::ctor\tdstExtent="
                            << alpaka::extent::getExtentVec(viewDst) << ", m_dstExtent=" << dstExtent << std::endl;
#endif
                            // std::cout << "TaskCopyOmp4<" << TDim::value << ",...>::ctor\tdstExtent="
                            // << alpaka::extent::getExtentVec(viewDst) << ", m_dstExtent=" << m_dstExtent << std::endl;
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << __func__
                                << " ddev: " << m_iDstDevice
                                << " ew: " << m_extent
                                // << " dw: " << m_dstExtent
                                << " dptr: " << m_dstMemNative
                                << " sdev: " << m_iSrcDevice
                                // << " sw: " << m_srcExtent
                                << " sptr: " << m_srcMemNative
                                << std::endl;
                        }
#endif
                        int m_iDstDevice;
                        int m_iSrcDevice;
                        vec::Vec<TDim, size_t> m_extent;
                        // vec::Vec<TDim, size_t> m_dstExtent;
                        // vec::Vec<TDim, size_t> m_srcExtent;
                        vec::Vec<TDim, size_t> m_dstPitchBytes;
                        vec::Vec<TDim, size_t> m_srcPitchBytes;
                        void * m_dstMemNative;
                        void const * m_srcMemNative;

                        //-----------------------------------------------------------------------------
                        //! Executes the kernel function object.
                        ALPAKA_FN_HOST auto operator()() const
                        -> void
                        {
                            ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            printDebug();
#endif
                            constexpr auto lastDim = TDim::value - 1;

                            if(m_extent.prod() > 0)
                            {
                                // offsets == 0 by ptr shift (?)
                                auto dstOffset(vec::Vec<TDim, size_t>::zeros());
                                auto srcOffset(vec::Vec<TDim, size_t>::zeros());

                                auto dstExtentFull(vec::Vec<TDim, size_t>::zeros());
                                auto srcExtentFull(vec::Vec<TDim, size_t>::zeros());

                                const size_t elementSize =
                                    ( m_dstPitchBytes[0]%sizeof(elem::Elem<TViewDst>) || m_srcPitchBytes[0]%sizeof(elem::Elem<TViewDst>) )
                                    ? 1 : sizeof(elem::Elem<TViewDst>);

                                dstExtentFull[lastDim] = m_dstPitchBytes[lastDim]/elementSize;
                                srcExtentFull[lastDim] = m_srcPitchBytes[lastDim]/elementSize;
                                for(int i = lastDim - 1; i >= 0; --i)
                                {
                                    dstExtentFull[i] = m_dstPitchBytes[i]/m_dstPitchBytes[i+1];
                                    srcExtentFull[i] = m_srcPitchBytes[i]/m_srcPitchBytes[i+1];
                                }

                                std::cout << "copy " << TDim::value << "d\textent=" << extent
                                    << "\tdstExtentFull=" << dstExtentFull << " (p " << dstPitchBytes
                                    << " )\tsrcExtentFull=" << srcExtentFull << " (p " << srcPitchBytes
                                    << " )\telementSize=" << elementSize << std::endl;

                                ALPAKA_OMP4_CHECK(
                                    omp_target_memcpy_rect(
                                        m_dstMemNative, const_cast<void*>(m_srcMemNative),
                                        sizeof(elem::Elem<TViewDst>),
                                        TDim::value,
                                        reinterpret_cast<size_t const *>(&m_extent),
                                        reinterpret_cast<size_t const *>(&dstOffset),
                                        reinterpret_cast<size_t const *>(&srcOffset),
                                        reinterpret_cast<size_t const *>(&dstExtentFull),
                                        reinterpret_cast<size_t const *>(&srcExtentFull),
                                        m_iDstDevice, m_iSrcDevice));
                            }
                        }
                    };

                    //#############################################################################
                    //! The Omp4 memory copy trait.
                    template<
                        typename TViewDst,
                        typename TViewSrc,
                        typename TExtent>
                    struct TaskCopyOmp4<
                        dim::DimInt<1>,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        static_assert(
                            !std::is_const<TViewDst>::value,
                            "The destination view can not be const!");

                        static_assert(
                            dim::Dim<TViewSrc>::value == 1,
                            "The source view is required to have dimensionality 1!");
                        static_assert(
                            dim::Dim<TViewDst>::value == 1,
                            "The source view is required to have dimensionality 1!");
                        static_assert(
                            dim::Dim<TExtent>::value == 1,
                            "The extent is required to have dimensionality 1!");
                        // TODO: Maybe check for Idx of TViewDst and TViewSrc to have greater or equal range than TExtent.
                        static_assert(
                            std::is_same<elem::Elem<TViewDst>, typename std::remove_const<elem::Elem<TViewSrc>>::type>::value,
                            "The source and the destination view are required to have the same element type!");

                        using Idx = idx::Idx<TExtent>;

                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST TaskCopyOmp4(
                            TViewDst & viewDst,
                            TViewSrc const & viewSrc,
                            TExtent const & extent,
                            int const & iDstDevice,
                            int const & iSrcDevice) :
                                m_iDstDevice(iDstDevice),
                                m_iSrcDevice(iSrcDevice),
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                                m_extentWidth(extent::getWidth(extent)),
                                m_dstWidth(static_cast<Idx>(extent::getWidth(viewDst))),
                                m_srcWidth(static_cast<Idx>(extent::getWidth(viewSrc))),
#endif
                                m_extentWidthBytes(extent::getWidth(extent) * static_cast<Idx>(sizeof(elem::Elem<TViewDst>))),
                                m_dstMemNative(reinterpret_cast<void *>(mem::view::getPtrNative(viewDst))),
                                m_srcMemNative(reinterpret_cast<void const *>(mem::view::getPtrNative(viewSrc)))
                        {
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            ALPAKA_ASSERT(m_extentWidth <= m_dstWidth);
                            ALPAKA_ASSERT(m_extentWidth <= m_srcWidth);
#endif
                        }

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        //-----------------------------------------------------------------------------
                        ALPAKA_FN_HOST auto printDebug() const
                        -> void
                        {
                            std::cout << __func__
                                << " ddev: " << m_iDstDevice
                                << " ew: " << m_extentWidth
                                << " ewb: " << m_extentWidthBytes
                                << " dw: " << m_dstWidth
                                << " dptr: " << m_dstMemNative
                                << " sdev: " << m_iSrcDevice
                                << " sw: " << m_srcWidth
                                << " sptr: " << m_srcMemNative
                                << std::endl;
                        }
#endif
                        int m_iDstDevice;
                        int m_iSrcDevice;
#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                        Idx m_extentWidth;
                        Idx m_dstWidth;
                        Idx m_srcWidth;
#endif
                        Idx m_extentWidthBytes;
                        void * m_dstMemNative;
                        void const * m_srcMemNative;

                        //-----------------------------------------------------------------------------
                        //! Executes the kernel function object.
                        ALPAKA_FN_HOST auto operator()() const
                        -> void
                        {
                            ALPAKA_DEBUG_FULL_LOG_SCOPE;

#if ALPAKA_DEBUG >= ALPAKA_DEBUG_FULL
                            printDebug();
#endif
                            if(m_extentWidthBytes == 0)
                            {
                                return;
                            }

                            std::cout << "Omp4 copy " << srcNativePtr<< " (" << iSrcDev << ") to "
                                << dstNativePtr << " (" << iDstDev << ")"<< std::endl;
                            ALPAKA_OMP4_CHECK(
                                omp_target_memcpy(
                                    m_dstMemNative, const_cast<void*>(m_srcMemNative), m_extentWidthBytes,
                                    0,0, m_iDstDevice, m_iSrcDevice));
                        }
                    };
                }
            }

            //-----------------------------------------------------------------------------
            // Trait specializations for CreateTaskCopy.
            namespace traits
            {
                namespace omp4
                {
                    namespace detail
                    {
                        //#############################################################################
                        //! The Omp4 memory copy task creation trait detail.
                        template<
                            typename TDim,
                            typename TDevDst,
                            typename TDevSrc>
                        struct CreateTaskCopyImpl
                        {
                            //-----------------------------------------------------------------------------
                            template<
                                typename TExtent,
                                typename TViewSrc,
                                typename TViewDst>
                            ALPAKA_FN_HOST static auto createTaskCopy(
                                TViewDst & viewDst,
                                TViewSrc const & viewSrc,
                                TExtent const & extent,
                                int iDeviceDst = 0,
                                int iDeviceSrc = 0
                                )
                            -> mem::view::omp4::detail::TaskCopyOmp4<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>
                            {
                                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                                return
                                    mem::view::omp4::detail::TaskCopyOmp4<
                                        TDim,
                                        TViewDst,
                                        TViewSrc,
                                        TExtent>(
                                            viewDst,
                                            viewSrc,
                                            extent,
                                            iDeviceDst,
                                            iDeviceSrc);
                            }
                        };
                    }
                }

                //#############################################################################
                //! The CPU to Omp4 memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevOmp4,
                    dev::DevCpu>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::omp4::detail::TaskCopyOmp4<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        auto * ptr = mem::view::getPtrNative(viewSrc);
                        std::cout << "copy from host [0,10,100]:\t" << ptr[0] << '\t' << ptr[10] << '\t' << ptr[100] << std::endl;

                        return
                            mem::view::omp4::detail::TaskCopyOmp4<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    dev::getDev(viewDst).m_spDevOmp4Impl->iDevice(),
                                    omp_get_initial_device()
                                    );
                    }
                };

                //#############################################################################
                //! The Omp4 to CPU memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevCpu,
                    dev::DevOmp4>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::omp4::detail::TaskCopyOmp4<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return
                            mem::view::omp4::detail::TaskCopyOmp4<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    omp_get_initial_device(),
                                    dev::getDev(viewSrc).m_spDevOmp4Impl->iDevice()
                                    );
                    }
                };

                //#############################################################################
                //! The Omp4 to Omp4 memory copy trait specialization.
                template<
                    typename TDim>
                struct CreateTaskCopy<
                    TDim,
                    dev::DevOmp4,
                    dev::DevOmp4>
                {
                    //-----------------------------------------------------------------------------
                    template<
                        typename TExtent,
                        typename TViewSrc,
                        typename TViewDst>
                    ALPAKA_FN_HOST static auto createTaskCopy(
                        TViewDst & viewDst,
                        TViewSrc const & viewSrc,
                        TExtent const & extent)
                    -> mem::view::omp4::detail::TaskCopyOmp4<
                        TDim,
                        TViewDst,
                        TViewSrc,
                        TExtent>
                    {
                        ALPAKA_DEBUG_FULL_LOG_SCOPE;

                        return
                            mem::view::omp4::detail::TaskCopyOmp4<
                                TDim,
                                TViewDst,
                                TViewSrc,
                                TExtent>(
                                    viewDst,
                                    viewSrc,
                                    extent,
                                    dev::getDev(viewDst).m_spDevOmp4Impl->iDevice(),
                                    dev::getDev(viewSrc).m_spDevOmp4Impl->iDevice()
                                    );
                    }
                };
            }
        }
    }
}

#endif
