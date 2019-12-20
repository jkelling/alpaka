/* Copyright 2019 Benjamin Worpitz
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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/Properties.hpp>

#include <alpaka/core/Omp4.hpp>

#include <alpaka/queue/IGenericQueue.hpp>
#include <alpaka/queue/QueueGenericNonBlocking.hpp>
#include <alpaka/queue/QueueGenericBlocking.hpp>

namespace alpaka
{
    namespace dev
    {
        class DevOmp4;
    }
    namespace pltf
    {
        namespace traits
        {
            template<
                typename TPltf,
                typename TSfinae>
            struct GetDevByIdx;
        }
        class PltfOmp4;
    }

    namespace dev
    {
        namespace omp4
        {
            namespace detail
            {
                //#############################################################################
                //! The Omp4 device implementation.
                class DevOmp4Impl
                {
                private:

                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto GetAllQueueImpls(
                        std::vector<std::weak_ptr<queue::IGenericQueue<DevOmp4>>> & queues) const
                    -> std::vector<std::shared_ptr<queue::IGenericQueue<DevOmp4>>>
                    {
                        std::vector<std::shared_ptr<queue::IGenericQueue<DevOmp4>>> vspQueues;

                        std::lock_guard<std::mutex> lk(m_Mutex);

                        for(auto it = queues.begin(); it != queues.end();)
                        {
                            auto spQueue(it->lock());
                            if(spQueue)
                            {
                                vspQueues.emplace_back(std::move(spQueue));
                                ++it;
                            }
                            else
                            {
                                it = queues.erase(it);
                            }
                        }
                        return vspQueues;
                    }

                public:
                    //-----------------------------------------------------------------------------
                    DevOmp4Impl() = default;
                    //-----------------------------------------------------------------------------
                    DevOmp4Impl(DevOmp4Impl const &) = delete;
                    //-----------------------------------------------------------------------------
                    DevOmp4Impl(DevOmp4Impl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevOmp4Impl const &) -> DevOmp4Impl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevOmp4Impl &&) -> DevOmp4Impl & = delete;
                    //-----------------------------------------------------------------------------
                    ~DevOmp4Impl() = default;

                    ALPAKA_FN_HOST auto GetAllQueues() const
                    -> std::vector<std::shared_ptr<queue::IGenericQueue<DevOmp4>>>
                    {
                        return GetAllQueueImpls(m_queues);
                    }

                    //-----------------------------------------------------------------------------
                    //! Registers the given queue on this device.
                    //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
                    ALPAKA_FN_HOST auto RegisterQueue(std::shared_ptr<queue::IGenericQueue<DevOmp4>> spQueue)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Register this queue on the device.
                        m_queues.push_back(spQueue);
                    }

                    int iDevice() const {return m_iDevice;}

                private:
                    std::mutex mutable m_Mutex;
                    std::vector<std::weak_ptr<queue::IGenericQueue<DevOmp4>>> mutable m_queues;
                    int m_iDevice = 0;
                };
            }
        }
        //#############################################################################
        //! The Omp4 device handle.
        class DevOmp4 : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, DevOmp4>
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfOmp4>;

        protected:
            //-----------------------------------------------------------------------------
            DevOmp4() :
                m_spDevOmp4Impl(std::make_shared<omp4::detail::DevOmp4Impl>())
            {}
        public:
            //-----------------------------------------------------------------------------
            DevOmp4(DevOmp4 const &) = default;
            //-----------------------------------------------------------------------------
            DevOmp4(DevOmp4 &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevOmp4 const &) -> DevOmp4 & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevOmp4 &&) -> DevOmp4 & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevOmp4 const & rhs) const
            -> bool
            {
                return m_spDevOmp4Impl->iDevice() == rhs.m_spDevOmp4Impl->iDevice();
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevOmp4 const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~DevOmp4() = default;

        public:
            std::shared_ptr<omp4::detail::DevOmp4Impl> m_spDevOmp4Impl;
        };
    }

    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device name get trait specialization.
            template<>
            struct GetName<
                dev::DevOmp4>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevOmp4 const &)
                -> std::string
                {
                    return std::string("OMP4 target");
                }
            };

            //#############################################################################
            //! The CUDA RT device available memory get trait specialization.
            template<>
            struct GetMemBytes<
                dev::DevOmp4>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevOmp4 const & dev)
                -> std::size_t
                {
                    alpaka::ignore_unused(dev); //! \TODO
                    // std::size_t freeInternal(0u);
                    std::size_t totalInternal(6ull<<30); //! \TODO

                    return totalInternal;
                }
            };

            //#############################################################################
            //! The CUDA RT device free memory get trait specialization.
            template<>
            struct GetFreeMemBytes<
                dev::DevOmp4>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevOmp4 const & dev)
                -> std::size_t
                {
                    alpaka::ignore_unused(dev); //! \todo query device
                    std::size_t freeInternal((6ull<<30));
                    // std::size_t totalInternal(0u);

                    return freeInternal;
                }
            };

            //#############################################################################
            //! The CUDA RT device reset trait specialization.
            template<>
            struct Reset<
                dev::DevOmp4>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevOmp4 const & dev)
                -> void
                {
                    alpaka::ignore_unused(dev); //! \TODO
                }
            };

            //#############################################################################
            //! The Omp4 device register queue trait specialization.
            template<>
            struct RegisterQueue<
                dev::DevOmp4>
            {
                //-----------------------------------------------------------------------------
                template<
                    template<typename TDev> class TQueue>
                ALPAKA_FN_HOST static auto registerQueue(
                    dev::DevOmp4 const & dev,
                    TQueue<dev::DevOmp4> const & queue)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    dev.m_spDevOmp4Impl->RegisterQueue(queue.m_spQueueImpl);
                }
            };

            //#############################################################################
            //! The Omp4 device get all queues trait specialization.
            template<>
            struct GetAllQueues<
                dev::DevOmp4>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getAllQueues(
                    dev::DevOmp4 const & dev)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
                -> decltype(dev.m_spDevOmp4Impl->GetAllQueues())
#endif
                {
                    return dev.m_spDevOmp4Impl->GetAllQueues();
                }
            };
        }
    }
    namespace mem
    {
        namespace buf
        {
            template<
                typename TElem,
                typename TDim,
                typename TIdx>
            class BufOmp4;

            namespace traits
            {
                //#############################################################################
                //! The CUDA RT device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct BufType<
                    dev::DevOmp4,
                    TElem,
                    TDim,
                    TIdx>
                {
                    using type = mem::buf::BufOmp4<TElem, TDim, TIdx>;
                };
            }
        }
    }
    namespace pltf
    {
        namespace traits
        {
            //#############################################################################
            //! The CUDA RT device platform type trait specialization.
            template<>
            struct PltfType<
                dev::DevOmp4>
            {
                using type = pltf::PltfOmp4;
            };
        }
    }
    namespace queue
    {
        using QueueOmp4NonBlocking = QueueGenericNonBlocking<dev::DevOmp4>;
        using QueueOmp4Blocking = QueueGenericBlocking<dev::DevOmp4>;

        namespace traits
        {
            template<>
            struct QueueType<
                dev::DevOmp4,
                queue::Blocking
            >
            {
                using type = queue::QueueOmp4Blocking;
            };

            template<>
            struct QueueType<
                dev::DevOmp4,
                queue::NonBlocking
            >
            {
                using type = queue::QueueOmp4NonBlocking;
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The thread Omp4 device wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
            template<>
            struct CurrentThreadWaitFor<
                dev::DevOmp4>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevOmp4 const & dev)
                -> void
                {
                    ALPAKA_DEBUG_FULL_LOG_SCOPE;

                    generic::currentThreadWaitForDevice(dev);
// #pragma omp taskwait
                }
            };
        }
    }
}

#endif
