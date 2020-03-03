/* Copyright 2019 Benjamin Worpitz
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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/Properties.hpp>

#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>
#include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>
#include <alpaka/queue/QueueGenericThreadsBlocking.hpp>

#include <openacc.h>

namespace alpaka
{
    namespace dev
    {
        class DevOacc;
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
        class PltfOacc;
    }

    namespace dev
    {
        namespace omp4
        {
            namespace detail
            {
                //#############################################################################
                //! The Oacc device implementation.
                class DevOaccImpl
                {
                public:
                    //-----------------------------------------------------------------------------
                    DevOaccImpl() :
                        m_deviceType(::acc_get_device_type()),
                        m_iDevice(::acc_get_device_num(m_deviceType))
                    {}
                    // DevOaccImpl(int iDevice) : m_iDevice(iDevice) {}
                    //-----------------------------------------------------------------------------
                    DevOaccImpl(DevOaccImpl const &) = delete;
                    //-----------------------------------------------------------------------------
                    DevOaccImpl(DevOaccImpl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevOaccImpl const &) -> DevOaccImpl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevOaccImpl &&) -> DevOaccImpl & = delete;
                    //-----------------------------------------------------------------------------
                    ~DevOaccImpl() = default;

                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto getAllExistingQueues() const
                    -> std::vector<std::shared_ptr<queue::IGenericThreadsQueue<DevOacc>>>
                    {
                        std::vector<std::shared_ptr<queue::IGenericThreadsQueue<DevOacc>>> vspQueues;

                        std::lock_guard<std::mutex> lk(m_Mutex);
                        vspQueues.reserve(m_queues.size());

                        for(auto it = m_queues.begin(); it != m_queues.end();)
                        {
                            auto spQueue(it->lock());
                            if(spQueue)
                            {
                                vspQueues.emplace_back(std::move(spQueue));
                                ++it;
                            }
                            else
                            {
                                it = m_queues.erase(it);
                            }
                        }
                        return vspQueues;
                    }

                    //-----------------------------------------------------------------------------
                    //! Registers the given queue on this device.
                    //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
                    ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<queue::IGenericThreadsQueue<DevOacc>> spQueue)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Register this queue on the device.
                        m_queues.push_back(spQueue);
                    }

                    int iDevice() const {return m_iDevice;}
                    acc_device_t deviceType() const {return m_deviceType;}

                private:
                    std::mutex mutable m_Mutex;
                    std::vector<std::weak_ptr<queue::IGenericThreadsQueue<DevOacc>>> mutable m_queues;
                    acc_device_t m_deviceType;
                    int m_iDevice;
                };
            }
        }
        //#############################################################################
        //! The Oacc device handle.
        class DevOacc : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, DevOacc>
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfOacc>;

        protected:
            //-----------------------------------------------------------------------------
            DevOacc(int iDevice) :
                m_spDevOaccImpl(std::make_shared<omp4::detail::DevOaccImpl>())
            {}
        public:
            //-----------------------------------------------------------------------------
            DevOacc(DevOacc const &) = default;
            //-----------------------------------------------------------------------------
            DevOacc(DevOacc &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevOacc const &) -> DevOacc & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevOacc &&) -> DevOacc & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevOacc const & rhs) const
            -> bool
            {
                return m_spDevOaccImpl->iDevice() == rhs.m_spDevOaccImpl->iDevice();
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevOacc const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~DevOacc() = default;
            int iDevice() const {return m_spDevOaccImpl->iDevice();}
            void makeCurrent() const
            {
                acc_set_device_num(m_spDevOaccImpl->iDevice(), m_spDevOaccImpl->deviceType());
            }

            ALPAKA_FN_HOST auto getAllQueues() const
            -> std::vector<std::shared_ptr<queue::IGenericThreadsQueue<DevOacc>>>
            {
                return m_spDevOaccImpl->getAllExistingQueues();
            }

            //-----------------------------------------------------------------------------
            //! Registers the given queue on this device.
            //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
            ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<queue::IGenericThreadsQueue<DevOacc>> spQueue) const
            -> void
            {
                m_spDevOaccImpl->registerQueue(spQueue);
            }

        public:
            std::shared_ptr<omp4::detail::DevOaccImpl> m_spDevOaccImpl;
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
                dev::DevOacc>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevOacc const &)
                -> std::string
                {
                    return std::string("OMP4 target");
                }
            };

            //#############################################################################
            //! The CUDA RT device available memory get trait specialization.
            template<>
            struct GetMemBytes<
                dev::DevOacc>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevOacc const & dev)
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
                dev::DevOacc>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevOacc const & dev)
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
                dev::DevOacc>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevOacc const & dev)
                -> void
                {
                    alpaka::ignore_unused(dev); //! \TODO
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
            class BufOacc;

            namespace traits
            {
                //#############################################################################
                //! The CUDA RT device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct BufType<
                    dev::DevOacc,
                    TElem,
                    TDim,
                    TIdx>
                {
                    using type = mem::buf::BufOacc<TElem, TDim, TIdx>;
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
                dev::DevOacc>
            {
                using type = pltf::PltfOacc;
            };
        }
    }
    namespace queue
    {
        using QueueOaccNonBlocking = QueueGenericThreadsNonBlocking<dev::DevOacc>;
        using QueueOaccBlocking = QueueGenericThreadsBlocking<dev::DevOacc>;

        namespace traits
        {
            template<>
            struct QueueType<
                dev::DevOacc,
                queue::Blocking
            >
            {
                using type = queue::QueueOaccBlocking;
            };

            template<>
            struct QueueType<
                dev::DevOacc,
                queue::NonBlocking
            >
            {
                using type = queue::QueueOaccNonBlocking;
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The thread Oacc device wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
            template<>
            struct CurrentThreadWaitFor<
                dev::DevOacc>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevOacc const & dev)
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
