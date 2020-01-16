/* Copyright 2019 Benjamin Worpitz
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

#include <alpaka/dev/Traits.hpp>
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/Traits.hpp>
#include <alpaka/wait/Traits.hpp>

#include <alpaka/queue/Traits.hpp>
#include <alpaka/queue/Properties.hpp>

#include <alpaka/core/Omp5.hpp>

#include <alpaka/queue/cpu/IGenericThreadsQueue.hpp>
#include <alpaka/queue/QueueGenericThreadsNonBlocking.hpp>
#include <alpaka/queue/QueueGenericThreadsBlocking.hpp>

#ifdef SPEC_FAKE_OMP_TARGET_CPU
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <map>
#endif

namespace alpaka
{
    namespace dev
    {
        class DevOmp5;
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
        class PltfOmp5;
    }

    namespace dev
    {
        namespace omp5
        {
            namespace detail
            {
                //#############################################################################
                //! The Omp5 device implementation.
                class DevOmp5Impl
                {
                public:
                    //-----------------------------------------------------------------------------
#ifndef SPEC_FAKE_OMP_TARGET_CPU
                    DevOmp5Impl(int iDevice) : m_iDevice(iDevice) {}
#endif
                    //-----------------------------------------------------------------------------
                    DevOmp5Impl(DevOmp5Impl const &) = delete;
                    //-----------------------------------------------------------------------------
                    DevOmp5Impl(DevOmp5Impl &&) = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevOmp5Impl const &) -> DevOmp5Impl & = delete;
                    //-----------------------------------------------------------------------------
                    auto operator=(DevOmp5Impl &&) -> DevOmp5Impl & = delete;
                    //-----------------------------------------------------------------------------
#ifndef SPEC_FAKE_OMP_TARGET_CPU
                    ~DevOmp5Impl() = default;
#else
                    void spawn(int n)
                    {
                        m_threadPool.reserve(n);
                        m_poolActiveCount = n-m_threadPool.size();
                        while(m_threadPool.size() < n)
                        {
                            m_threadPool.emplace_back([this,n](int threadNum)
                                {
                                    unsigned long long prevPoolGeneration = m_poolGeneration;
                                    {
                                        std::unique_lock<std::mutex> lock(m_mtxPool);
                                        if(--m_poolActiveCount == 0)
                                            m_cvPool.notify_all();
                                    }
                                    while(true)
                                    {
                                        std::unique_lock<std::mutex> lock(m_mtxPool);
                                        // std::cout << "Worker " << threadNum << " going to sleep." << std::endl;
                                        m_cvPool.wait(lock, [this, prevPoolGeneration, threadNum]()
                                            {
                                                return prevPoolGeneration < m_poolGeneration
                                                    && m_poolActiveCount > threadNum;
                                            });
                                        // std::cout << "Worker " << threadNum << " woke up." << std::endl;
                                        lock.unlock();
                                        if(m_poolTask)
                                            m_poolTask();
                                        else
                                            return;
                                        // std::cout << "Worker " << threadNum << " done." << std::endl;
                                        lock.lock();
                                        // std::cout << "Worker " << threadNum << " decrementing." << std::endl;
                                        if(--m_poolActiveCount == 0)
                                            m_cvPool.notify_all();
                                    }
                                }, m_threadPool.size());
                            // std::cout << "Created Worker " << m_threadPool.size()-1 << std::endl;
                            m_idMap[m_threadPool.back().get_id()] = m_threadPool.size()-1;
                        }
                        std::unique_lock<std::mutex> lock(m_mtxPool);
                        // std::cout << "waiting for new workers to spawn..." << std::endl;
                        m_cvPool.wait(lock, [this](){return m_poolActiveCount == 0;});
                        // std::cout << "workers spawned" << std::endl;
                    }

                    void invoke(std::function<void()> fn, int n)
                    {
                        if(n > m_threadPool.size())
                            spawn(n);
                        m_poolTask = fn;
                        ++m_poolGeneration;
                        m_poolActiveCount = n;
                        m_cvPool.notify_all();
                        // std::cout << "notifying " << n << " workers" << std::endl;
                        std::unique_lock<std::mutex> lock(m_mtxPool);
                        m_cvPool.wait(lock, [this](){return m_poolActiveCount == 0;});
                        // std::cout << "workers done" << std::endl;
                    }

                    DevOmp5Impl(int iDevice)
                    {
                        spawn(1);
                        m_masterThread = m_idMap.begin()->first;
                    }

                    ~DevOmp5Impl()
                    {
                        m_poolActiveCount = m_threadPool.size();
                        m_poolTask = std::function<void()>();
                        m_cvPool.notify_all();
                        for(auto& t : m_threadPool)
                            t.join();
                    }

                    std::vector<std::thread> m_threadPool;
                    unsigned long long m_poolGeneration = 0, m_poolActiveCount = 0;
                    std::mutex m_mtxPool;
                    std::condition_variable m_cvPool;
                    std::function<void()> m_poolTask;
                    std::map<std::thread::id, int> m_idMap;
                    std::thread::id m_masterThread;
#endif

                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST auto getAllExistingQueues() const
                    -> std::vector<std::shared_ptr<queue::IGenericThreadsQueue<DevOmp5>>>
                    {
                        std::vector<std::shared_ptr<queue::IGenericThreadsQueue<DevOmp5>>> vspQueues;

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
                    ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<queue::IGenericThreadsQueue<DevOmp5>> spQueue)
                    -> void
                    {
                        std::lock_guard<std::mutex> lk(m_Mutex);

                        // Register this queue on the device.
                        m_queues.push_back(spQueue);
                    }

                    int iDevice() const {return m_iDevice;}

                private:
                    std::mutex mutable m_Mutex;
                    std::vector<std::weak_ptr<queue::IGenericThreadsQueue<DevOmp5>>> mutable m_queues;
                    int m_iDevice = 0;
                };
            }
        }
        //#############################################################################
        //! The Omp5 device handle.
        class DevOmp5 : public concepts::Implements<wait::ConceptCurrentThreadWaitFor, DevOmp5>
        {
            friend struct pltf::traits::GetDevByIdx<pltf::PltfOmp5>;

        protected:
            //-----------------------------------------------------------------------------
            DevOmp5(int iDevice) :
                m_spDevOmp5Impl(std::make_shared<omp5::detail::DevOmp5Impl>(iDevice))
            {}
        public:
            //-----------------------------------------------------------------------------
            DevOmp5(DevOmp5 const &) = default;
            //-----------------------------------------------------------------------------
            DevOmp5(DevOmp5 &&) = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevOmp5 const &) -> DevOmp5 & = default;
            //-----------------------------------------------------------------------------
            auto operator=(DevOmp5 &&) -> DevOmp5 & = default;
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator==(DevOmp5 const & rhs) const
            -> bool
            {
                return m_spDevOmp5Impl->iDevice() == rhs.m_spDevOmp5Impl->iDevice();
            }
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST auto operator!=(DevOmp5 const & rhs) const
            -> bool
            {
                return !((*this) == rhs);
            }
            //-----------------------------------------------------------------------------
            ~DevOmp5() = default;
            int iDevice() const {return m_spDevOmp5Impl->iDevice();}

            ALPAKA_FN_HOST auto getAllQueues() const
            -> std::vector<std::shared_ptr<queue::IGenericThreadsQueue<DevOmp5>>>
            {
                return m_spDevOmp5Impl->getAllExistingQueues();
            }

            //-----------------------------------------------------------------------------
            //! Registers the given queue on this device.
            //! NOTE: Every queue has to be registered for correct functionality of device wait operations!
            ALPAKA_FN_HOST auto registerQueue(std::shared_ptr<queue::IGenericThreadsQueue<DevOmp5>> spQueue) const
            -> void
            {
                m_spDevOmp5Impl->registerQueue(spQueue);
            }

        public:
            std::shared_ptr<omp5::detail::DevOmp5Impl> m_spDevOmp5Impl;

#ifdef SPEC_FAKE_OMP_TARGET_CPU
            void invoke(std::function<void()> fn, int n)
            {
                m_spDevOmp5Impl->invoke(fn, n);
            }

            const std::map<std::thread::id, int> * idMapP() const {return &m_spDevOmp5Impl->m_idMap;}
            std::thread::id masterThread() const {return m_spDevOmp5Impl->m_masterThread;}
#endif
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
                dev::DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getName(
                    dev::DevOmp5 const &)
                -> std::string
                {
                    return std::string("OMP5 target");
                }
            };

            //#############################################################################
            //! The CUDA RT device available memory get trait specialization.
            template<>
            struct GetMemBytes<
                dev::DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getMemBytes(
                    dev::DevOmp5 const & dev)
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
                dev::DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getFreeMemBytes(
                    dev::DevOmp5 const & dev)
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
                dev::DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto reset(
                    dev::DevOmp5 const & dev)
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
            class BufOmp5;

            namespace traits
            {
                //#############################################################################
                //! The CUDA RT device memory buffer type trait specialization.
                template<
                    typename TElem,
                    typename TDim,
                    typename TIdx>
                struct BufType<
                    dev::DevOmp5,
                    TElem,
                    TDim,
                    TIdx>
                {
                    using type = mem::buf::BufOmp5<TElem, TDim, TIdx>;
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
                dev::DevOmp5>
            {
                using type = pltf::PltfOmp5;
            };
        }
    }
    namespace queue
    {
        using QueueOmp5NonBlocking = QueueGenericThreadsNonBlocking<dev::DevOmp5>;
        using QueueOmp5Blocking = QueueGenericThreadsBlocking<dev::DevOmp5>;

        namespace traits
        {
            template<>
            struct QueueType<
                dev::DevOmp5,
                queue::Blocking
            >
            {
                using type = queue::QueueOmp5Blocking;
            };

            template<>
            struct QueueType<
                dev::DevOmp5,
                queue::NonBlocking
            >
            {
                using type = queue::QueueOmp5NonBlocking;
            };
        }
    }
    namespace wait
    {
        namespace traits
        {
            //#############################################################################
            //! The thread Omp5 device wait specialization.
            //!
            //! Blocks until the device has completed all preceding requested tasks.
            //! Tasks that are enqueued or queues that are created after this call is made are not waited for.
            template<>
            struct CurrentThreadWaitFor<
                dev::DevOmp5>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto currentThreadWaitFor(
                    dev::DevOmp5 const & dev)
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
