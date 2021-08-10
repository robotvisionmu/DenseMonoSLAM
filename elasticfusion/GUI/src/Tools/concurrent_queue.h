#ifndef CONCURRENT_QUEUE_H_
#define CONCURRENT_QUEUE_H_

#include <boost/thread/mutex.hpp>
#include <boost/thread/lockable_concepts.hpp>

#include <vector>

template<class T>
class concurrent_queue
{
	public:
		concurrent_queue(){};
		virtual ~concurrent_queue(){};

		void push_back(T object)
		{
			boost::mutex::scoped_lock lock(mutex);
			container.push_back(object);
			lock.unlock();
		}

		std::vector<T> snapshot() const
		{
			boost::mutex::scoped_lock lock(mutex);
			std::vector<T> v(container.begin(), container.end());
			lock.unlock();
			return v;
		}

		T & operator[](int index)
		{
			boost::mutex::scoped_lock lock(mutex);
			T & object = container[index];
			lock.unlock();
			return object;
		}

		const T & operator[](int index) const
		{
			boost::mutex::scoped_lock lock(mutex);
			const T & object = container[index];
			lock.unlock();
			return object;	
		}

		int size()
		{
			boost::mutex::scoped_lock lock(mutex);
			int size = container.size();
			lock.unlock();
			return size;
		}

		void clear()
		{
			boost::mutex::scoped_lock lock(mutex);
			container.clear();
			lock.unlock();
		}

	private:
		mutable boost::mutex mutex;
		std::vector<T> container;
};
#endif /*CONCURRENT_QUEUE_H_*/