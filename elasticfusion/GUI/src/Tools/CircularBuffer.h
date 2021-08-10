#ifndef CIRCULARBUFFER_H_
#define CIRCULARBUFFER_H_

#include <boost/thread/mutex.hpp>
#include <boost/thread/lockable_concepts.hpp> 
#include <boost/circular_buffer.hpp>
#include <boost/thread/condition.hpp>
#include <boost/thread/thread.hpp>
#include <boost/call_traits.hpp>
#include <boost/bind.hpp>

#include <future>

class CircularBuffer
{
	public:
		CircularBuffer(int capacity = 100)
		:capacity(capacity)
		{
			start = end = count = 0;

			container = new Frame[capacity];
			for(int i = 0; i < capacity; i++)
			{
				std::shared_ptr<unsigned short> dp(new unsigned short[Resolution::getInstance().numPixels() * 2], std::default_delete<unsigned short[]>());
				std::shared_ptr<unsigned char> rgb(new unsigned char[Resolution::getInstance().numPixels() * 3], std::default_delete<unsigned char[]>());
				int64_t t = -1;

				container[i] = Frame(std::pair<std::shared_ptr<unsigned short>, std::shared_ptr<unsigned char>>(dp, rgb), t); 
			}
		}

		virtual ~CircularBuffer()
		{
			delete [] container;
		}

/*		void push(std::unique_ptr<unsigned char[]> rgb, std::unique_ptr<unsigned short[]> depth, int64_t timestamp)
		{
			boost::mutex::scoped_lock lock(mutex);
			not_full.wait(lock, boost::bind(&CircularBuffer::is_not_full, this));
			container.push_front(Frame(std::move(depth), std::move(rgb), timestamp));
			lock.unlock();	
			not_empty.notify_one();
		}*/
		
		void push(unsigned char * rgb, unsigned short * dp, int64_t timestamp/*std::unique_ptr<unsigned char[]> rgb, std::unique_ptr<unsigned short[]> dp, int64_t timestamp*/)
		{
			memcpy(container[end].first.first.get(), dp, Resolution::getInstance().numPixels() * 2);
			memcpy(container[end].first.second.get(), rgb, Resolution::getInstance().numPixels() * 3);
			container[end].second = timestamp;
			count = count == capacity ? count : count + 1;

			end = (end + 1) % capacity;
			start = end == start ? (start + 1) % capacity : start; // change this to block when full?
		}

		void pop(std::shared_ptr<unsigned char> & rgb, std::shared_ptr<unsigned short> & dp, int64_t & timestamp)
		{
			memcpy(dp.get(), container[start].first.first.get(), Resolution::getInstance().numPixels() * 2);//start
			memcpy(rgb.get(), container[start].first.second.get(),  Resolution::getInstance().numPixels() * 3);
			timestamp = container[start].second;
			count = count == 0 ? count : count - 1;

			start = (start + 1) % capacity;
			end = end == start? (end + 1) % capacity : end; // change this to block when empty?
		}

		/*bool try_push(std::unique_ptr<unsigned char[]> rgb, std::unique_ptr<unsigned short[]> depth, int64_t timestamp)
		{
			boost::mutex::scoped_lock lock(mutex);
			if(!is_not_full())
			{
				lock.unlock();
				return false;
			}
			container.push_front(Frame(std::move(depth), std::move(rgb), timestamp));
			lock.unlock();	
			not_empty.notify_one();

			return true;
		}*/

		/*std::future<void> async_push(std::unique_ptr<unsigned char[]> rgb, std::unique_ptr<unsigned short[]> depth, int64_t timestamp)
		{
			std::future<void> future(std::async(std::launch::async, [&]{push(std::move(rgb), std::move(depth), timestamp);}));
           
           	return future;		
		}*/

		/*void pop(std::shared_ptr<unsigned char> & rgb, std::shared_ptr<unsigned short> & depth, int64_t & timestamp)
		{
			boost::mutex::scoped_lock lock(mutex);
			not_empty.wait(lock, boost::bind(&CircularBuffer::is_not_empty, this));

			Frame & frame = container.back();
			depth = frame.depth;
			rgb = frame.rgb;
			timestamp = frame.timestamp;
			container.pop_back();
			lock.unlock();
			not_full.notify_one();
		}*/

/*		bool try_pop(std::shared_ptr<unsigned char> & rgb, std::shared_ptr<unsigned short> & depth, int64_t & timestamp)
		{
			boost::mutex::scoped_lock lock(mutex);
			
			if(!is_not_empty())
			{
				lock.unlock();
				return false;
			}

			Frame & frame = container.back();
			depth = frame.depth;
			rgb = frame.rgb;
			timestamp = frame.timestamp;
			
			container.pop_back();
			lock.unlock();
			not_full.notify_one();
			return true;
		}*/

		/*std::future<void> async_pop(std::shared_ptr<unsigned char> & rgb, std::shared_ptr<unsigned short> & depth, int64_t & timestamp)
		{
			std::future<void> future(std::async(std::launch::async, [&]{pop(rgb, depth, timestamp);}));
           
           	return future;
		}*/

		bool empty()
		{
			return count == 0;
		}

	private:
		bool is_not_empty(){ return start != end; }
		bool is_not_full(){ return (end - start - 1) % capacity; }

/*		boost::mutex mutex;
		boost::condition  not_empty;
		boost::condition not_full;*/

		int capacity;
		int count;
		int start;
		int end;
		int latest;
		using Frame = std::pair<std::pair<std::shared_ptr<unsigned short>, std::shared_ptr<unsigned char>>, int64_t>;
		Frame * container;
		

		/*using Buffer = boost::circular_buffer_space_optimized<Frame>;
		Buffer container;*/
};
#endif //CIRCULARBUFFER_H_