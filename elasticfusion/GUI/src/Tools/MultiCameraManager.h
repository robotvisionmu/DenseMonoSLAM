#ifndef MULTICAMERAMANAGER_H_
#define MULTICAMERAMANAGER_H_

#include <vector>
#include <unordered_map>

#include <boost/lockfree/queue.hpp>

#include <Utils/Options.h>
#include <lcm/lcm-cpp.hpp>

class MultiCameraManager
{
	public:
		MultiCameraManager(){}

		virtual ~MultiCameraManager(){}

		virtual std::vector<std::shared_ptr<LogReader>> devices() const = 0;
		
		virtual void reset() = 0;
};
#endif //MULTICAMERAMANAGER_H_