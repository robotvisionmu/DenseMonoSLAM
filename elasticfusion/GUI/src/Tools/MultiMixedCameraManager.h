#ifndef MULTIMIXEDCAMERAMANAGER_H_
#define MULTIMIXEDCAMERAMANAGER_H_


#include <vector>

#include "MultiUsbCameraManager.h"
#include "MultiLiveCameraManager.h"
#include "MultiLogCameraManager.h"
class MultiMixedCameraManager : public MultiCameraManager
{
	public:
		MultiMixedCameraManager()
		{
			m_multi_log_camera_manager = new MultiLogCameraManager();
            m_multi_live_camera_manager = new MultiLiveCameraManager();
            m_multi_usb_camera_manager = new MultiUsbCameraManager();
		}

		virtual ~MultiMixedCameraManager()
        {
            delete m_multi_live_camera_manager;
            delete m_multi_log_camera_manager;
            delete m_multi_usb_camera_manager;
        }

		std::vector<std::shared_ptr<LogReader>> devices() const 
		{
            std::vector<std::shared_ptr<LogReader>> log_ds = m_multi_log_camera_manager->devices();
            std::vector<std::shared_ptr<LogReader>> live_ds = m_multi_live_camera_manager->devices();
            std::vector<std::shared_ptr<LogReader>> usb_ds = m_multi_usb_camera_manager->devices();
            log_ds.insert(log_ds.end(), live_ds.begin(), live_ds.end());
            log_ds.insert(log_ds.end(), usb_ds.begin(), usb_ds.end());
			return log_ds;
		}
		
		void reset()
		{
			m_multi_live_camera_manager->reset();
            m_multi_log_camera_manager->reset();
            m_multi_usb_camera_manager->reset();
		}

	private:
		MultiLogCameraManager * m_multi_log_camera_manager;
        MultiLiveCameraManager * m_multi_live_camera_manager;
        MultiUsbCameraManager * m_multi_usb_camera_manager;
};


#endif /*MULTIMIXEDCAMERAMANAGER_H_*/