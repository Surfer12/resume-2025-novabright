import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const ActivityFeed = ({ activities, loading }) => {
  if (loading && (!activities || activities.length === 0)) {
    return (
      <div className="bg-gray-800/80 p-6 rounded-lg shadow-lg animate-pulse border border-gray-700 h-full">
        <div className="h-4 bg-gray-700 rounded w-1/4 mb-6"></div>
        <div className="space-y-4">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-gray-700 rounded-full"></div>
              <div className="flex-1 space-y-2">
                <div className="h-3 bg-gray-700 rounded"></div>
                <div className="h-2 bg-gray-700 rounded w-1/2"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <motion.div
      className="bg-gray-800/80 backdrop-blur-sm p-6 rounded-lg shadow-lg border border-gray-700 h-full"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
    >
      <h3 className="text-lg font-semibold text-white mb-4">Recent Activity</h3>
      <div className="overflow-y-auto max-h-96 pr-2">
        <ul className="space-y-4">
          <AnimatePresence>
            {activities && activities.map((activity, index) => (
              <motion.li
                key={activity.id}
                className="flex items-center space-x-4"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <div className="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center border border-gray-600">
                  <span className="text-sm font-bold text-gray-400">
                    {activity.user?.name?.charAt(0) || '?'}
                  </span>
                </div>
                <div className="flex-1">
                  <p className="text-sm text-gray-300">
                    <span className="font-semibold text-white">{activity.user?.name || 'Unknown User'}</span> {activity.message}
                  </p>
                  <p className="text-xs text-gray-500">
                    {new Date(activity.timestamp).toLocaleString()}
                  </p>
                </div>
                <div className="text-xs font-mono px-2 py-1 bg-gray-700/50 text-blue-400 rounded">
                  {activity.type}
                </div>
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      </div>
    </motion.div>
  );
};

export default ActivityFeed; 