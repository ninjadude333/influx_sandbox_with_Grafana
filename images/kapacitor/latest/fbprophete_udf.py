import os
import sys
import time
from datetime import datetime
from fbprophet import Prophet
from kapacitor.udf.agent import Agent, Handler, Server
from kapacitor.udf import udf_pb2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.special import inv_boxcox

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger()

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])


class TimeUtils(object):
    @staticmethod
    def nanosec_to_datestamp(nanosecTimestamp):
        sec_timestamp = nanosecTimestamp / 1e9
        dt = datetime.fromtimestamp(sec_timestamp)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def datestamp_to_nanosec(datestamp_string):
        sec_timestamp = time.mktime(time.strptime(datestamp_string, '%Y-%m-%d %H:%M:%S'))
        nanosec_timestamp = int(sec_timestamp * 1e9)
        return nanosec_timestamp
		
class ProphetAnomalyDetection(object):
    def __init__(self):
        self._df_array = []
        self._prophet_prediction_df = None
        self._scaler = StandardScaler(with_mean=False)

    @staticmethod
    def ts_resample(points_df, resample_length):
        new_index = points_df.resample(resample_length + 's', closed='right').asfreq()[:-1]
        concat_sort = pd.concat([points_df, new_index]).sort_index().interpolate(method='time')
        resampled_data = concat_sort[~concat_sort.index.duplicated()].reindex(new_index.index)
        resampled_data.index = pd.to_datetime(resampled_data.index)
        return resampled_data

    def append_training_data(self, datestamp, value):
        self._df_array.append([datestamp, value])

    def clear(self):
        del self._df_array[:]
        self._df_array = []
        del self._prophet_prediction_df
        self._prophet_prediction_df = None

    def _create_training_dataframe(self):
        df = pd.DataFrame(self._df_array, columns=['time', 'value'])
        df.index = pd.to_datetime(df.time, utc=False)
        #df.value = pd.to_numeric(df.value, downcast='float')
        df.index.name = None
        del df['time']
        return df

    def _prepare_training_data(self):
        points_df = self._create_training_dataframe()
        prophet_training_df = self.ts_resample(points_df, '300')
        prophet_training_df.reset_index(inplace=True)
        prophet_training_df.columns = ['ds', 'y']
        
        #prophet_training_df['y'] = self._scaler.fit_transform(prophet_training_df)
        #prophet_training_df['y'],self._lmbda = stats.boxcox(prophet_training_df['y']+1)
        return prophet_training_df

    def prophet_prediction(self, intervalwidth):
        prophet_training_df = self._prepare_training_data()
		
		# fit the prophet model
        with suppress_stdout_stderr():
            m = Prophet(interval_width=intervalwidth, weekly_seasonality=24, daily_seasonality=24)
            m.fit(prophet_training_df)

        # Return the predicted points to dataframe
        #print >> sys.stderr, "Start Prediction! "+str(intervalwidth)
        logger.info("Start Prediction!")
        future = m.make_future_dataframe(periods=600, freq='300S', include_history=False)
        fcst = m.predict(future)
        predicted_points = pd.DataFrame(fcst, columns=['ds','yhat_lower','yhat_upper','yhat'])
        predicted_points = predicted_points.set_index('ds')
        #predicted_points = inv_boxcox(predicted_points,self._lmbda)-1
        #predicted_points[['yhat_lower','yhat_upper','yhat']] = self._scaler.inverse_transform(predicted_points)
        predicted_points.index.name = None
        #print >> sys.stderr, "End Prediction!"
        logger.info("End Prediction!")

        self._prophet_prediction_df = predicted_points
        
    def build_prediction_points(self, _point_tags):
        if self._prophet_prediction_df is None:
            return []

        pointList = []
        #print >> sys.stderr, "--Start build points list--"
        logger.info("--Start build points list--")
        
        for index, row in self._prophet_prediction_df.iterrows():
            response = udf_pb2.Response()
            response.point.tags.update(_point_tags)
            df_yhat_lower = row[0]
            df_yhat_upper = row[1]
            df_yhat = row[2]
            newPointTime = TimeUtils.datestamp_to_nanosec(str(index))
            
            response.point.time = newPointTime
            response.point.fieldsDouble['yhat_lower'] = df_yhat_lower
            response.point.fieldsDouble['yhat_upper'] = df_yhat_upper
            response.point.fieldsDouble['yhat'] = df_yhat
            pointList.append(response)
        #print >> sys.stderr, "--End build points list--"
        logger.info("--End build points list--")
        return pointList

class ProphetHandler(Handler):
    def __init__(self, agent):
         self._agent = agent;
         self._field = ''
         self._intervalwidth = 0.90
         self._prophet = ProphetAnomalyDetection()
         self._point_tags = None

    def info(self):
         response = udf_pb2.Response()
         response.info.wants = udf_pb2.BATCH
         response.info.provides = udf_pb2.BATCH

         response.info.options['field'].valueTypes.append(udf_pb2.STRING)
         response.info.options['intervalwidth'].valueTypes.append(udf_pb2.DOUBLE)

         return response

    def init(self, init_req):
        response = udf_pb2.Response()
        success = True
        msg = ''
        for opt in init_req.options:
            if opt.name == 'field':
                 self._field = opt.values[0].stringValue
				# TODO: Add check if field is from double or int (String fields should set to False)
            elif opt.name == 'intervalwidth':
                self._intervalwidth = opt.values[0].doubleValue

        if self._field == '':
            success = False
            msg += ' must supply field name';
        if self._intervalwidth == '':
            success = False
            msg += ' must supply interval width value';
              
        response.init.success = success
        response.init.error = msg[1:]

        return response

    def begin_batch(self, begin_req):
        self._prophet.clear()
        
        response = udf_pb2.Response()
        response.begin.CopyFrom(begin_req)
        self._begin_response = response

    def point(self, point):
		 # Get Point time
        pointTime = point.time
        prophetDS = TimeUtils.nanosec_to_datestamp(pointTime)

		 # Get Point value
        pointValue = point.fieldsDouble[self._field]
        
		 # Append data
        self._prophet.append_training_data(prophetDS, pointValue)
        # Get Point tags
        self._point_tags = point.tags
		
    def end_batch(self, end_req):
        
        self._prophet.prophet_prediction(self._intervalwidth)
        points = self._prophet.build_prediction_points(self._point_tags);
		
		# Create beginBatch response
        self._begin_response.begin.size = len(points)
        self._agent.write_response(self._begin_response)
        
        #print >> sys.stderr, "--Start send points--"
        logger.info("--Start send points--")
        for point in points:
            self._agent.write_response(point)
        #print >> sys.stderr, "--End send points--"
        logger.info("--End send points--")
        
        response = udf_pb2.Response()
        response.end.CopyFrom(end_req)
        response.end.tmax = points[-1].point.time
        self._agent.write_response(response)
        
    def snapshot(self):
        response = udf_pb2.Response()
        response.snapshot.snapshot = b''
        return response

    def restore(self, restore_req):
        response = udf_pb2.Response()
        response.restore.success = False
        response.restore.error = 'Not Implemented'
        return response

if __name__ == '__main__':
    agent = Agent()
    handler = ProphetHandler(agent)
    agent.handler = handler
	
	#print >> sys.stderr, "Starting Prophet Agent"
    #print ("Starting Prophet Agent", end="", file=sys.stderr)
    logger.info("Starting Prophet Agent")
    agent.start()
    agent.wait()
    logger.info("Agent Prophet Finished")
	#print >> sys.stderr, "Agent Prophet Finished"
