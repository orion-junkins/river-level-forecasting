#%%
from aws_dispatcher import AWSDispatcher

dispatcher = AWSDispatcher('illinois-kerby', 'Block_GRU_6hour')

dispatcher.rebuild_historical_forecast(horizon=24, stride=1)
