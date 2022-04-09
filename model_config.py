model_parameters = {
	'RL_CNN':{ # model that uses only log-returns as state inputs
		'use_forecasts':False,
		'nb_forecasts':None,
		'forecast_type':'strong', # this value does not matter 
		'use_cnn_state':True,
	},
	'RL_str_fcast':{ # model that uses only strong forecasts as state inputs
		'use_forecasts':True,
		'nb_forecasts':2,
		'forecast_type':'strong',
		'use_cnn_state':False,
	},
	'RL_all_inp':{ # model that uses both log-returns and strong forecasts as state inputs
		'use_forecasts':True,
		'nb_forecasts':2,
		'forecast_type':'strong',
		'use_cnn_state':True,
	},
}