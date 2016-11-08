require 'optim'
local train = {  }
train.inputs = torch.CudaTensor(  )
train.labels = torch.CudaTensor(  )
train.netTimer = torch.Timer(  )
train.dataTimer = torch.Timer(  )
function train.setOption( opt )
	assert( opt.batchSize > 0 )
	assert( opt.epochSize > 0 )
	assert( opt.numOut > 0 )
	assert( opt.pathModel:match( '(.+).t7$' ):len(  ) > 0 )
	assert( opt.pathOptim:match( '(.+).t7$' ):len(  ) > 0 )
	assert( opt.pathTrainLog:match( '(.+).log$' ):len(  ) > 0 )
	train.batchSize = opt.batchSize
	train.epochSize = opt.epochSize
	train.numOut = opt.numOut
	train.pathModel = opt.pathModel
	train.pathOptim = opt.pathOptim
	train.pathTrainLog = opt.pathTrainLog
end
function train.setModel( modelSet )
	assert( #modelSet.params == #modelSet.grads )
	assert( #modelSet.params == #modelSet.optims )
	for g = 1, #modelSet.params do 
		assert( modelSet.params[ g ]:numel(  ) == modelSet.grads[ g ]:numel(  ) ) 
	end
	train.model = modelSet.model
	train.criterion = modelSet.criterion
	train.params = modelSet.params
	train.grads = modelSet.grads
	train.optims = modelSet.optims
end
function train.setDonkey( donkeys )
	train.donkeys = donkeys
end
function train.setFunction( dataLoader, evaluator )
	train.dataLoader = dataLoader
	train.evaluator = evaluator
end
function train.train( epoch )
	-- Initialization.
	local trainLogger = io.open( train.pathTrainLog, 'a' )
	local epochTimer = torch.Timer(  )
	local dataLoader = train.dataLoader
	local trainBatch = train.trainBatch
	train.epoch = epoch
	train.evalEpoch = torch.Tensor( train.numOut ):fill( 0 )
	train.lossEpoch = 0
	train.batchNumber = 0
	-- Do the job.
	train.print( string.format( 'Train epoch %d.', epoch ) )
	cutorch.synchronize(  )
	train.model:training(  )
	for b = 1, train.epochSize do
		train.donkeys:addjob(
			function(  ) 
				return dataLoader(  )
			end, -- Job callback.
			function( x, y ) 
				trainBatch( x, y ) 
			end -- End callback.
		)
	end
	train.donkeys:synchronize(  )
	cutorch.synchronize(  )
	train.evalEpoch = train.evalEpoch / train.epochSize
	train.lossEpoch = train.lossEpoch / train.epochSize
	local evalEpochStr = train.tensor2str( train.evalEpoch, '%.4f' )
	train.print( string.format( 'Epoch %d, time %.2fs, avg loss %.4f, eval %s', 
		epoch, epochTimer:time(  ).real, train.lossEpoch, evalEpochStr ) )
	trainLogger:write( string.format( '%03d %.4f %s\n', epoch, train.lossEpoch, evalEpochStr ) )
	trainLogger:close(  )
	-- Save model.
	train.print( 'Save model.' )
	train.model:clearState()
	saveDataParallel( train.pathModel:format( epoch ), train.model )
	torch.save( train.pathOptim:format( epoch ), train.optims )
	train.print( 'Done.' )
	collectgarbage(  )
end
function train.trainBatch( inputsCpu, labelsCpu )
	-- Initialization.
	local dataTime = train.dataTimer:time(  ).real
	train.netTimer:reset(  )
	train.inputs:resize( inputsCpu:size(  ) ):copy( inputsCpu )
	train.labels:resize( labelsCpu:size(  ) ):copy( labelsCpu )
	train.model:zeroGradParameters(  )
	-- Forward pass.
	cutorch.synchronize(  )
	local outputs = train.model:forward( train.inputs )
	local err = train.criterion:forward( outputs, train.labels )
	-- Backward pass.
	local gradOutputs = train.criterion:backward( outputs, train.labels )
	train.model:backward( train.inputs, gradOutputs )
	-- Update weights.
	for g = 1, #train.params do
		optim.sgd( function( x ) return _, train.grads[ g ] end, train.params[ g ], train.optims[ g ] )
	end
	if train.model.needsSync then train.model:syncParameters(  ) end
	cutorch.synchronize(  )
	train.batchNumber = train.batchNumber + 1
	train.lossEpoch = train.lossEpoch + err
	-- Task evaluation.
	local eval = train.evaluator( outputs, labelsCpu )
	train.evalEpoch = train.evalEpoch + eval
	local evalStr = train.tensor2str( eval, '%.2f' )
	local netTime = train.netTimer:time(  ).real
	local totalTime = dataTime + netTime
	local speed = train.batchSize / totalTime
	-- Print information
	train.print( string.format( 'Epoch %d, %d/%d, %dim/s (data %.2fs, net %.2fs), err %.2f, eval %s', 
		train.epoch, train.batchNumber, train.epochSize, speed, dataTime, netTime, err, evalStr ) )
	train.dataTimer:reset(  )
	collectgarbage(  )
end
function train.tensor2str( tensor, precision )
	local str = ''
	for i = 1, tensor:numel(  ) do
		str = string.format( '%s' .. precision .. ' ', str, tensor[ i ] )
	end
	return str
end
function train.print( str )
	print( 'TRAIN) ' .. str )
end
return train
