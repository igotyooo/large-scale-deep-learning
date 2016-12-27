local ffi = require 'ffi'
local task = torch.class( 'TaskManager' )
--------------------------------------------
-------- TASK-INDEPENDENT FUNCTIONS --------
--------------------------------------------
function task:__init(  )
	self.opt = {  }
	self.dbtr = {  }
	self.dbval = {  }
	self.inputStat = {  }
	self.numBatchTrain = 0
	self.numBatchVal = 0
end
function task:setOption( arg )
	self.opt = self:parseOption( arg )
	assert( self.opt.numGpu )
	assert( self.opt.backend )
	assert( self.opt.numDonkey )
	assert( self.opt.data )
	assert( self.opt.numEpoch )
	assert( self.opt.epochSize )
	assert( self.opt.batchSize )
	assert( self.opt.learnRate )
	assert( self.opt.momentum )
	assert( self.opt.weightDecay )
	assert( self.opt.startFrom )
	assert( self.opt.dirRoot )
	assert( self.opt.pathDbTrain )
	assert( self.opt.pathDbVal )
	assert( self.opt.pathImStat )
	assert( self.opt.dirModel )
	assert( self.opt.pathModel )
	assert( self.opt.pathOptim )
	assert( self.opt.pathTrainLog )
	assert( self.opt.pathValLog )
	paths.mkdir( self.opt.dirRoot )
	paths.mkdir( self.opt.dirModel )
end
function task:getOption(  )
	return self.opt
end
function task:setDb(  )
	paths.dofile( string.format( '../db/%s.lua', self.opt.data ) )
	if paths.filep( self.opt.pathDbTrain ) then
		self:print( 'Load train db.' )
		self.dbtr = torch.load( self.opt.pathDbTrain )
		self:print( 'Done.' )
	else
		self:print( 'Create train db.' )
		self.dbtr = self:createDbTrain(  )
		torch.save( self.opt.pathDbTrain, self.dbtr )
		self:print( 'Done.' )
	end
	if paths.filep( self.opt.pathDbVal ) then
		self:print( 'Load val db.' )
		self.dbval = torch.load( self.opt.pathDbVal )
		self:print( 'Done.' )
	else
		self:print( 'Create val db.' )
		self.dbval = self:createDbVal(  )
		torch.save( self.opt.pathDbVal, self.dbval )
		self:print( 'Done.' )
	end
	self.numBatchTrain, self.numBatchVal = self:setNumBatch(  )
	assert( self.numBatchTrain > 0 )
	assert( self.numBatchVal > 0 )
end
function task:getNumBatch(  )
	return self.numBatchTrain, self.numBatchVal
end
function task:setInputStat(  )
	if paths.filep( self.opt.pathImStat ) then
		self:print( 'Load input data statistics.' )
		self.inputStat = torch.load( self.opt.pathImStat )
		self:print( 'Done.' )
	else
		self:print( 'Estimate input data statistics.' )
		self.inputStat = self:estimateInputStat(  )
		torch.save( self.opt.pathImStat, self.inputStat )
		self:print( 'Done.' )
	end
end
function task:getFunctionTrain(  )
	return
		function(  ) return self:getBatchTrain(  ) end,
		function( x, y ) return self:evalBatch( x, y ) end
end
function task:getFunctionVal(  )
	return
		function( i ) return self:getBatchVal( i ) end,
		function( x, y ) return self:evalBatch( x, y ) end
end
function task:getModel(  )
	local numEpoch = self.opt.numEpoch
	local pathModel = self.opt.pathModel
	local pathOptim = self.opt.pathOptim
	local numGpu = self.opt.numGpu
	local startFrom = self.opt.startFrom
	local backend = self.opt.backend
	local startEpoch = 1
	for e = 1, numEpoch do
		local modelPath = pathModel:format( e )
		local optimPath = pathOptim:format( e )
		if not paths.filep( modelPath ) then startEpoch = e break end 
	end
	local model, params, grads, optims
	if startEpoch == 1 and startFrom:len(  ) == 0 then
		self:print( 'Create model.' )
		model = self:defineModel(  )
		if backend == 'cudnn' then
			require 'cudnn'
			cudnn.convert( model, cudnn )
		end
		params, grads, optims = self:groupParams( model )
	elseif startEpoch == 1 and startFrom:len(  ) > 0 then
		self:print( 'Load user-defined model.' .. startFrom )
		model = loadDataParallel( startFrom, numGpu, backend )
		params, grads, optims = self:groupParams( model )
	elseif startEpoch > 1 then
		self:print( string.format( 'Load model from epoch %d.', startEpoch - 1 ) )
		model = loadDataParallel( pathModel:format( startEpoch - 1 ), numGpu, backend )
		params, grads, _ = self:groupParams( model )
		optims = torch.load( pathOptim:format( startEpoch - 1 ) )
	end
	self:print( 'Done.' )
	local criterion = self:defineCriterion(  )
	self:print( 'Model looks' )
	print( model )
	print(criterion)
	self:print( 'Convert model to cuda.' )
	model = model:cuda(  )
	criterion:cuda(  )
	self:print( 'Done.' )
	cutorch.setDevice( 1 )
	local modelSet = {  }
	modelSet.model = model
	modelSet.criterion = criterion
	modelSet.params = params
	modelSet.grads = grads
	modelSet.optims = optims
	-- Verification
	assert( #self.opt.learnRate == #modelSet.params )
	assert( #self.opt.learnRate == #modelSet.grads )
	assert( #self.opt.learnRate == #modelSet.optims )
	for g = 1, #modelSet.params do
		assert( modelSet.params[ g ]:numel(  ) == modelSet.grads[ g ]:numel(  ) )
	end
	return modelSet, startEpoch
end
function task:print( str )
	print( 'TASK MANAGER) ' .. str )
end
-----------------------------------------
-------- TASK-SPECIFIC FUNCTIONS --------
-----------------------------------------
function task:parseOption( arg )
	local cmd = torch.CmdLine(  )
	cmd:option( '-task', arg[ 2 ] )
	-- System.
	cmd:option( '-numGpu', 2, 'Number of GPUs.' )
	cmd:option( '-backend', 'cudnn', 'cudnn or nn.' )
	cmd:option( '-numDonkey', 16, 'Number of donkeys for data loading.' )
	-- Data.
	cmd:option( '-data', 'UCF101', 'Name of dataset defined in "./db/"' )
	cmd:option( '-imageSize', 240, 'Short side of initial resize.' )
	cmd:option( '-cropSize', 224, 'Size of random square crop.' )
	cmd:option( '-keepAspect', 0, '1 for keep, 0 for no.' )
	cmd:option( '-normalizeStd', 0, '1 for normalize piexel std to 1, 0 for no.' )
	cmd:option( '-seqLength', 16, 'Number of frames per input video' )
	cmd:option( '-caffeInput', 1, '1 for caffe input, 0 for no.' )
	-- Model.
	cmd:option( '-dropout', 0.5, 'Dropout ratio.' )
	cmd:option( '-hiddenSize', 256, 'Size of hidden layer.' )
	cmd:option( '-videoPool', 'sum', 'Pooling method for frames per video' )
	cmd:option( '-numOut', 1, 'Number of outputs from net.' )
	-- Train.
	cmd:option( '-numEpoch', 50, 'Number of total epochs to run.' )
	cmd:option( '-epochSize', 2384, 'Number of batches per epoch.' )
	cmd:option( '-batchSize', 256, 'Frame-level mini-batch size.' )
	cmd:option( '-learnRate', '1e-3,1e-3,1e-3', 'Supports multi-lr for multi-module like "lr1,lr2,lr3".' )
	cmd:option( '-momentum', 0.9, 'Momentum.' )
	cmd:option( '-weightDecay', 5e-4, 'Weight decay.' )
	cmd:option( '-startFrom', '', 'Path to the initial model. Using it for LR decay is recommended.' )
	local opt = cmd:parse( arg or {  } )
	-- Set dst paths.
	local dirRoot = paths.concat( gpath.dataout, opt.data )
	local pathDbTrain = paths.concat( dirRoot, 'dbTrain.t7' )
	local pathDbVal = paths.concat( dirRoot, 'dbVal.t7' )
	local pathImStat = paths.concat( dirRoot, 'inputStat.t7' )
	if opt.caffeInput == 1 then pathImStat = pathImStat:match( '(.+).t7$' ) .. 'Caffe.t7' end
	local ignore = { numGpu=true, backend=true, numDonkey=true, data=true, numEpoch=true, startFrom=true }
	local dirModel = paths.concat( dirRoot, cmd:string( opt.task, opt, ignore ) )
	if opt.startFrom ~= '' then
		local baseDir, epoch = opt.startFrom:match( '(.+)/model_(%d+).t7' )
		dirModel = paths.concat( baseDir, cmd:string( 'model_' .. epoch, opt, ignore ) )
	end
	opt.dirRoot = dirRoot
	opt.pathDbTrain = pathDbTrain
	opt.pathDbVal = pathDbVal
	opt.pathImStat = pathImStat
	opt.dirModel = dirModel
	opt.pathModel = paths.concat( opt.dirModel, 'model_%03d.t7' )
	opt.pathOptim = paths.concat( opt.dirModel, 'optimState_%03d.t7' )
	opt.pathTrainLog = paths.concat( opt.dirModel, 'train.log' )
	opt.pathValLog = paths.concat( opt.dirModel, 'val.log' )
	-- Value processing.
	opt.normalizeStd = opt.normalizeStd > 0
	opt.keepAspect = opt.keepAspect > 0
	opt.caffeInput = opt.caffeInput > 0
	opt.learnRate = opt.learnRate:split( ',' )
	for k,v in pairs( opt.learnRate ) do opt.learnRate[ k ] = tonumber( v ) end
	-- Verification.
	assert( opt.imageSize >= opt.cropSize )
	assert( opt.batchSize % opt.seqLength == 0 )
	assert( opt.numOut > 0 )
	return opt
end
function task:createDbTrain(  )
	local dbtr = {  }
	dbtr.vid2path,
	dbtr.vid2numim,
	dbtr.vid2cid,
	dbtr.cid2name,
	dbtr.frameFormat = genDb( 'train' )
	local numVideo = dbtr.vid2path:size( 1 )
	local numClass = dbtr.cid2name:size( 1 )
	self:print( string.format( 'Train: %d videos, %d classes.', numVideo, numClass ) )
	-- Verification.
	assert( dbtr.vid2path:size( 1 ) == dbtr.vid2numim:numel(  ) )
	assert( dbtr.vid2path:size( 1 ) == dbtr.vid2cid:numel(  ) )
	assert( dbtr.cid2name:size( 1 ) == dbtr.vid2cid:max(  ) )
	return dbtr
end
function task:createDbVal(  )
	local dbval = {  }
	dbval.vid2path,
	dbval.vid2numim,
	dbval.vid2cid,
	dbval.cid2name,
	dbval.frameFormat = genDb( 'val' )
	local numVideo = dbval.vid2path:size( 1 )
	local numClass = dbval.cid2name:size( 1 )
	self:print( string.format( 'Val: %d videos, %d classes.', numVideo, numClass ) )
	-- Verification.
	assert( dbval.vid2path:size( 1 ) == dbval.vid2numim:numel(  ) )
	assert( dbval.vid2path:size( 1 ) == dbval.vid2cid:numel(  ) )
	assert( dbval.cid2name:size( 1 ) == dbval.vid2cid:max(  ) )
	return dbval
end
function task:setNumBatch(  )
	local seqLength = self.opt.seqLength
	local batchSize = self.opt.batchSize
	local numBatchTrain = math.floor( self.dbtr.vid2path:size( 1 ) * seqLength / batchSize )
	local numBatchVal = math.floor( self.dbval.vid2path:size( 1 ) * seqLength / batchSize )
	return numBatchTrain, numBatchVal
end
function task:estimateInputStat(  )
	local numIm = 10000
	local batchSize = self.opt.batchSize
	local seqLength = self.opt.seqLength
	local numBatch = math.ceil( numIm / batchSize )
	self.opt.seqLength = 1
	local meanEstimate = torch.Tensor( 3 ):fill( 0 )
	local stdEstimate = torch.Tensor( 3 ):fill( 0 )
	for b = 1, numBatch do
		local batch = self:getBatchTrain(  )
		assert( batch:dim(  ) == 4 )
		self:print( string.format( '%.1f%% (%d/%d)', b * 100 / numBatch, b, numBatch ) )
		meanEstimate:add( batch:mean( 4 ):mean( 3 ):mean( 1 ):squeeze(  ) )
		stdEstimate:add( batch:view( batchSize, 3, -1 ):std( 3 ):mean( 1 ):squeeze(  )  )
	end
	self.opt.seqLength = seqLength
	meanEstimate:div( numBatch )
	stdEstimate:div( numBatch )
	return { mean = meanEstimate, std = stdEstimate }
end
function task:defineModel(  )
	require 'loadcaffe'
	-- Set params.
	local featSize = 4096
	local hiddenSize = 256
	local numClass = self.dbtr.cid2name:size( 1 )
	local dropout = self.opt.dropout
	-- Load pre-trained CNN.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), featSize
	self:print( 'Load pre-trained Caffe feature.' )
	local proto = gpath.net.vggm_caffe_proto
	local caffemodel = gpath.net.vggm_caffe_model
	local features = loadcaffe.load( proto, caffemodel, self.opt.backend )
	features:remove(  )
	features:remove(  )
	features:remove(  )
	features:remove(  )
	features:remove(  )
	features:remove(  ) -- Removes dropout.
	features:add( nn.Dropout( dropout ) )
	features:cuda(  )
	features = makeDataParallel( features, self.opt.numGpu, 1 )
	-- Create FC.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), hiddenSize
	local fc = nn.Sequential(  )
	fc:add( nn.Linear( featSize, hiddenSize ) )
	fc:add( nn.ReLU(  ) )
	fc:add( nn.Dropout( dropout ) )
	fc:cuda(  )
	-- Create FC classifier.
	-- In:  ( numVideo X seqLength ), hiddenSize
	-- Out: ( numVideo X seqLength ), numClass
	local classifierFc = nn.Sequential(  )
	classifierFc:add( nn.Linear( hiddenSize, numClass ) )
	classifierFc:add( nn.LogSoftMax(  ) )
	classifierFc:cuda(  )
	-- Combine sub models.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), numClass
	local model = nn.Sequential(  )
	model:add( features )
	model:add( fc )
	model:add( classifierFc )
	model:cuda(  )
	-- Check options.
	assert( self.opt.dropout <= 1 and self.opt.dropout >= 0 )
	assert( self.opt.numOut == 1 )
	assert( self.opt.caffeInput )
	assert( not self.opt.normalizeStd )
	assert( not self.opt.keepAspect )
	return model
end
function task:defineCriterion(  )
	return nn.ClassNLLCriterion(  )
end
function task:groupParams( model )
	local params, grads, optims = {  }, {  }, {  }
	params[ 1 ], grads[ 1 ] = model.modules[ 1 ]:getParameters(  ) -- Features.
	params[ 2 ], grads[ 2 ] = model.modules[ 2 ]:getParameters(  ) -- FC.
	params[ 3 ], grads[ 3 ] = model.modules[ 3 ]:getParameters(  ) -- Classifier.
	optims[ 1 ] = { -- Features.
		learningRate = self.opt.learnRate[ 1 ],
		learningRateDecay = 0.0,
		momentum = self.opt.momentum,
		dampening = 0.0,
		weightDecay = self.opt.weightDecay 
	}
	optims[ 2 ] = { -- FC.
		learningRate = self.opt.learnRate[ 2 ],
		learningRateDecay = 0.0,
		momentum = self.opt.momentum,
		dampening = 0.0,
		weightDecay = self.opt.weightDecay 
	}
	optims[ 3 ] = { -- Classifier.
		learningRate = self.opt.learnRate[ 3 ],
		learningRateDecay = 0.0,
		momentum = self.opt.momentum,
		dampening = 0.0,
		weightDecay = self.opt.weightDecay 
	}
	return params, grads, optims
end
function task:getBatchTrain(  )
	local batchSize = self.opt.batchSize
	local seqLength = self.opt.seqLength
	local cropSize = self.opt.cropSize
	local numVideoToSample = batchSize / seqLength
	local input = torch.Tensor( batchSize, 3, cropSize, cropSize )
	local label = torch.LongTensor( batchSize )
	local numVideo = self.dbtr.vid2path:size( 1 )
	local fcnt = 0
	for v = 1, numVideoToSample do
		local vid = torch.random( 1, numVideo )
		local vpath = ffi.string( torch.data( self.dbtr.vid2path[ vid ] ) )
		local numFrame = self.dbtr.vid2numim[ vid ]
		local cid = self.dbtr.vid2cid[ vid ]
		local startFrame = torch.random( 1, math.max( 1, numFrame - seqLength + 1 ) )
		local rw = torch.uniform(  )
		local rh = torch.uniform(  )
		local rf = torch.uniform(  )
		for f = 1, seqLength do
			local fid = math.min( numFrame, startFrame + f - 1 )
			local fpath = paths.concat( vpath, string.format( self.dbtr.frameFormat, fid ) )
			fcnt = fcnt + 1
			input[ fcnt ]:copy( self:processImageTrain( fpath, rw, rh, rf ) )
			label[ fcnt ] = cid
		end
	end
	return input, label
end
function task:getBatchVal( fidStart )
	local seqLength = self.opt.seqLength
	local batchSize = self.opt.batchSize
	local cropSize = self.opt.cropSize
	local vidStart = ( fidStart - 1 ) / seqLength + 1
	local numVideoToSample = batchSize / seqLength
	local input = torch.Tensor( batchSize, 3, cropSize, cropSize )
	local label = torch.LongTensor( batchSize )
	local numVideo = self.dbval.vid2path:size( 1 )
	local fcnt = 0
	for v = 1, numVideoToSample do
		local vid = vidStart + v - 1
		local vpath = ffi.string( torch.data( self.dbval.vid2path[ vid ] ) )
		local numFrame = self.dbval.vid2numim[ vid ]
		local cid = self.dbval.vid2cid[ vid ]
		local startFrame = math.floor( math.max( 0, numFrame - seqLength ) / 2 ) + 1
		for f = 1, seqLength do
			local fid = math.min( numFrame, startFrame + f - 1 )
			local fpath = paths.concat( vpath, string.format( self.dbval.frameFormat, fid ) )
			fcnt = fcnt + 1
			input[ fcnt ]:copy( self:processImageVal( fpath ) )
			label[ fcnt ] = cid
		end
	end
	return input, label
end
function task:evalBatch( fid2out, fid2label )
	if type( fid2out ) ~= 'table' then fid2out = { fid2out } end
	local numOut = #fid2out
	local seqLength = self.opt.seqLength
	local batchSize = self.opt.batchSize
	local numVideo = fid2out[ 1 ]:size( 1 ) / seqLength
	local pooling = self.opt.videoPool
	local oid2eval = torch.Tensor( numOut ):fill( 0 )
	assert( batchSize == fid2label:numel(  ) )
	assert( numVideo == batchSize / seqLength )
	assert( numOut == self.opt.numOut )
	for oid, out in pairs( fid2out ) do
		local top1 = 0
		for v = 1, numVideo do
			local fs = ( v - 1 ) * seqLength + 1
			local fe = fs + seqLength - 1
			assert( fid2label[ fs ] == fid2label[ fe ] )
			local cid2score
			if pooling == 'sum' then
				cid2score = out[ { { fs, fe }, {  } } ]:mean( 1 )
			elseif pooling == 'max' then
				cid2score = out[ { { fs, fe }, {  } } ]:max( 1 )
			end
			local _, rank2cid = cid2score:float(  ):sort( 2, true )
			if rank2cid[ 1 ][ 1 ] == fid2label[ fs ] then
				top1 = top1 + 1
			end
		end
		oid2eval[ oid ] = top1 * 100 / numVideo
	end
	return oid2eval
end
--------------------------------------------------
-------- TASK-SPECIFIC INTERNAL FUNCTIONS --------
--------------------------------------------------
require 'image'
function task:processImageTrain( path, rw, rh, rf )
	collectgarbage(  )
	local input = self:loadImage( path )
	local iW = input:size( 3 )
	local iH = input:size( 2 )
	-- Do random crop.
	local oW = self.opt.cropSize
	local oH = self.opt.cropSize
	local h1 = math.ceil( ( iH - oH ) * rh )
	local w1 = math.ceil( ( iW - oW ) * rw )
	if iH == oH then h1 = 0 end
	if iW == oW then w1 = 0 end
	local out = image.crop( input, w1, h1, w1 + oW, h1 + oH )
	assert( out:size( 3 ) == oW )
	assert( out:size( 2 ) == oH )
	-- Do horz-flip.
	if rf > 0.5 then out = image.hflip( out ) end
	-- Normalize.
	out = self:normalizeImage( out )
	return out
end
function task:processImageVal( path )
	collectgarbage(  )
	local input = self:loadImage( path )
	local iW = input:size( 3 )
	local iH = input:size( 2 )
	-- Do central crop.
	local oW = self.opt.cropSize
	local oH = self.opt.cropSize
	local h1 = math.ceil( ( iH - oH ) / 2 )
	local w1 = math.ceil( ( iW - oW ) / 2 )
	if iH == oH then h1 = 0 end
	if iW == oW then w1 = 0 end
	local out = image.crop( input, w1, h1, w1 + oW, h1 + oH )
	assert( out:size( 3 ) == oW )
	assert( out:size( 2 ) == oH )
	-- Normalize.
	out = self:normalizeImage( out )
	return out
end
function task:resizeImage( im )
	local imageSize = self.opt.imageSize
	if self.opt.keepAspect then
		if input:size( 3 ) < input:size( 2 ) then
			im = image.scale( im, imageSize, imageSize * im:size( 2 ) / im:size( 3 ) )
		else
			im = image.scale( im, imageSize * im:size( 3 ) / im:size( 2 ), imageSize )
		end
	else
		im = image.scale( im, imageSize, imageSize )
	end
	return im
end
function task:loadImage( path )
	local im = image.load( path, 3, 'float' )
	im = self:resizeImage( im )
	if self.opt.caffeInput then
		im = im * 255
		im = im:index( 1, torch.LongTensor{ 3, 2, 1 } )
	end
	return im
end
function task:normalizeImage( im )
	for i = 1, 3 do
		if self.inputStat.mean then im[ i ]:add( -self.inputStat.mean[ i ] ) end
		if self.inputStat.std and self.opt.normalizeStd then im[ i ]:div( self.inputStat.std[ i ] ) end
	end
	return im
end
function task:tableToTensor( inputTable, labelTable )
	local inputTensor, labelTensor
	local quantity = #labelTable
	assert( inputTable[ 1 ]:dim(  ) == 3 )
	inputTensor = torch.Tensor( quantity, 3, self.opt.cropSize, self.opt.cropSize )
	labelTensor = torch.LongTensor( quantity ):fill( 0 )
	for i = 1, #inputTable do
		inputTensor[ i ]:copy( inputTable[ i ] )
		labelTensor[ i ] = labelTable[ i ]
	end
	return inputTensor, labelTensor
end
