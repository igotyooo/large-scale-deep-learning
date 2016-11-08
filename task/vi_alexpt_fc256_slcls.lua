require 'image'
local ffi = require 'ffi'
local task = torch.class( 'TaskManager' )
-------------------------------------
-------- INTERFACE FUNCTIONS --------
-------------------------------------
function task:__init(  )
	self.name = 'vi_alexpt_fc256_slcls'
	self.opt = {  }
	self.dbtr = {  }
	self.dbval = {  }
	self.mean = 0
	self.std = 0
end
function task:setOption( arg )
	assert( self.name == arg[ 2 ] )
	local cmd = torch.CmdLine(  )
	cmd:option( '-task', self.name )
	-- System.
	cmd:option( '-numGpu', 4, 'Number of GPUs.' )
	cmd:option( '-backend', 'cudnn', 'cudnn or nn.' )
	cmd:option( '-numDonkey', 4, 'Number of donkeys for data loading.' )
	-- Data. 
	cmd:option( '-data', 'UCF101', 'Name of dataset defined in "./db/"' )
	cmd:option( '-imageSize', 240, 'Short side of initial resize.' )
	cmd:option( '-cropSize', 224, 'Size of random square crop.' )
	cmd:option( '-keepAspect', 0, '1 for keep, 0 for no.' )
	cmd:option( '-normalizeStd', 0, '1 for normalize piexel std to 1, 0 for no.' )
	cmd:option( '-seqLength', 16, 'Number of frames per input video' )
	cmd:option( '-caffeInput', 1, '1 for caffe input, 0 for no.' )
	-- Train.
	cmd:option( '-numEpoch', 32, 'Number of total epochs to run.' )
	cmd:option( '-epochSize', 2048, 'Number of batches per epoch.' )
	cmd:option( '-batchSize', 256, 'Frame-level mini-batch size.' )
	cmd:option( '-learnRate', 1e-3, 'Supports multi-lr for multi-module like "lr1,lr2,lr3".' )
	cmd:option( '-momentum', 0.9, 'Momentum.' )
	cmd:option( '-weightDecay', 5e-4, 'Weight decay.' )	
	cmd:option( '-startFrom', '', 'Path to the initial model. Using it for LR decay is recommended.' )
	cmd:option( '-numOut', 1, 'Number of outputs from net.' )
	-- Value processing.
	local opt = cmd:parse( arg or {  } )
	opt.normalizeStd = opt.normalizeStd > 0
	opt.keepAspect = opt.keepAspect > 0
	opt.caffeInput = opt.caffeInput > 0
	-- Set dst paths.
	local dirRoot = paths.concat( gpath.dataout, opt.data )
	local pathDbTrain = paths.concat( dirRoot, 'dbTrain.t7' )
	local pathDbVal = paths.concat( dirRoot, 'dbVal.t7' )
	local pathImStat = paths.concat( dirRoot, 'inputStats.t7' )
	if opt.caffeInput then pathImStat = pathImStat:match( '(.+).t7$' ) .. 'Caffe.t7' end
	local ignore = { numGpu=true, backend=true, numDonkey=true, data=true, numEpoch=true, startFrom=true }
	local dirModel = paths.concat( dirRoot, cmd:string( self.name, opt, ignore ) )
	if opt.startFrom ~= '' then
		local baseDir, epoch = opt.startFrom:match( '(.+)/model_(%d+).t7' )
		dirModel = paths.concat( baseDir, cmd:string( 'model_' .. epoch, opt, ignore ) )
	end
	opt.pathDbTrain = pathDbTrain
	opt.pathDbVal = pathDbVal
	opt.pathImStat = pathImStat
	opt.dirModel = dirModel
	opt.pathModel = paths.concat( opt.dirModel, 'model_%03d.t7' )
	opt.pathOptim = paths.concat( opt.dirModel, 'optimState_%03d.t7' )
	opt.pathTrainLog = paths.concat( opt.dirModel, 'train.log' )
	opt.pathValLog = paths.concat( opt.dirModel, 'val.log' )
	opt.pathVideoLevelValLog = paths.concat( opt.dirModel, 'video_level_test_of_%03d' )
	paths.mkdir( dirRoot )
	paths.mkdir( dirModel )
	self.opt = opt
	-- Verification.
	assert( opt.imageSize >= opt.cropSize )
	assert( opt.batchSize % opt.seqLength == 0 )
	assert( opt.numOut > 0 )
end
function task:getOption(  )
	return self.opt
end
function task:createDb(  )
	paths.dofile( string.format( '../db/%s.lua', self.opt.data ) )
	if paths.filep( self.opt.pathDbTrain ) then
		self:print( 'Load train db.' )
		self.dbtr = torch.load( self.opt.pathDbTrain )
		self:print( 'Done.' )
	else
		self:print( 'Create train db.' )
		self.dbtr.vid2path,
		self.dbtr.vid2numim,
		self.dbtr.vid2cid,
		self.dbtr.cid2name,
		self.dbtr.frameFormat = genDb( 'train' )
		torch.save( self.opt.pathDbTrain, self.dbtr )
		self:print( 'Done.' )
	end
	local numTrain = self.dbtr.vid2path:size( 1 )
	local numClass = self.dbtr.cid2name:size( 1 )
	self:print( string.format( 'Train: %d videos, %d classes.', numTrain, numClass ) )
	if paths.filep( self.opt.pathDbVal ) then
		self:print( 'Load val db.' )
		self.dbval = torch.load( self.opt.pathDbVal )
		self:print( 'Done.' )
	else
		self:print( 'Create val db.' )
		self.dbval.vid2path,
		self.dbval.vid2numim,
		self.dbval.vid2cid,
		self.dbval.cid2name,
		self.dbval.frameFormat = genDb( 'val' )
		torch.save( self.opt.pathDbVal, self.dbval )
		self:print( 'Done.' )
	end
	local numVal = self.dbval.vid2path:size( 1 )
	local numClass = self.dbval.cid2name:size( 1 )
	self:print( string.format( 'Val: %d videos, %d classes.', numVal, numClass ) )
	-- Verification.
	assert( self.dbtr.vid2path:size( 1 ) == self.dbtr.vid2numim:numel(  ) )
	assert( self.dbtr.vid2path:size( 1 ) == self.dbtr.vid2cid:numel(  ) )
	assert( self.dbtr.cid2name:size( 1 ) == self.dbtr.vid2cid:max(  ) )
	assert( self.dbval.vid2path:size( 1 ) == self.dbval.vid2numim:numel(  ) )
	assert( self.dbval.vid2path:size( 1 ) == self.dbval.vid2cid:numel(  ) )
	assert( self.dbval.cid2name:size( 1 ) == self.dbval.vid2cid:max(  ) )
	assert( self.dbtr.cid2name:size( 1 ) == self.dbval.vid2cid:max(  ) )
end
function task:getNumVal(  )
	return self.dbval.vid2numim:numel(  ) * self.opt.seqLength
end
function task:estimateInputStat(  )
	if paths.filep( self.opt.pathImStat ) then
		local meanstd = torch.load( self.opt.pathImStat )
		self.mean = meanstd.mean
		self.std = meanstd.std
		self:print( 'Loaded mean and std.' )
	else
		local numIm = 10000
		local batchSize = self.opt.batchSize
		local seqLength = self.opt.seqLength
		local numBatch = math.ceil( numIm / batchSize )
		self.opt.seqLength = 1
		self:print( string.format( 'Estimate RGB mean and std over %d images.', numIm ) )
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
		self.mean = meanEstimate
		self.std = stdEstimate
		local cache = { mean = self.mean, std = self.std }
		torch.save( self.opt.pathImStat, cache )
		self:print( 'Done.' )
	end
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
	return modelSet, startEpoch
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
------------------------------------
-------- INTERNAL FUNCTIONS --------
------------------------------------
function task:defineModel(  )
	require 'loadcaffe'
	-- Set params.
	local featSize = 4096
	local hiddenSize = 256
	local numCls = self.dbtr.cid2name:size( 1 )
	-- Load pre-trained CNN.
	-- In:  ( numVideo X seqLength ), 3, 224, 224
	-- Out: ( numVideo X seqLength ), featSize
	self:print( 'Load pre-trained Caffe feature.' )
	local proto = gpath.net.alex_caffe_proto
	local caffemodel = gpath.net.alex_caffe_model
	local features = loadcaffe.load( proto, caffemodel, self.opt.backend )
	features:remove(  )
	features:remove(  )
	features:remove(  )
	features:remove(  )
	features:remove(  )
	features:cuda(  )
	features = makeDataParallel( features, self.opt.numGpu, 1 )
	-- Create FC tower.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), hiddenSize
	local fc = nn.Sequential(  )
	fc:add( nn.Linear( featSize, hiddenSize ) )
	fc:add( nn.Tanh(  ) )
	fc:add( nn.Dropout( 0.5 ) )
	fc:cuda(  )
	-- Create FC classifier.
	-- In:  ( numVideo X seqLength ), featSize
	-- Out: ( numVideo X seqLength ), numClass
	local classifierFc = nn.Sequential(  )
	classifierFc:add( nn.Linear( hiddenSize, numCls ) )
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
		learningRate = self.opt.lrFeature,
		learningRateDecay = 0.0,
		momentum = self.opt.momentum,
		dampening = 0.0,
		weightDecay = self.opt.weightDecay 
	}
	optims[ 2 ] = { -- FC.
		learningRate = self.opt.lrFc,
		learningRateDecay = 0.0,
		momentum = self.opt.momentum,
		dampening = 0.0,
		weightDecay = self.opt.weightDecay 
	}
	optims[ 3 ] = { -- Classifier.
		learningRate = self.opt.lrClassifier,
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
	local numVideoToSample = batchSize / seqLength
	local inputTable = {  }
	local labelTable = {  }
	local numVideo = self.dbtr.vid2path:size( 1 )
	for v = 1, numVideoToSample do
		local vid = torch.random( 1, numVideo )
		local vpath = ffi.string( torch.data( self.dbtr.vid2path[ vid ] ) )
		local numFrame = self.dbtr.vid2numim[ vid ]
		local cid = self.dbtr.vid2cid[ vid ]
		local startFrame = torch.random( 1, math.max( 1, numFrame - seqLength + 1 ) )
		local rw, rh, rf
		for f = 1, seqLength do
			local fid = math.min( numFrame, startFrame + f - 1 )
			local fpath = paths.concat( vpath, string.format( self.dbtr.frameFormat, fid ) )
			if f == 1 then
				rw = torch.uniform(  )
				rh = torch.uniform(  )
				rf = torch.uniform(  )
			end
			local out = self:processImageTrain( fpath, rw, rh, rf )
			table.insert( inputTable, out )
			table.insert( labelTable, cid )
		end
	end
	local inputTensor, labelTensor = self:tableToTensor( inputTable, labelTable )
	return inputTensor, labelTensor
end
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
function task:evalBatch( fid2out, fid2gt )
	if type( fid2out ) ~= 'table' then fid2out = { fid2out } end
	local numOut = #fid2out
	local seqLength = self.opt.seqLength
	local batchSize = fid2out[ 1 ]:size( 1 )
	local numVideo = batchSize / seqLength
	local oid2eval = torch.Tensor( numOut ):fill( 0 )
	assert( numOut == self.opt.numOut )
	assert( batchSize == self.opt.batchSize )
	for oid, out in pairs( fid2out ) do
		local _, fid2pcid = out:float(  ):sort( 2, true )
		local vid2true = torch.zeros( numVideo, 1 )
		local top1 = 0
		for v = 1, numVideo do
			local fbias = ( v - 1 ) * seqLength
			local pcid2num = torch.zeros( out:size( 2 ) )
			local cid = fid2gt[ fbias + 1 ]
			for f = 1, seqLength do
				local fid = fbias + f
				local pcid = fid2pcid[ fid ][ 1 ]
				pcid2num[ pcid ] = pcid2num[ pcid ] + 1
			end
			local _, rank2pcid = pcid2num:sort( true )
			if cid == rank2pcid[ 1 ] then top1 = top1 + 1 end
		end
		top1 = top1 * 100 / numVideo
		oid2eval[ oid ] = top1
	end
	return oid2eval
end
function task:getBatchVal( fidStart )
	local seqLength = self.opt.seqLength
	local batchSize = self.opt.batchSize
	local vidStart = ( fidStart - 1 ) / seqLength + 1
	local inputTable = {  }
	local labelTable = {  }
	local numVideoToSample = batchSize / seqLength
	local numVideo = self.dbval.vid2path:size( 1 )
	for v = 1, numVideoToSample do
		local vid = vidStart + v - 1
		local vpath = ffi.string( torch.data( self.dbval.vid2path[ vid ] ) )
		local numFrame = self.dbval.vid2numim[ vid ]
		local cid = self.dbval.vid2cid[ vid ]
		local startFrame = torch.random( 1, math.max( 1, numFrame - seqLength + 1 ) )
		for f = 1, seqLength do
			local fid = math.min( numFrame, startFrame + f - 1 )
			local fpath = paths.concat( vpath, string.format( self.dbval.frameFormat, fid ) )
			local out = self:processImageVal( fpath )
			table.insert( inputTable, out )
			table.insert( labelTable, cid )
		end
	end
	local inputTensor, labelTensor = self:tableToTensor( inputTable, labelTable )
	return inputTensor, labelTensor
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
		if self.mean ~= 0 then im[ i ]:add( -self.mean[ i ] ) end
		if self.std ~= 0 and self.opt.normalizeStd then im[ i ]:div( self.std[ i ] ) end
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
function task:print( str )
	print( 'TASK MANAGER) ' .. str )
end
