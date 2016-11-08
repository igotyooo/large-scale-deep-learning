torch.setdefaulttensortype( 'torch.FloatTensor' )
paths.dofile( 'util.lua' )
paths.dofile( 'setpath.lua' )

-- Set task manager.
assert( arg[ 1 ] == '-task', 'Specify a defined task name.' )
local taskFile = paths.concat( 'task', arg[ 2 ] .. '.lua' )
paths.dofile( taskFile )
local task = TaskManager(  )
task:setOption( arg )
task:createDb(  )
task:estimateInputStat(  )
local opt = task:getOption(  )
local model, se = task:getModel(  )

-- Hire donkeys working for data loading.
-- This is modified from Soumith's data.lua.
local Threads = require 'threads'
local donkeys = {  }
Threads.serialization( 'threads.sharedserialize' )
if opt.numDonkey > 0 then
	donkeys = Threads(
		opt.numDonkey,
		function(  )
			paths.dofile( taskFile )
		end,
		function( tid )
			local seed = ( se - 1 ) * 32 + tid
			torch.manualSeed( seed )
			torch.setnumthreads( 1 )
			print( string.format( 'DONKEY) Start donkey %d with seed %d.', tid, seed ) )
		end
	)
else
	function donkeys:addjob( f1, f2 ) f2( f1(  ) ) end
	function donkeys:synchronize(  ) end
	torch.manualSeed( se )
end
donkeys:synchronize(  ) 

-- Set train.
train = paths.dofile( 'train.lua' )
train.setOption( opt )
train.setModel( model )
train.setDonkey( donkeys )
train.setFunction( task:getFunctionTrain(  ) )

-- Set val.
val = paths.dofile( 'val.lua' )
val.setOption( opt, task:getNumVal(  ) )
val.setModel( model )
val.setDonkey( donkeys )
val.setFunction( task:getFunctionVal(  ) )

-- Do the job.
for e = se, opt.numEpoch do
	train.train( e )
	val.val( e )
end
