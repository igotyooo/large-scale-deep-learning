torch.setdefaulttensortype( 'torch.FloatTensor' )
paths.dofile( 'util.lua' )
paths.dofile( 'setpath.lua' )

-- Define task.
assert( arg[ 1 ] == '-task', 'Specify a defined task name.' )
local taskFile = paths.concat( 'task', arg[ 2 ] .. '.lua' )
paths.dofile( taskFile )

-- Set task manager.
local task = TaskManager(  )
task:setOption( arg )
task:setDb(  )
task:setInputStat(  )

-- Get necessary data. 
local opt = task:getOption(  )
local model, se = task:getModel(  )
local fun1, fun2, fun3 = task:getFunctionTest(  )
local numQuery = task:getNumQuery(  )

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

-- Set test.
test = paths.dofile( 'test.lua' )
test.setOption( opt, numQuery )
test.setModel( model )
test.setDonkey( donkeys )
test.setFunction( fun1, fun2, fun3 )

-- Do the job.
test.test( 0 )
