require 'paths'
require 'nn'
require 'optim'
require 'cutorch'
require 'cudnn'
require 'cunn'

cutorch.setDevice(1)
trainset = torch.load('data/train_32x32.t7','ascii')
testset = torch.load('data/train_32x32.t7','ascii')

setmetatable(trainset,
	{__index = function(t,i)
	return {t.data[i]:cuda(),t.labels[i]}
	end}
);
trainset.data = trainset.data:double()
testset.data = testset.data:double()
function trainset:size()
	return self.data:size(1)
end

mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,1 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end
testset.data = testset.data:cuda()

function create_net()
	net = nn.Sequential()
	net:add(cudnn.SpatialConvolution(1,6,5,5))
	net:add(cudnn.ReLU())
	net:add(cudnn. SpatialMaxPooling(2,2,2,2))
	net:add(cudnn.SpatialConvolution(6,16,5,5))
	net:add(cudnn.ReLU())
	net:add(cudnn.SpatialMaxPooling(2,2,2,2))
	net:add(nn.View(16*5*5))
	net:add(nn.Linear(16*5*5,120))
	net:add(cudnn.ReLU())                       -- non-linearity 
	net:add(nn.Linear(120, 84))
	net:add(cudnn.ReLU())                       -- non-linearity 
	net:add(nn.Linear(84, 10))                   -- 10 is the number of outputs of the network (in this case, 10 digits)
	net:add(cudnn.LogSoftMax())     
	return(net:cuda())
end                -- converts the output to a log-probability. Useful for classification problems

--Loss Function
criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()
--Training Neural Network

function sgd(x, dx,	lr)
	x:add(-lr,dx)
end

function accuracy(testset,net)
	correct = 0
	for i=1,10000 do
		local groundtruth = testset.labels[i]
		local prediction = net:forward(testset.data[i])
		local max_prob, indices = torch.sort(prediction,true)  -- true means sorting in descending order
		if groundtruth == indices[1] then
			correct = correct + 1
		end
	end
	print(correct, 100*correct/10000 .. '%')
end

function class_performances(testset,net)
	class_performance = {0,0,0,0,0,0,0,0,0,0}
	total_classes = {0,0,0,0,0,0,0,0,0,0}
	for i=1,10000 do
		local groundtruth = testset.labels[i]
		total_classes[testset.labels[i]] = total_classes[testset.labels[i]] + 1
		local prediction = net:forward(testset.data[i])
		local max_prob, indices = torch.sort(prediction, true)
		if groundtruth == indices[1] then
			class_performance[groundtruth] = class_performance[groundtruth] + 1
		end
	end
	for i= 1, 10 do
		print (i, 100*class_performance[i]/total_classes[i])
	end
end

function train(trainset,batch_size,epoch,net,currentLearningRate)
    params,grad_params = net:getParameters()
    for l=1,epoch do  
        time_start = sys.clock()
        for t = 1,trainset:size()/batch_size do
        	currentError = 0
        	grad_params:zero()
        	local input = torch.zeros(batch_size,1,32,32)
        	local target = torch.zeros(batch_size)
        	input = input:cuda()
        	for s = 1,batch_size do
    	        local example = trainset[(t-1)*batch_size+s];
    	        input[s] = example[1];
    	        target[s] = example[2];
    	    end
            error = criterion:forward(net:forward(input), target);
            currentError = currentError + error
            net:backward(input, criterion:backward(net.output, target));
            sgd(params,grad_params,currentLearningRate)
            xlua.progress(t,trainset:size()/batch_size)
        end
        time_end = sys.clock()
        accuracy(testset,net)
        class_performances(testset,net)
    end
    return(time_end-time_start)
end

logger = optim.Logger('./benchmark.log')
logger:setNames{'Batch Size', 'Time Taken in s'}

batch_size = torch.Tensor({1,2,5,10,20,50,100,200,500,1000})
epoch = 1
time_taken = torch.Tensor(10)
learning_rate = 0.01

for i=1,batch_size:size()[1] do 
    time_taken[i] = train(trainset,batch_size[i],epoch,create_net(),learning_rate)
    logger:add{batch_size[i],time_taken[i]}
end

logger:style{'+-', '+-'}






torch.save('mnist_cnn_trained.t7',net)             








